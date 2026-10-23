"""Fit the MPC model to measured data, from a four-port, OC and WD, or sym and asym.

All inputs are turned into a weighted residual vector through the forward model of their own
setup, so the optimizers do not care where the data came from.

Misfit. Complex log ratio ln(model/data). Its real part is the magnitude error in nepers and
its imaginary part the phase error in radians. Each residual is divided by an assumed error
sigma = max(rel_floor, abs_floor / |data|), so points near the VNA noise floor carry little
weight. The floors are assumptions, set them for your instrument.

Parameters are fitted in log space, so every parameter stays positive and scales are comparable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from .mpc import EV, EV_INV, MPCParams, modal_admittances
from .setups import s21, y_from_s

# ----------------------------------------------------------------------------
# Model variants and parameter handling
# ----------------------------------------------------------------------------

DEFAULT_BOUNDS = dict(
    LC=(1e-7, 1.0), CC=(1e-14, 1e-8), RC=(1e1, 1e9),
    LD=(1e-9, 1e-3), CD=(1e-14, 1e-8), RD=(1e0, 1e8),
    CW=(1e-14, 1e-8), Lw=(1e-10, 1e-4), Cwt=(1e-14, 1e-8), Rw=(1e0, 1e8),
    Rcr=(1e0, 1e8), Ccr=(1e-14, 1e-9), Rrl=(1e-1, 1e7), Lrl=(1e-9, 1e-3), Cg=(1e-15, 1e-10), Rdc=(1e-3, 1e2),
)


@dataclass(frozen=True)
class Variant:
    """Which optional parts of the model are fitted.

    lcr_w  parallel LCR tank in series with CW, needed when Csym = 2 CD - CW is negative
    cr     number of R-C branches across each CM winding (Foster II, eq. (3) of the paper)
    rl     number of R-L branches across each CM winding (frequency dependent permeability)
    cg     capacitance from each port to ground, only observable from a four-port
    rdc    winding resistance in series with the C, D and W modes, matters for small SMD chokes
    """
    lcr_w: bool = True
    cr: int = 1
    rl: int = 0
    cg: bool = True
    rdc: bool = False   # winding resistance, an extension of the paper model

    def names(self) -> list[str]:
        n = ["LC", "CC", "RC", "LD", "CD", "RD", "CW"]
        if self.lcr_w:
            n += ["Lw", "Cwt", "Rw"]
        n += [f"Rcr{i or ''}" for i in range(self.cr)] + [f"Ccr{i or ''}" for i in range(self.cr)]
        n += [f"Rrl{i or ''}" for i in range(self.rl)] + [f"Lrl{i or ''}" for i in range(self.rl)]
        if self.cg:
            n += ["Cg"]
        if self.rdc:
            n += ["Rdc"]
        return n

    def build(self, v: dict[str, float], name: str = "") -> MPCParams:
        g = lambda k, i: v[f"{k}{i or ''}"]
        return MPCParams(
            LC=v["LC"], CC=v["CC"], RC=v["RC"], LD=v["LD"], CD=v["CD"], RD=v["RD"], CW=v["CW"],
            lcr_w=(v["Lw"], v["Cwt"], v["Rw"]) if self.lcr_w else None,
            cr_branches=[(g("Rcr", i), g("Ccr", i)) for i in range(self.cr)],
            rl_branches=[(g("Rrl", i), g("Lrl", i)) for i in range(self.rl)],
            Cg=v.get("Cg", 0.0), Rdc=v.get("Rdc", 0.0), name=name)


def _bound(name: str, bounds: dict) -> tuple[float, float]:
    base = name.rstrip("0123456789") if name not in bounds else name
    return bounds[name] if name in bounds else bounds[base]


# ----------------------------------------------------------------------------
# Measurements
# ----------------------------------------------------------------------------

def _sigma(data, rel_floor, abs_floor):
    return np.maximum(rel_floor, abs_floor / np.abs(data))


@dataclass
class FourPort:
    """Measured four-port S matrix, shape (N, 4, 4). Fits the four modal admittances.

    Needs a mirror-symmetric choke. Cg is observable here and nowhere else.
    mode_weights lets you down-weight modes that are noise limited, for example W and G at
    low frequency.
    """
    f: np.ndarray
    S: np.ndarray
    z0: float = 50.0
    f_range: tuple[float, float] = (1e6, 1e9)
    abs_floor: float = 2e-5      # S, on the admittance
    rel_floor: float = 0.02
    mode_weights: dict = field(default_factory=lambda: dict(G=0.5, C=1.0, W=1.0, D=1.0))

    def __post_init__(self):
        b = (self.f >= self.f_range[0]) & (self.f <= self.f_range[1])
        self.f = np.asarray(self.f)[b]
        Y = y_from_s(np.asarray(self.S)[b], self.z0)
        self.data = {m: np.einsum("ij,njk,ki->ni", EV_INV, Y, EV)[:, i] for i, m in enumerate("GCWD")}
        self.sigma = {m: _sigma(self.data[m], self.rel_floor, self.abs_floor) for m in "GCWD"}

    @property
    def observable(self):
        return {"G", "C", "W", "D"}

    def residuals(self, p: MPCParams, use_cg: bool) -> np.ndarray:
        mod = modal_admittances(self.f, p)
        out = []
        for m in "GCWD":
            w = self.mode_weights[m] * (0.0 if (m == "G" and not use_cg) else 1.0)
            if w == 0:
                continue
            lr = np.log(mod[m] / self.data[m])
            out += [w * lr.real / self.sigma[m], w * lr.imag / self.sigma[m]]
        return np.concatenate(out)


@dataclass
class SetupCurves:
    """S21 curves of two or more setups, for example {"oc": ..., "wd": ...} or {"asym": ..., "sym": ...}.

    Curves may be complex, which also fits the phase, or real magnitudes in linear units.
    Use from_db for magnitudes given in dB. OC and WD together identify YC, YD and YW. The
    asym and sym pair identifies YC and YD - YW only, so CW is not separable from CD.
    """
    f: np.ndarray
    curves: dict
    abs_floor: float = 1e-4      # linear S21, about -80 dB
    rel_floor: float = 0.05
    f_range: tuple[float, float] = (1e5, 1.5e8)

    @classmethod
    def from_db(cls, f, curves_db: dict, **kw):
        return cls(f, {k: 10 ** (np.asarray(v) / 20) for k, v in curves_db.items()}, **kw)

    def __post_init__(self):
        f = np.asarray(self.f, float)
        b = (f >= self.f_range[0]) & (f <= self.f_range[1])
        self.f = f[b]
        self.curves = {k: np.asarray(v)[b] for k, v in self.curves.items()}
        self.sigma = {k: _sigma(v, self.rel_floor, self.abs_floor) for k, v in self.curves.items()}

    def residuals(self, p: MPCParams, use_cg: bool) -> np.ndarray:
        out = []
        for k, d in self.curves.items():
            lr = np.log(s21(k, self.f, p) / d) if np.iscomplexobj(d) else np.log(np.abs(s21(k, self.f, p)) / np.abs(d))
            out.append(lr.real / self.sigma[k])
            if np.iscomplexobj(d):
                out.append(lr.imag / self.sigma[k])
        return np.concatenate(out)


# ----------------------------------------------------------------------------
# Optimizers. Each takes (cost_fn, resid_fn, lo, hi, rng, **kw) and returns theta.
# ----------------------------------------------------------------------------

F_SCALE = 3.0


def _soft_l1_cost(r: np.ndarray) -> float:
    z = (r / F_SCALE) ** 2
    return float(np.sum(F_SCALE ** 2 * (np.sqrt(1 + z) - 1)))


def opt_multistart_ls(resid, lo, hi, rng, nstart=50, **_):
    best, costs = None, []
    for _i in range(nstart):
        try:
            r = least_squares(resid, rng.uniform(lo, hi), bounds=(lo, hi), loss="soft_l1", f_scale=F_SCALE, max_nfev=300)
        except Exception:
            continue
        costs.append(r.cost)
        if best is None or r.cost < best.cost:
            best = r
    return best.x, np.sort(costs)


def opt_differential_evolution(resid, lo, hi, rng, popsize=20, maxiter=300, **_):
    r = differential_evolution(lambda th: _soft_l1_cost(resid(th)), list(zip(lo, hi)), popsize=popsize,
                               maxiter=maxiter, seed=int(rng.integers(1 << 31)), tol=1e-8, polish=False)
    return r.x, np.array([r.fun])


OPTIMIZERS: dict[str, Callable] = {"least_squares": opt_multistart_ls, "de": opt_differential_evolution}


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

@dataclass
class FitResult:
    params: MPCParams
    names: list
    theta: np.ndarray           # log of the parameter values
    log_std: np.ndarray         # linearised, from the Jacobian, no truncation
    at_bound: list
    rel_singular_values: np.ndarray
    cost: float
    start_costs: np.ndarray
    method: str
    seconds: float
    nfev: int

    @property
    def hits(self) -> int:
        """Number of multistart runs that reached the best cost, within 1 percent."""
        c = np.asarray(self.start_costs)
        return int(np.sum(c <= c.min() * 1.01 + 1e-6)) if len(c) else 0

    def report(self) -> str:
        rows = []
        for n, th, sd, ab in zip(self.names, self.theta, self.log_std, self.at_bound):
            st = "determined" if sd < 0.3 else ("weak" if sd < 1 else "NOT determined")
            rows.append(f"  {n:6s} {np.exp(th):11.4g}  log-std {sd:9.3g}  {st}{'  AT BOUND' if ab else ''}")
        n = len(self.start_costs)
        note = f"\nbest cost reached by {self.hits} of {n} starts" if n > 1 else ""
        if n > 1 and self.hits == 1:
            note += ", only one. This may be a local minimum, increase nstart or use method='de'"
        return (f"method {self.method}, cost {self.cost:.4g}, {self.seconds:.1f} s{note}\n" + "\n".join(rows))


def fit(meas, variant: Variant = Variant(), method: str | Callable = "least_squares", bounds: dict | None = None,
        seed: int = 0, polish: bool = True, name: str = "", **opt_kw) -> FitResult:
    """Fit the MPC model to a FourPort or SetupCurves measurement."""
    if isinstance(meas, SetupCurves) and variant.cg:
        raise ValueError("Cg is not observable from setup curves. Use Variant(cg=False).")
    b = {**DEFAULT_BOUNDS, **(bounds or {})}
    names = variant.names()
    lo = np.log([_bound(n, b)[0] for n in names])
    hi = np.log([_bound(n, b)[1] for n in names])
    nfev = [0]

    def resid(th):
        nfev[0] += 1
        return meas.residuals(variant.build(dict(zip(names, np.exp(th))), name), variant.cg)

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    opt = OPTIMIZERS[method] if isinstance(method, str) else method
    th, costs = opt(resid, lo, hi, rng, **opt_kw)
    if polish:
        r2 = least_squares(resid, th, bounds=(lo, hi), loss="soft_l1", f_scale=F_SCALE, max_nfev=500)
        if _soft_l1_cost(resid(r2.x)) <= _soft_l1_cost(resid(th)):
            th = r2.x
    J = least_squares(resid, th, bounds=(lo, hi), max_nfev=1).jac
    _, s, vt = np.linalg.svd(J, full_matrices=False)
    std = np.sqrt(np.sum((vt.T / np.maximum(s, 1e-300)) ** 2, axis=1))
    at = [bool(np.isclose(t, l, atol=1e-2) or np.isclose(t, h, atol=1e-2)) for t, l, h in zip(th, lo, hi)]
    p = variant.build(dict(zip(names, np.exp(th))), name)
    return FitResult(p, names, th, std, at, s / s[0], _soft_l1_cost(resid(th)) / 2, costs,
                     method if isinstance(method, str) else getattr(method, "__name__", "custom"),
                     time.perf_counter() - t0, nfev[0])
