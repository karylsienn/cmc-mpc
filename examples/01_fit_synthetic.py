"""Fit a synthetic choke with known parameters from three kinds of input and compare optimizers.

Run from the repo root:  python examples/01_fit_synthetic.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from cmc_mpc import FourPort, SetupCurves, Variant, fit
from cmc_mpc.circuit import mpc_netlist, port_y
from cmc_mpc.mpc import MPCParams
from cmc_mpc.setups import s21, s_from_y

F = np.logspace(5, 9, 401)
TRUTH = MPCParams(LC=7.1e-3, CC=3.1e-12, RC=35.3e3, LD=7.3e-6, CD=3.3e-12, RD=16.1e3, CW=2.9e-12,
                  lcr_w=(3e-7, 9e-12, 1e3), cr_branches=[(2e3, 4e-10)], Cg=4e-13)
S = s_from_y(port_y(mpc_netlist(TRUTH), F))
rng = np.random.default_rng(0)
S = S + 3e-5 * (rng.standard_normal(S.shape) + 1j * rng.standard_normal(S.shape))

fq = np.logspace(6, 9, 300)
def rms_db(p, k):
    return np.sqrt(np.mean((20*np.log10(abs(s21(k, fq, p))) - 20*np.log10(abs(s21(k, fq, TRUTH))))**2))

cases = {
    "four-port, least squares": (FourPort(F, S), Variant(cg=True), dict(method="least_squares", nstart=30)),
    "four-port, differential evolution": (FourPort(F, S), Variant(cg=True), dict(method="de", popsize=15, maxiter=200)),
    "OC + WD, least squares": (SetupCurves(F, {k: s21(k, F, TRUTH) for k in ("oc", "wd")}), Variant(cg=False), dict(nstart=30)),
    "asym + sym, least squares": (SetupCurves(F, {k: s21(k, F, TRUTH) for k in ("asym", "sym")}), Variant(cg=False), dict(nstart=30)),
}
print(f"{'input, optimizer':36s}{'time s':>8s}{'evals':>8s}" + "".join(f"{k:>8s}" for k in ("oc", "wd", "asym", "sym")) + "   rms error of S21 vs truth, dB")
for name, (m, v, kw) in cases.items():
    r = fit(m, v, seed=1, **kw)
    print(f"{name:36s}{r.seconds:8.1f}{r.nfev:8d}" + "".join(f"{rms_db(r.params, k):8.2f}" for k in ("oc", "wd", "asym", "sym")))
