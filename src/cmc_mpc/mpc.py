"""Modal-parameters circuit (MPC) model of a two-winding common-mode choke.

Reference: A. Ojeda-Rodriguez, J. Bernal-Mendez, M. A. Martin-Prats,
"Modal Theory and Approach for Accurate Characterization of Common-Mode Chokes",
IEEE Trans. Power Electron., 38(9), 2023, Fig. 3, Table I, Table II, eq. (3).

Port numbering follows Fig. 1 of the paper.
    port 1 = L input, port 2 = L output, port 3 = N input, port 4 = N output.

Natural modes (Table I, eq. (12)), as port-voltage vectors (v1, v2, v3, v4):
    G = ( 1,  1,  1,  1)   all ports at the same potential, only ground coupling
    C = ( 1, -1,  1, -1)   CM current through both windings
    W = ( 1,  1, -1, -1)   potential difference between the windings, no winding current
    D = ( 1, -1, -1,  1)   DM current through both windings

All values in SI units (H, F, Ohm, Hz).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path

import numpy as np

# Columns are the eigenvectors G, C, W, D in port order (1, 2, 3, 4).
EV = np.array(
    [
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1],
    ],
    dtype=float,
)
EV_INV = EV.T / 4.0  # EV is orthogonal with columns of norm 2
MODE_NAMES = ("G", "C", "W", "D")


@dataclass
class MPCParams:
    """Parameters of the MPC model in Fig. 3.

    LC, CC, RC   CM block. Each winding of the CM block has self inductance LC/2,
                 the two windings are perfectly coupled, and CC and RC sit across
                 each winding. The CM inductance seen in the CISPR 17 asymmetrical
                 setup is therefore LC/2, which corresponds to the datasheet rating.
    LD, CD, RD   DM block. Four perfectly coupled LD/8 inductors, two per path.
                 CD and RD sit across each complete path.
    CW           interwinding capacitance, connected between the path midpoints.
    lcr_w        optional (L, C, R) parallel tank in series with CW (Fig. 4b, model B).
    cr_branches  optional list of (R, C) series branches across each CM winding (YRLC).
    rl_branches  optional list of (R, L) series branches across each CM winding (YRLC).
    Cg           capacitance from each port to ground. The paper uses Cg = 0.
    Rdc          extension, not in the paper. Winding resistance of each of the two paths, split
                 into Rdc/2 at each port. It is in series with the C, D and W modes, so
                 Y' = 1/(1/Y + Rdc/2), before Cg is added. Rdc = 0 gives the paper model.
    """

    LC: float
    CC: float
    RC: float
    LD: float
    CD: float
    RD: float
    CW: float
    lcr_w: tuple[float, float, float] | None = None
    cr_branches: list[tuple[float, float]] = field(default_factory=list)
    rl_branches: list[tuple[float, float]] = field(default_factory=list)
    Cg: float = 0.0
    Rdc: float = 0.0
    name: str = ""

    @property
    def C_sym(self) -> float:
        """Csym = 2 CD - CW. Negative values call for the LCR_W tank (model B)."""
        return 2.0 * self.CD - self.CW

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MPCParams":
        d = dict(d)
        if d.get("lcr_w") is not None:
            d["lcr_w"] = tuple(d["lcr_w"])
        d["cr_branches"] = [tuple(b) for b in d.get("cr_branches", [])]
        d["rl_branches"] = [tuple(b) for b in d.get("rl_branches", [])]
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def load_params(path: str | Path) -> dict[str, MPCParams]:
    """Load a JSON file mapping part names to parameter dictionaries."""
    with open(path) as fh:
        raw = json.load(fh)
    return {k: MPCParams.from_dict({**v, "name": k}) for k, v in raw.items() if not k.startswith("_")}


def _jw(f):
    return 2j * np.pi * np.asarray(f, dtype=float)


def y_rlc(f, p: MPCParams):
    """Optional Foster II branches across each CM winding, eq. (3)."""
    jw = _jw(f)
    y = np.zeros_like(jw)
    for R, C in p.cr_branches:
        y = y + 1.0 / (R + 1.0 / (jw * C))
    for R, L in p.rl_branches:
        y = y + 1.0 / (R + jw * L)
    return y


def _with_rdc(y, p: MPCParams):
    return y if p.Rdc == 0 else 1.0 / (1.0 / y + p.Rdc / 2.0)


def Y_C(f, p: MPCParams):
    """C-mode admittance, Table II and eq. (3), with the optional winding resistance."""
    jw = _jw(f)
    return _with_rdc(2.0 * (jw * p.CC + 1.0 / (jw * p.LC) + 1.0 / p.RC + y_rlc(f, p)), p)


def Y_D(f, p: MPCParams):
    """D-mode admittance, Table II, with the optional winding resistance."""
    jw = _jw(f)
    return _with_rdc(2.0 * (jw * p.CD + 1.0 / (jw * p.LD) + 1.0 / p.RD), p)


def Y_W(f, p: MPCParams):
    """W-mode admittance, Table II and Fig. 4."""
    jw = _jw(f)
    z = 1.0 / (jw * p.CW)
    if p.lcr_w is not None:
        L, C, R = p.lcr_w
        z = z + 1.0 / (1.0 / (jw * L) + jw * C + 1.0 / R)
    return _with_rdc(1.0 / z, p)


def Y_G(f, p: MPCParams):
    """G-mode admittance, Table II."""
    return _jw(f) * p.Cg


def modal_admittances(f, p: MPCParams) -> dict[str, np.ndarray]:
    """Eigenvalues of the port admittance matrix for the circuit of Fig. 3.

    A capacitance Cg from every port to ground adds jw Cg to the port admittance
    matrix as a multiple of the identity, so it shifts all four eigenvalues, not only
    YG. Table II lists jw Cg under the G mode only. The difference vanishes for
    Cg = 0, which is the case treated in the paper. See docs/decisions.md.
    """
    yg = Y_G(f, p)
    return {"G": yg, "C": Y_C(f, p) + yg, "W": Y_W(f, p) + yg, "D": Y_D(f, p) + yg}


def y_matrix(f, p: MPCParams) -> np.ndarray:
    """4x4 port admittance matrix, shape (len(f), 4, 4), built as EV diag(Y) EV^-1."""
    m = modal_admittances(np.atleast_1d(f), p)
    lam = np.stack([m[k] for k in MODE_NAMES], axis=-1)  # (N, 4)
    return np.einsum("ij,nj,jk->nik", EV, lam, EV_INV)
