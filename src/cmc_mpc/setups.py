"""Transmission coefficients of a CMC in the CISPR 17 and OC/WD measurement setups.

Two independent routes are provided.

1. Closed-form expressions in terms of modal admittances, eqs. (1), (2), (4), (5) of the paper.
2. A general termination solver that takes any 4x4 port admittance matrix and computes
   S21 for a given setup by nodal analysis. It is used to verify route 1 and will be
   reused for measured or simulated Y/S data.

All S21 values use the convention S21 = 2 V_load / V_source with equal source and load
resistance R, so that IL_dB = -20 log10 |S21|.
"""

from __future__ import annotations

import numpy as np

from .mpc import MPCParams, modal_admittances

R0 = 50.0


# ----------------------------------------------------------------------------
# Route 1. Closed-form modal expressions
# ----------------------------------------------------------------------------

def s21_asym(f, p: MPCParams, R: float = R0):
    """CISPR 17 asymmetrical (CM) setup, eq. (1)."""
    m = modal_admittances(f, p)
    YC, YG = m["C"], m["G"]
    return 2 * R * (YC - YG) / ((2 * R * YC + 1) * (2 * R * YG + 1))


def s21_sym(f, p: MPCParams, R: float = R0):
    """CISPR 17 symmetrical (DM) setup, eq. (2)."""
    m = modal_admittances(f, p)
    YD, YW = m["D"], m["W"]
    return 2 * R * (YD - YW) / ((R * YD + 2) * (R * YW + 2))


def s21_oc(f, p: MPCParams, R: float = R0):
    """OC setup of Fig. 5(a), eq. (4)."""
    m = modal_admittances(f, p)
    YC, YD = m["C"], m["D"]
    return 2 * R * YC * YD / (2 * R * YC * YD + YC + YD)


def s21_wd(f, p: MPCParams, R: float = R0):
    """WD setup of Fig. 5(b), eq. (5)."""
    m = modal_admittances(f, p)
    YD, YW = m["D"], m["W"]
    return 2 * R * YD * YW / (2 * R * YD * YW + YD + YW)


SETUPS = {"asym": s21_asym, "sym": s21_sym, "oc": s21_oc, "wd": s21_wd}


def s21(setup: str, f, p: MPCParams, R: float = R0):
    return SETUPS[setup](f, p, R)


def il_db(s):
    """Insertion loss in dB from a transmission coefficient.

    Valid when source and load resistances are equal, as in CISPR 17 50 Ohm / 50 Ohm.
    Without the DUT the load sees Vs/2, so IL = 20 log10(V_without / V_with) = -20 log10 |S21|.
    """
    return -20.0 * np.log10(np.abs(s))


def il_cm(f, p: MPCParams, R: float = R0):
    """CM insertion loss in dB, CISPR 17 asymmetrical setup."""
    return il_db(s21_asym(f, p, R))


def il_dm(f, p: MPCParams, R: float = R0):
    """DM insertion loss in dB, CISPR 17 symmetrical setup."""
    return il_db(s21_sym(f, p, R))


# ----------------------------------------------------------------------------
# Route 2. General termination of a 4-port admittance matrix
# ----------------------------------------------------------------------------
# Each setup is described by how the four DUT ports connect to a source node and a
# load node. Ports can be tied to the source terminal, tied to the load terminal or
# left open. For the symmetrical setup the ideal 180 degree dividers are modelled as
# a floating differential source and a floating differential load.



def s21_from_y(setup: str, Y: np.ndarray, R: float = R0) -> np.ndarray:
    """S21 of a setup computed from 4x4 port admittance matrices Y with shape (N, 4, 4).

    Port indices 0..3 correspond to ports 1..4 of the paper.
    """
    if Y.ndim == 2:
        Y = Y[None]
    out = np.empty(Y.shape[0], dtype=complex)
    G = 1.0 / R
    for n, Yn in enumerate(Y):
        if setup == "asym":
            # Ports 1 and 3 tied to node a, ports 2 and 4 tied to node b.
            # Source Vs with R into a, load R from b to ground.
            P = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], float)  # port -> node map
            Yr = P.T @ Yn @ P
            Yr[0, 0] += G
            Yr[1, 1] += G
            v = np.linalg.solve(Yr, np.array([G, 0.0]))  # Vs = 1
            out[n] = 2 * v[1]
        elif setup == "oc":
            # Source into port 1, load at port 2, ports 3 and 4 open.
            Yr = Yn.astype(complex).copy()
            Yr[0, 0] += G
            Yr[1, 1] += G
            v = np.linalg.solve(Yr, np.array([G, 0, 0, 0], complex))
            out[n] = 2 * v[1]
        elif setup == "wd":
            # Source into port 1, load at port 3, ports 2 and 4 open.
            Yr = Yn.astype(complex).copy()
            Yr[0, 0] += G
            Yr[2, 2] += G
            v = np.linalg.solve(Yr, np.array([G, 0, 0, 0], complex))
            out[n] = 2 * v[2]
        elif setup == "x14":
            # Source into port 1, load at port 4, ports 2 and 3 open.
            # Not used in the paper. Series impedance is 1/YC + 1/YW.
            Yr = Yn.astype(complex).copy()
            Yr[0, 0] += G
            Yr[3, 3] += G
            v = np.linalg.solve(Yr, np.array([G, 0, 0, 0], complex))
            out[n] = 2 * v[3]
        elif setup == "sym":
            # Floating differential source Vs with internal R between ports 1 and 3,
            # floating differential load R between ports 2 and 4.
            Yr = Yn.astype(complex).copy()
            for a, b in ((0, 2), (1, 3)):
                Yr[a, a] += G
                Yr[b, b] += G
                Yr[a, b] -= G
                Yr[b, a] -= G
            rhs = np.array([G, 0, -G, 0], complex)  # Norton equivalent of Vs = 1
            # With Cg = 0 the circuit has no ground reference, so fix it with a tiny
            # conductance at every port. Its effect is far below numerical precision
            # of the result.
            Yr += np.eye(4) * 1e-15
            v = np.linalg.solve(Yr, rhs)
            out[n] = 2 * (v[1] - v[3])
        else:
            raise ValueError(f"unknown setup {setup!r}")
    return out


def y_from_s(S: np.ndarray, R: float = R0) -> np.ndarray:
    """Port admittance matrix from an S matrix measured with reference R at all ports."""
    I = np.eye(S.shape[-1])
    return np.linalg.solve(I + S, I - S) / R


def il_from_s4p(S: np.ndarray, R_meas: float = R0, R: float = R0) -> dict[str, np.ndarray]:
    """CISPR 17 CM and DM insertion loss from a 4-port S matrix, shape (N, 4, 4).

    S is the single-ended matrix as measured by a 4-port VNA with R_meas at every port,
    port order 1..4 of the paper (1 = L in, 2 = L out, 3 = N in, 4 = N out).
    The matrix is converted to Y, which does not depend on the reference impedance,
    and then terminated as in the CISPR 17 setups with source and load R.
    The standard mixed-mode SCC21 and SDD21 at 50 Ohm per port use 25 Ohm CM and
    100 Ohm DM references and therefore do not give the CISPR 17 insertion loss.
    """
    Y = y_from_s(S, R_meas)
    return {"cm": il_db(s21_from_y("asym", Y, R)), "dm": il_db(s21_from_y("sym", Y, R))}


def s_from_y(Y: np.ndarray, R: float = R0) -> np.ndarray:
    """Port S matrix from admittance matrix, eq. (8), for equal reference R at all ports."""
    I = np.eye(Y.shape[-1])
    Yn = Y * R
    return np.linalg.solve(I + Yn, I - Yn)


# Mixed-mode change of basis, eq. (10). Order (CM1, CM2, DM1, DM2).
M_MIXED = np.array(
    [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, -1, 0], [0, 1, 0, -1]], float
) / np.sqrt(2)


def mixed_mode(S: np.ndarray) -> np.ndarray:
    """Mixed-mode S matrix, eq. (9). Returns [[Scc, Scd], [Sdc, Sdd]] blocks."""
    Minv = np.linalg.inv(M_MIXED)
    return Minv @ S @ M_MIXED


# ----------------------------------------------------------------------------
# From two-port series measurements to modal admittances
# ----------------------------------------------------------------------------
# With the other two ports open, the DUT between the measured ports acts as a series
# impedance Z, so S21 = 2R / (2R + Z) and Z = 2R (1/S21 - 1). The three possible
# connections of port 1 to another port give
#     Z_OC  (1 -> 2) = 1/YC + 1/YD
#     Z_WD  (1 -> 3) = 1/YD + 1/YW
#     Z_X14 (1 -> 4) = 1/YC + 1/YW
# Two of them leave three unknowns, so the paper needs a circuit model to split OC
# into YC and YD. All three together give the modal admittances without any model.

def series_z(s21, R: float = R0):
    """Series impedance from S21 of a two-port series connection."""
    return 2 * R * (1.0 / s21 - 1.0)


def modal_from_series(s21_oc, s21_wd, s21_x14, R: float = R0) -> dict[str, np.ndarray]:
    """Model-free YC, YD, YW from complex S21 of the OC, WD and X14 setups (YG = 0)."""
    zoc, zwd, zx = series_z(s21_oc, R), series_z(s21_wd, R), series_z(s21_x14, R)
    zc = (zoc + zx - zwd) / 2
    zd = (zoc + zwd - zx) / 2
    zw = (zwd + zx - zoc) / 2
    return {"C": 1 / zc, "D": 1 / zd, "W": 1 / zw}


def il_from_series(s21_oc, s21_wd, s21_x14, R: float = R0) -> dict[str, np.ndarray]:
    """CISPR 17 CM and DM insertion loss from the three series measurements."""
    m = modal_from_series(s21_oc, s21_wd, s21_x14, R)
    YC, YD, YW = m["C"], m["D"], m["W"]
    s_asym = 2 * R * YC / (2 * R * YC + 1)
    s_sym = 2 * R * (YD - YW) / ((R * YD + 2) * (R * YW + 2))
    return {"cm": il_db(s_asym), "dm": il_db(s_sym)}
