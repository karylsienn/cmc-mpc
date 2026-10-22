"""Verification of step 1.1.

The closed-form modal expressions of the paper are checked against an independent
element-level nodal analysis of the circuit in Fig. 3.
"""

from pathlib import Path

import numpy as np
import pytest

from cmc_mpc.mpc import EV, EV_INV, MPCParams, load_params, y_matrix, modal_admittances
from cmc_mpc.circuit import mpc_netlist, port_y
from cmc_mpc import setups

ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_params(ROOT / "data/paper/table3_mpc_params.json")
F = np.logspace(4, 8, 41)  # 10 kHz to 100 MHz
RTOL = 1e-7


def _variants():
    """Paper parameter sets plus stress cases with ground capacitance and RL branches."""
    out = dict(PARAMS)
    we = PARAMS["WE_7448261418"]
    out["WE_with_Cg"] = MPCParams.from_dict({**we.to_dict(), "Cg": 2e-12, "name": "WE_with_Cg"})
    out["WE_with_RL"] = MPCParams.from_dict(
        {**we.to_dict(), "rl_branches": [(300.0, 1e-3), (3e3, 0.2e-3)], "name": "WE_with_RL"}
    )
    return out


VARIANTS = _variants()


@pytest.mark.parametrize("name", list(PARAMS))
def test_csym_matches_table(name):
    p = PARAMS[name]
    raw = __import__("json").load(open(ROOT / "data/paper/table3_mpc_params.json"))[name]
    assert p.C_sym == pytest.approx(raw["C_sym_paper"], abs=0.05e-12)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_circuit_y_equals_modal_y(name):
    """Nodal analysis of Fig. 3 reproduces EV diag(YG, YC, YW, YD) EV^-1."""
    p = VARIANTS[name]
    Yc = port_y(mpc_netlist(p), F)
    Ym = y_matrix(F, p)
    scale = np.abs(Ym).max(axis=(1, 2))[:, None, None]
    assert np.max(np.abs(Yc - Ym) / scale) < RTOL


@pytest.mark.parametrize("name", list(VARIANTS))
def test_natural_modes_diagonalise_circuit(name):
    """Table I modes are the eigenvectors of the circuit admittance matrix."""
    p = VARIANTS[name]
    Yc = port_y(mpc_netlist(p), F)
    Ymod = np.einsum("ij,njk,kl->nil", EV_INV, Yc, EV)
    diag = np.einsum("nii->ni", Ymod)
    off = Ymod - np.einsum("ni,ij->nij", diag, np.eye(4))
    assert np.max(np.abs(off)) < RTOL * np.max(np.abs(diag))
    m = modal_admittances(F, p)
    for i, k in enumerate("GCWD"):
        ref = m[k]
        if np.all(ref == 0):
            assert np.max(np.abs(diag[:, i])) < 1e-12
        else:
            np.testing.assert_allclose(diag[:, i], ref, rtol=RTOL)


@pytest.mark.parametrize("setup", ["asym", "sym", "oc", "wd"])
@pytest.mark.parametrize("name", list(VARIANTS))
def test_closed_form_equals_terminated_circuit(name, setup):
    """Eqs. (1), (2), (4), (5) equal direct termination of the circuit Y matrix."""
    p = VARIANTS[name]
    Yc = port_y(mpc_netlist(p), F)
    s_circ = setups.s21_from_y(setup, Yc)
    s_form = setups.s21(setup, F, p)
    if p.Cg > 0 and setup in ("oc", "wd"):
        pytest.skip("eqs. (4) and (5) assume YG = 0")
    np.testing.assert_allclose(s_circ, s_form, rtol=1e-6)


@pytest.mark.parametrize("name", list(PARAMS))
def test_cispr_setups_equal_mixed_mode(name):
    """Reference impedances of the CISPR 17 setups expressed as mixed-mode S parameters.

    The 50 Ohm / 50 Ohm symmetrical setup has a 50 Ohm differential reference, which is
    SDD21 with 25 Ohm per port. The asymmetrical setup has a 50 Ohm common-mode
    reference, which is SCC21 with 100 Ohm per port.
    """
    p = PARAMS[name]
    Y = y_matrix(F, p)
    Sm = setups.mixed_mode(setups.s_from_y(Y, R=25.0))
    np.testing.assert_allclose(Sm[:, 3, 2], setups.s21_sym(F, p), rtol=1e-9)
    Sm100 = setups.mixed_mode(setups.s_from_y(Y, R=100.0))
    np.testing.assert_allclose(Sm100[:, 1, 0], setups.s21_asym(F, p), rtol=1e-9)
    Sm = setups.mixed_mode(setups.s_from_y(Y))
    # No CM/DM conversion, eq. (11).
    assert np.max(np.abs(Sm[:, :2, 2:])) < 1e-12
    assert np.max(np.abs(Sm[:, 2:, :2])) < 1e-12


@pytest.mark.parametrize("name", list(PARAMS))
def test_low_frequency_cm_asymptote(name):
    """At low frequency the CM insertion loss follows |1 + j w (LC/2) / (2R)|."""
    p = PARAMS[name]
    f = np.array([1e3])
    il = setups.il_db(setups.s21_asym(f, p))
    ref = 20 * np.log10(np.abs(1 + 2j * np.pi * f * (p.LC / 2) / 100.0))
    assert il[0] == pytest.approx(ref[0], abs=0.05)


@pytest.mark.parametrize("name", list(PARAMS))
def test_il_from_s4p_roundtrip(name):
    """CISPR 17 IL recovered from a simulated 50 Ohm 4-port S matrix equals the modal formulas."""
    p = PARAMS[name]
    S = setups.s_from_y(y_matrix(F, p), R=50.0)
    il = setups.il_from_s4p(S)
    np.testing.assert_allclose(il["cm"], setups.il_cm(F, p), atol=1e-9)
    np.testing.assert_allclose(il["dm"], setups.il_dm(F, p), atol=1e-9)


@pytest.mark.parametrize("name", list(PARAMS))
def test_three_series_setups_give_cispr_il(name):
    """OC, WD and X14 measured with phase determine YC, YD, YW and hence CISPR 17 IL."""
    p = PARAMS[name]
    Y = port_y(mpc_netlist(p), F)
    s = {k: setups.s21_from_y(k, Y) for k in ("oc", "wd", "x14")}
    m = setups.modal_from_series(s["oc"], s["wd"], s["x14"])
    ref = modal_admittances(F, p)
    for k in "CDW":
        np.testing.assert_allclose(m[k], ref[k], rtol=1e-5)
    il = setups.il_from_series(s["oc"], s["wd"], s["x14"])
    np.testing.assert_allclose(il["cm"], setups.il_cm(F, p), atol=1e-4)
    np.testing.assert_allclose(il["dm"], setups.il_dm(F, p), atol=1e-4)


def test_rdc_closed_form_matches_netlist():
    import dataclasses
    from cmc_mpc.circuit import mpc_netlist, port_y
    from cmc_mpc.mpc import y_matrix
    p = dataclasses.replace(PARAMS["WE_7448261418"], Cg=3e-13, Rdc=2.5)
    f = np.logspace(5, 9, 60)
    assert np.allclose(port_y(mpc_netlist(p), f), y_matrix(f, p), rtol=1e-8, atol=1e-12)
