"""Parameter recovery on a synthetic choke. Truth is the KEMET SC-02-30G set of Table III extended with
an R-C branch and a ground capacitance, data come from the element netlist, so the closed form is checked too."""
import numpy as np
import pytest

from cmc_mpc import FourPort, SetupCurves, Variant, fit
from cmc_mpc.circuit import mpc_netlist, port_y
from cmc_mpc.mpc import MPCParams
from cmc_mpc.setups import s21, s_from_y

F = np.logspace(5, 9, 401)
TRUTH = MPCParams(LC=7.1e-3, CC=3.1e-12, RC=35.3e3, LD=7.3e-6, CD=3.3e-12, RD=16.1e3, CW=2.9e-12,
                  lcr_w=(3e-7, 9e-12, 1e3), cr_branches=[(2e3, 4e-10)], Cg=4e-13)


@pytest.fixture(scope="module")
def s4p():
    S = s_from_y(port_y(mpc_netlist(TRUTH), F))
    rng = np.random.default_rng(0)
    return S + 3e-5 * (rng.standard_normal(S.shape) + 1j * rng.standard_normal(S.shape))


def il_err(p, setup, f=np.logspace(6, 9, 300)):
    return np.sqrt(np.mean((20 * np.log10(np.abs(s21(setup, f, p))) - 20 * np.log10(np.abs(s21(setup, f, TRUTH)))) ** 2))


def test_four_port_least_squares_recovers_setups(s4p):
    r = fit(FourPort(F, s4p), Variant(lcr_w=True, cr=1, rl=0, cg=True), nstart=15, seed=1)
    for setup in ("oc", "wd", "asym", "sym"):
        assert il_err(r.params, setup) < 0.5, setup
    assert abs(np.log(r.params.Cg / TRUTH.Cg)) < 0.3


def test_oc_wd_only(s4p):
    """The OC and WD fit has a local minimum at 1.26 dB error that about half of the random starts fall into.
    With 15 starts the result depended on the scipy version, so 60 starts are used, which found the global
    minimum for every seed tried. fit() reports how many starts reached the best cost."""
    meas = SetupCurves(F, {k: s21(k, F, TRUTH) for k in ("oc", "wd")})
    r = fit(meas, Variant(lcr_w=True, cr=1, rl=0, cg=False), nstart=60, seed=2)
    assert r.hits >= 2
    for setup in ("oc", "wd"):
        assert il_err(r.params, setup) < 0.5, setup


def test_differential_evolution_agrees_with_least_squares(s4p):
    meas = FourPort(F, s4p)
    v = Variant(lcr_w=True, cr=1, rl=0, cg=True)
    ls = fit(meas, v, nstart=15, seed=1)
    de = fit(meas, v, method="de", popsize=12, maxiter=120, seed=1)
    assert de.cost < 5 * ls.cost + 1.0


def test_cg_rejected_for_setup_curves():
    with pytest.raises(ValueError):
        fit(SetupCurves(F, {"oc": s21("oc", F, TRUTH)}), Variant(cg=True))
