from cmc_mpc.ltspice import subckt_text, param_map



def test_subckt_for_extended_model(tmp_path):
    from cmc_mpc.mpc import MPCParams
    p = MPCParams(LC=7e-3, CC=3e-12, RC=3e4, LD=7e-6, CD=3e-12, RD=1.6e4, CW=3e-12, lcr_w=(3e-7, 9e-12, 1e3),
                  cr_branches=[(2e3, 4e-10), (5e3, 1e-10)], rl_branches=[(1e3, 1e-5)], Cg=4e-13, Rdc=2.5)
    txt = subckt_text(p, "MPC_t")      # asserts every expression against the netlist value
    assert ".subckt MPC_t port1 port2 port3 port4 params:" in txt
    assert "Rcr1=" in txt and "RDC=2.5" in txt and "Rs1 port1 q1 {RDC/2}" in txt
    names = [k.lower() for k in param_map(p)]
    assert len(names) == len(set(names))        # LTspice parameters are case insensitive


def test_generic_matches_variant_models():
    import pathlib
    import numpy as np
    root = pathlib.Path(__file__).resolve().parents[1]
    from cmc_mpc import load_params
    from cmc_mpc.ltspice import generic_params, mpc_from_generic, generic_subckt_text
    from cmc_mpc.mpc import y_matrix
    from cmc_mpc.setups import s21_from_y
    sets = list(load_params(root / "data/paper/table3_mpc_params.json").values())
    for fn in sorted((root / "examples/results").glob("choke*_params.json")):
        sets += list(load_params(fn).values())
    assert len(sets) >= 7
    f = np.logspace(6, 9, 200)
    for p in sets:
        g = mpc_from_generic(generic_params(p))
        for s in ("oc", "wd", "asym", "sym"):
            d = 20 * np.log10(np.abs(s21_from_y(s, y_matrix(f, g)) / s21_from_y(s, y_matrix(f, p))))
            assert np.max(np.abs(d)) < 0.05, (p.name, s, np.max(np.abs(d)))
    assert ".subckt MPC port1" in generic_subckt_text()
