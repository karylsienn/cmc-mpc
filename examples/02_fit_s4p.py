"""Fit your own measurement and export an LTspice subcircuit.

python examples/02_fit_s4p.py MEASUREMENT.s4p TAG [--rl N] [--rdc] [--no-cg] [--no-lcrw]

The four-port must use the port order of the paper, 1 and 2 on one winding and 3 and 4 on the other.
Results go to results_<TAG>/ as a parameter file, an LTspice subcircuit and a Python reference s4p.
"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cmc_mpc import FourPort, Variant, fit, read_s4p
from cmc_mpc.ltspice import write_ltspice

ap = argparse.ArgumentParser()
ap.add_argument("s4p"); ap.add_argument("tag")
ap.add_argument("--rl", type=int, default=1, help="number of R-L branches, default 1")
ap.add_argument("--cr", type=int, default=1, help="number of R-C branches, default 1")
ap.add_argument("--rdc", action="store_true", help="fit a winding resistance (extension of the paper model)")
ap.add_argument("--no-cg", action="store_true"); ap.add_argument("--no-lcrw", action="store_true")
ap.add_argument("--method", default="least_squares", choices=["least_squares", "de"])
a = ap.parse_args()

f, S, z0 = read_s4p(a.s4p)
v = Variant(lcr_w=not a.no_lcrw, cr=a.cr, rl=a.rl, cg=not a.no_cg, rdc=a.rdc)
r = fit(FourPort(f, S, z0, mode_weights=dict(G=0.0 if a.no_cg else 0.5, C=1, W=1, D=1)), v, method=a.method, name=a.tag)
print(r.report())
out = pathlib.Path(f"results_{a.tag}"); out.mkdir(exist_ok=True)
json.dump({a.tag: r.params.to_dict()}, open(out / f"{a.tag}_params.json", "w"), indent=1)
print(write_ltspice(r.params, out, a.tag, a.tag))
