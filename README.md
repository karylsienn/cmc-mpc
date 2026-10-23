# cmc-mpc

[![tests](https://github.com/karylsienn/cmc-mpc/actions/workflows/tests.yml/badge.svg)](https://github.com/karylsienn/cmc-mpc/actions/workflows/tests.yml)

Modal-parameters circuit (MPC) models of common-mode chokes in Python, with fitting to measured data and export to LTspice.

The model follows A. Ojeda-Rodriguez, J. Bernal-Mendez and M. A. Martin-Prats, "Modal Theory and Approach for Accurate Characterization of Common-Mode Chokes", IEEE Trans. Power Electron., 38(9), 2023. This repository is an independent implementation and extension. It is not affiliated with the authors.

## What is in it

- `src/cmc_mpc/mpc.py` closed-form modal admittances of the model (G, C, W and D modes).
- `src/cmc_mpc/circuit.py` the same model as an element netlist with a nodal solver, used to verify the closed form and as the source for the LTspice export.
- `src/cmc_mpc/setups.py` insertion loss in the CISPR 17 symmetric and asymmetric setups and in the OC, WD and X14 setups, and the S to Y to modal transforms.
- `src/cmc_mpc/fit.py` fitting from a measured four-port, from OC and WD curves, or from sym and asym curves. Optimizers are multistart least squares and differential evolution, and any callable can be plugged in.
- `src/cmc_mpc/ltspice.py` writes an LTspice subcircuit whose element values are checked against the verified netlist.
- `examples/` a synthetic recovery test of the optimizers, a command-line fit for your own s4p, and the results for three commercial chokes.
- `tests/` 60 tests, including the closed form against the netlist and the exported expressions against the netlist.

## Quick start

Developed and tested with Python 3.13.2. Python 3.10 or newer is required.

```bash
git clone https://github.com/karylsienn/cmc-mpc
cd cmc-mpc
pip install -e .[dev]     # or: pip install -r requirements-dev.txt
pytest
python examples/01_fit_synthetic.py
python examples/02_fit_s4p.py my_choke.s4p mychoke --rl 1
```

```python
from cmc_mpc import FourPort, Variant, fit, read_s4p
from cmc_mpc.ltspice import write_ltspice

f, S, z0 = read_s4p("my_choke.s4p")
result = fit(FourPort(f, S, z0), Variant(lcr_w=True, cr=1, rl=1, cg=True))
print(result.report())
write_ltspice(result.params, "ltspice", "mychoke")
```

A fit prints for every parameter the linearised log-standard deviation and flags parameters that are not determined by the data or sit at a bound. Take these flags seriously, they tell you which parts of the model your measurement cannot see.

## What each input can identify

| Input | Identifies |
|---|---|
| Four-port S matrix | YC, YD, YW and YG, so also the ground capacitance Cg |
| OC and WD curves | YC, YD and YW |
| Sym and asym curves | YC and YD minus YW only, so CW is not separable from CD |

## Differences from the paper

The model has two additions with defaults that reproduce the paper exactly.

- Optional R-L branches across each CM winding, to represent frequency-dependent permeability, as suggested in the paper's reference [13].
- Optional winding resistance `Rdc`, split into Rdc/2 at each port and in series with the C, D and W modes. It matters for small SMD chokes, where the measured differential-mode admittance is resistive at low frequency.

A capacitance Cg from each port to ground shifts all four modal admittances, not only YG. See `docs/decisions.md`.

The L and C columns of the LCR_W network in the published Table III appear to be swapped. With the printed values the first WD peak moves to 11 MHz and the symmetric peak near 57 MHz disappears, which contradicts the paper's figures. `data/paper/table3_mpc_params.json` holds the values used here and, where it applies, the values as printed.

## Results on three commercial chokes

The vendor files are not redistributed. The chokes are called choke1 to choke3, and the fitted parameters, figures and the error table are in `examples/results/`. The error is the rms difference of S21 in dB from 1 MHz to 1 GHz against the measured four-port. The vendor netlist of choke2 is a behavioural model and is not compared.

| choke | model | OC | WD | X14 | asym | sym |
|---|---|---|---|---|---|---|
| choke1 | MPC | 0.41 | 2.22 | 3.46 | 0.35 | 0.14 |
| choke1 | vendor netlist | 3.74 | 3.88 | 4.89 | 3.24 | 2.31 |
| choke2 | MPC with Rdc | 0.89 | 0.59 | 0.86 | 0.88 | 0.38 |
| choke3 | MPC | 0.82 | 0.51 | 0.39 | 0.17 | 1.32 |
| choke3 | vendor netlist | 2.10 | 8.70 | 8.03 | 0.52 | 2.37 |

These numbers describe how well the structure fits one measurement of each part. They are not a validation on held-out parts. For choke2 the ground-coupling mode was excluded from the fit, and the D-mode resonance is still off.

## LTspice

`ltspice/MPC.sub` is one generic subcircuit with ports `port1` and `port2` on one winding and `port3` and `port4` on the other. It contains every optional part of the model, one R-C branch, one R-L branch, the LCR tank in the interwinding path, a winding resistance `RDC` and a ground capacitance `Cg`. Parts a given choke does not need are switched off by value, for example `Lw=1e-12` shorts the tank and `Rrl=1e12` opens the R-L branch. The switched-off model stays within 0.02 dB of the exact variant model for every parameter set shipped here, up to 3 GHz.

Two ways to use it for a particular choke:

- Add `.include MPC.sub` and `.include presets/MPC_choke3.sub`, then place the subcircuit `MPC_choke3`.
- Use the generic `MPC` and paste the contents of `presets/choke3.spiceline.txt` into the instance line or the symbol's SpiceLine.

`cmc_mpc.ltspice.write_preset` writes both files, and a Python reference s4p, for your own fit. The reference is computed from the same circuit and is meant for checking an LTspice run.

## Licence

MIT, see `LICENSE`.
