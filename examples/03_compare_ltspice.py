"""Compare an LTspice four-port run with the Python reference s4p.

python examples/03_compare_ltspice.py ltspice/MPC_testbench.raw ltspice/presets/MPC_choke3_python.s4p

The testbench drives port k with AC 1 V behind 50 Ohm (.step param k 1 4) and reads the port nodes N001 to N004.
Then S_jk = 2 V(Nj) / 1 V - delta_jk. Needs spicelib.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from spicelib import RawRead
from cmc_mpc import read_s4p

raw = RawRead(sys.argv[1])
assert len(raw.get_steps()) == 4, "expected four steps, one per driven port"
f = np.real(raw.get_trace("frequency").get_wave(0))
V = np.array([[raw.get_trace(f"V(n00{j})").get_wave(k) for j in range(1, 5)] for k in range(4)])  # drive, port, freq
S = np.transpose(2 * V, (2, 1, 0)) - np.eye(4)[None]
fr, Sr, _ = read_s4p(sys.argv[2])
Sr = np.array([[np.interp(f, fr, Sr[:, i, j].real) + 1j * np.interp(f, fr, Sr[:, i, j].imag) for j in range(4)] for i in range(4)]).transpose(2, 0, 1)
dB = lambda x: 20 * np.log10(np.abs(x))
print(f"{len(f)} points, {f[0]/1e6:.3g} to {f[-1]/1e6:.4g} MHz")
print(f"max |S_ltspice - S_reference| over all 16 entries: {np.abs(S - Sr).max():.3g}")
print(f"max dB difference over all 16 entries:             {np.abs(dB(S) - dB(Sr)).max():.4f} dB")
print(f"reciprocity of the LTspice result, max |Sij - Sji|: {np.abs(S - S.transpose(0, 2, 1)).max():.3g}")
