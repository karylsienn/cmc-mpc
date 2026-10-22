"""Minimal Touchstone reader for 4-port files, no external dependency."""
from __future__ import annotations

import numpy as np

_UNITS = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}


def read_s4p(path):
    """Return (f in Hz, S with shape (N, 4, 4), z0). Supports RI, MA and DB formats."""
    toks, z0, fmt, unit = [], 50.0, "ma", 1e9
    with open(path) as fh:
        for line in fh:
            line = line.split("!")[0].strip()
            if not line:
                continue
            if line.startswith("#"):
                t = line.lower().split()
                unit, fmt, z0 = _UNITS[t[1]], t[3], float(t[5])
                continue
            toks += [float(x) for x in line.split()]
    a = np.array(toks).reshape(-1, 33)
    f = a[:, 0] * unit
    p = a[:, 1:].reshape(-1, 16, 2)
    if fmt == "db":
        c = 10 ** (p[..., 0] / 20) * np.exp(1j * np.deg2rad(p[..., 1]))
    elif fmt == "ma":
        c = p[..., 0] * np.exp(1j * np.deg2rad(p[..., 1]))
    else:
        c = p[..., 0] + 1j * p[..., 1]
    # Touchstone v1 four-port files list S11 S12 S13 S14 on one row, so the order is row-major
    return f, c.reshape(-1, 4, 4), z0


def write_s4p(path, f, S, z0=50.0, comment=""):
    with open(path, "w") as fh:
        if comment:
            fh.write(f"! {comment}\n")
        fh.write(f"# Hz S RI R {z0:g}\n")
        for fr, Sn in zip(f, S):
            for r in range(4):
                fh.write((f"{fr:.9e} " if r == 0 else "    ")
                         + " ".join(f"{Sn[r, c].real:.9e} {Sn[r, c].imag:.9e}" for c in range(4)) + "\n")
