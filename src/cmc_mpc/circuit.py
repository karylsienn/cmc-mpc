"""Element-level netlist of the MPC model and a small MNA solver.

The netlist is the single source of truth for the circuit topology of Fig. 3.
It serves two purposes.

1. Independent verification. The 4-port admittance matrix computed from the
   netlist by nodal analysis must equal the modal construction in mpc.y_matrix.
2. Export. The same element list is written out as an LTspice or Simscape model
   in step 5, so the exported circuit is exactly the verified one.

Node names: "0" is ground, "p1".."p4" are the ports of Fig. 1.
Inductor orientation matters for coupling. All coupling coefficients are written
as positive values, and the dot convention is encoded in the node order (n1 = dot).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mpc import MPCParams

PORTS = ("p1", "p2", "p3", "p4")


@dataclass
class Element:
    kind: str  # "R", "C", "L"
    name: str
    n1: str
    n2: str
    value: float


@dataclass
class Coupling:
    name: str
    inductors: tuple[str, ...]  # all mutually coupled with coefficient k
    k: float = 1.0


@dataclass
class Netlist:
    elements: list[Element] = field(default_factory=list)
    couplings: list[Coupling] = field(default_factory=list)
    title: str = ""

    def add(self, kind, name, n1, n2, value):
        self.elements.append(Element(kind, name, n1, n2, float(value)))

    def couple(self, name, inductors, k=1.0):
        self.couplings.append(Coupling(name, tuple(inductors), float(k)))

    @property
    def nodes(self) -> list[str]:
        seen = []
        for e in self.elements:
            for n in (e.n1, e.n2):
                if n != "0" and n not in seen:
                    seen.append(n)
        return seen


def mpc_netlist(p: MPCParams) -> Netlist:
    """Element-level netlist of the MPC model of Fig. 3, Fig. 4 and eq. (3)."""
    nl = Netlist(title=f"MPC model {p.name}".strip())

    # Optional winding resistance, Rdc/2 between each port and the inner node q.
    q = {n: (f"q{n}" if p.Rdc > 0 else f"p{n}") for n in "1234"}
    if p.Rdc > 0:
        for n in "1234":
            nl.add("R", f"Rs{n}", f"p{n}", q[n], p.Rdc / 2)

    # CM block. Two perfectly coupled windings of LC/2, dots at the input side.
    nl.add("L", "Lc1", q["1"], "a1", p.LC / 2)
    nl.add("L", "Lc3", q["3"], "a3", p.LC / 2)
    nl.couple("Kc", ("Lc1", "Lc3"), 1.0)
    for w, (na, nb) in {"1": (q["1"], "a1"), "3": (q["3"], "a3")}.items():
        nl.add("C", f"Cc{w}", na, nb, p.CC)
        nl.add("R", f"Rc{w}", na, nb, p.RC)
        for i, (R, C) in enumerate(p.cr_branches):
            x = f"xcr{w}_{i}"
            nl.add("R", f"Rcr{w}_{i}", na, x, R)
            nl.add("C", f"Ccr{w}_{i}", x, nb, C)
        for i, (R, L) in enumerate(p.rl_branches):
            x = f"xrl{w}_{i}"
            nl.add("R", f"Rrl{w}_{i}", na, x, R)
            nl.add("L", f"Lrl{w}_{i}", x, nb, L)

    # DM block. Four perfectly coupled LD/8 inductors. The N-path inductors are
    # written with reversed node order so that DM currents add and W-mode and
    # C-mode currents cancel.
    nl.add("L", "Ld1", "a1", "m1", p.LD / 8)
    nl.add("L", "Ld2", "m1", q["2"], p.LD / 8)
    nl.add("L", "Ld3", "m3", "a3", p.LD / 8)
    nl.add("L", "Ld4", q["4"], "m3", p.LD / 8)
    nl.couple("Kd", ("Ld1", "Ld2", "Ld3", "Ld4"), 1.0)
    for w, (na, nb) in {"1": ("a1", q["2"]), "3": ("a3", q["4"])}.items():
        nl.add("C", f"Cd{w}", na, nb, p.CD)
        nl.add("R", f"Rd{w}", na, nb, p.RD)

    # YW between the path midpoints, Fig. 4.
    if p.lcr_w is None:
        nl.add("C", "Cw", "m1", "m3", p.CW)
    else:
        L, C, R = p.lcr_w
        nl.add("C", "Cw", "m1", "w1", p.CW)
        nl.add("L", "Lw", "w1", "m3", L)
        nl.add("C", "Cwt", "w1", "m3", C)
        nl.add("R", "Rw", "w1", "m3", R)

    if p.Cg > 0:
        for port in PORTS:
            nl.add("C", f"Cg{port[1]}", port, "0", p.Cg)
    return nl


def port_y(nl: Netlist, f, ports=PORTS) -> np.ndarray:
    """Port admittance matrix by MNA with inductor currents as unknowns.

    Ideal coupling (k = 1, singular inductance matrix) is handled exactly because
    the inductor equations are written in impedance form, V = jw M I.
    Returns an array of shape (len(f), n_ports, n_ports).
    """
    f = np.atleast_1d(np.asarray(f, float))
    nodes = nl.nodes
    for prt in ports:
        if prt not in nodes:
            raise ValueError(f"port node {prt} not in netlist")
    inds = [e for e in nl.elements if e.kind == "L"]
    ni = {n: i for i, n in enumerate(nodes)}
    li = {e.name: i for i, e in enumerate(inds)}
    nn, nL = len(nodes), len(inds)

    Lmat = np.diag([e.value for e in inds]).astype(float)
    for c in nl.couplings:
        for a in c.inductors:
            for b in c.inductors:
                if a != b:
                    Lmat[li[a], li[b]] = c.k * np.sqrt(inds[li[a]].value * inds[li[b]].value)

    port_idx = [ni[p] for p in ports]
    other = [i for i in range(nn + nL) if i not in port_idx]
    out = np.empty((f.size, len(ports), len(ports)), complex)

    for k, fk in enumerate(f):
        jw = 2j * np.pi * fk
        A = np.zeros((nn + nL, nn + nL), complex)
        for e in nl.elements:
            if e.kind == "L":
                continue
            y = 1.0 / e.value if e.kind == "R" else jw * e.value
            a = ni.get(e.n1)
            b = ni.get(e.n2)
            if a is not None:
                A[a, a] += y
            if b is not None:
                A[b, b] += y
            if a is not None and b is not None:
                A[a, b] -= y
                A[b, a] -= y
        for e in inds:
            r = nn + li[e.name]
            a = ni.get(e.n1)
            b = ni.get(e.n2)
            if a is not None:
                A[a, r] += 1.0
                A[r, a] += 1.0
            if b is not None:
                A[b, r] -= 1.0
                A[r, b] -= 1.0
        A[nn:, nn:] -= jw * Lmat
        App = A[np.ix_(port_idx, port_idx)]
        Api = A[np.ix_(port_idx, other)]
        Aip = A[np.ix_(other, port_idx)]
        Aii = A[np.ix_(other, other)]
        out[k] = App - Api @ np.linalg.solve(Aii, Aip)
    return out
