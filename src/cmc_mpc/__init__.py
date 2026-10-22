"""Modal-parameters circuit (MPC) models of common-mode chokes.

Model: Ojeda-Rodriguez, Bernal-Mendez and Martin-Prats, IEEE TPEL 38(9), 2023.
"""
from .mpc import MPCParams, load_params, modal_admittances, y_matrix  # noqa: F401
from .fit import FourPort, SetupCurves, Variant, fit  # noqa: F401
from .io import read_s4p, write_s4p  # noqa: F401

__version__ = "0.1.0"
