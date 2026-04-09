"""Python port of legacy MATLAB scripts in ``data_build``.

This package mirrors the old workflow and keeps function naming close
to MATLAB code for easier cross-checking.
"""

from .initialization import initialization
from .mesh_io import pre_amira_mesh, getsurface
from .fem_matrices import fem_linear, form_matrix_f
from .laplacian import getlaplace, new_getlaplace, new_getlaplace_ghb

__all__ = [
    "initialization",
    "pre_amira_mesh",
    "getsurface",
    "fem_linear",
    "form_matrix_f",
    "getlaplace",
    "new_getlaplace",
    "new_getlaplace_ghb",
]

