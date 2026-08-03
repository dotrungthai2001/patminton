"""Loader for the compiled CUDA shared library ``libpat_gpu.so``.

The library is compiled by ``pip install`` (see ``setup.py``) and placed next
to this file inside the installed package. Loading is lazy so that importing
:mod:`patminton` succeeds on machines without the compiled library (e.g. when
building the documentation); the error is raised only when a
:class:`patminton.PAT` operator is instantiated.

The environment variable ``PATMINTON_LIB`` overrides the library path, which is
useful to point the package at a library built manually with ``make``.
"""

import os
from ctypes import CDLL, POINTER, c_char
from pathlib import Path

_CREATORS = (
    "createPAT_points",
    "createPAT_cylinder_exact",
    "createPAT_cylinder_far_field",
    "createPAT_cylinder_lut",
    "createPAT_cylinder_as_arcs",
    "createPAT_cylinder_as_planes",
)

_lib = None


def get_lib():
    """Load (once) and return the ``libpat_gpu.so`` ctypes handle."""
    global _lib
    if _lib is not None:
        return _lib

    path = os.environ.get("PATMINTON_LIB")
    if path is None:
        candidate = Path(__file__).resolve().parent / "libpat_gpu.so"
        if not candidate.exists():
            raise OSError(
                f"Compiled CUDA library not found at {candidate}. "
                "Reinstall the package with the CUDA toolkit available "
                "(pip install patminton), or build it manually with nvcc and "
                "point the PATMINTON_LIB environment variable at the .so file."
            )
        path = str(candidate)

    lib = CDLL(path)
    # The constructors return an opaque pointer to the C++ PAT instance.
    for name in _CREATORS:
        getattr(lib, name).restype = POINTER(c_char)
    _lib = lib
    return _lib
