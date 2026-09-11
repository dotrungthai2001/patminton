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
from ctypes import CDLL, POINTER, RTLD_GLOBAL, c_char
from pathlib import Path

_CREATORS = (
    "createPAT_points",
    "createPAT_cylinder_exact",
    "createPAT_cylinder_trapezoidal",
    "createPAT_cylinder_lut",
    "createPAT_cylinder_as_arcs",
    "createPAT_cylinder_as_planes",
)

_lib = None


def _cufft_search_dirs():
    """Directories outside the loader path that may hold libcufft."""
    import sys

    dirs = []
    for entry in filter(None, sys.path):
        site = Path(entry)
        # pip CUDA wheels (nvidia/cufft for CUDA <= 12, nvidia/cu13 for
        # CUDA 13), then libraries bundled with torch
        dirs += [site / "nvidia" / "cufft" / "lib", site / "nvidia" / "cu13" / "lib",
                 site / "torch" / "lib"]
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        dirs.append(Path(cuda_home) / "lib64")
    return dirs


def _preload_cufft():
    """Load every libcufft found by _cufft_search_dirs into the global namespace.

    libpat_gpu.so links dynamically against libcufft (static linking costs
    270 MB). When the system loader cannot find it, it usually sits in the
    cuFFT wheel that PyTorch pulls in, which is not on the loader path.
    dlopen registers a library under its soname, and cuFFT symbols are
    versioned (e.g. cufftPlanMany@libcufft.so.12), so loading every candidate
    satisfies whichever version libpat_gpu.so was built against.
    """
    for directory in _cufft_search_dirs():
        for candidate in sorted(directory.glob("libcufft.so.*")):
            try:
                CDLL(str(candidate), mode=RTLD_GLOBAL)
            except OSError:
                pass


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

    try:
        lib = CDLL(path)
    except OSError as exc:
        if "libcufft" not in str(exc):
            raise
        _preload_cufft()
        try:
            lib = CDLL(path)
        except OSError:
            raise OSError(
                f"{exc}\n\nlibcufft could not be found. Install the CUDA toolkit, "
                "or the cuFFT wheel matching the CUDA major version of patminton "
                "(pip install patminton[cuda12] for libcufft.so.11, "
                "patminton[cuda13] for libcufft.so.12), or add the directory "
                "holding libcufft to LD_LIBRARY_PATH."
            ) from None
    # The constructors return an opaque pointer to the C++ PAT instance.
    for name in _CREATORS:
        getattr(lib, name).restype = POINTER(c_char)
    _lib = lib
    return _lib
