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
    "createPAT_cylinder_far_field",
    "createPAT_cylinder_lut",
    "createPAT_cylinder_as_arcs",
    "createPAT_cylinder_as_planes",
)

_lib = None


def _cufft_search_dirs():
    """Directories that may hold libcufft, most authoritative first."""
    import importlib.util

    dirs = []
    for module in ("nvidia.cufft", "torch"):
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            continue
        if spec is None or not spec.submodule_search_locations:
            continue
        root = Path(spec.submodule_search_locations[0])
        dirs += [root / "lib", root / "cufft" / "lib", root.parent / "cu13" / "lib"]
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        dirs.append(Path(cuda_home) / "lib64")
    return dirs


def _preload_cufft():
    """Load libcufft into the global namespace so libpat_gpu.so can resolve it.

    libpat_gpu.so links dynamically against libcufft (static linking costs
    270 MB). The system loader finds it when the CUDA toolkit is installed;
    otherwise it usually sits in the nvidia-cufft-cu1x wheel that PyTorch
    pulls in, which is not on the loader path. dlopen registers a library
    under its soname, so loading every candidate by absolute path satisfies
    whichever version libpat_gpu.so was built against.
    """
    try:
        CDLL("libcufft.so.11", mode=RTLD_GLOBAL)
        return
    except OSError:
        pass
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

    _preload_cufft()
    try:
        lib = CDLL(path)
    except OSError as exc:
        if "libcufft" not in str(exc):
            raise
        raise OSError(
            f"{exc}\n\nlibcufft could not be found. Install the CUDA toolkit, "
            "or the matching cuFFT wheel (pip install nvidia-cufft-cu12 for a "
            "CUDA 12 build of patminton, nvidia-cufft-cu13 for CUDA 13), or "
            "add the directory holding libcufft to LD_LIBRARY_PATH."
        ) from None
    # The constructors return an opaque pointer to the C++ PAT instance.
    for name in _CREATORS:
        getattr(lib, name).restype = POINTER(c_char)
    _lib = lib
    return _lib
