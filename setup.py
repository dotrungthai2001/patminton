"""Build script: compiles the CUDA library libpat_gpu.so with nvcc.

The library is not a Python extension module — it is a plain shared library
loaded at runtime with ctypes — so we bypass the default compiler and invoke
nvcc directly, placing the .so inside the patminton package.

Environment variables:
    PATMINTON_CUDA_ARCH  GPU architectures to compile for, e.g. "86" or
                     "70;80;86", or "native" for the GPU of the build
                     machine. The last one also gets embedded PTX for
                     forward compatibility. Default: every architecture
                     supported by the nvcc in use (see DEFAULT_ARCHS), so
                     that no GPU has to be visible at build time.
    PATMINTON_NVCC       Path to the nvcc binary (default: "nvcc" from PATH).
    PATMINTON_PLAT_TAG   Platform tag of the built wheel, e.g.
                     "manylinux_2_28_x86_64" (default: the build platform).
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

try:  # setuptools >= 70.1 vendors bdist_wheel; older versions need the wheel package
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # pragma: no cover
    try:
        from wheel.bdist_wheel import bdist_wheel
    except ImportError:
        bdist_wheel = None

ROOT = Path(__file__).resolve().parent
CUDA_DIR = ROOT / "src" / "patminton" / "cuda"

# Volta (70) was removed in CUDA 13; Blackwell (100, 120) needs CUDA >= 12.8.
DEFAULT_ARCHS = {
    12: ["70", "75", "80", "86", "89", "90"],
    13: ["75", "80", "86", "89", "90", "100", "120"],
}


def nvcc_major_version(nvcc):
    out = subprocess.run(
        [nvcc, "--version"], check=True, capture_output=True, text=True
    ).stdout
    m = re.search(r"release (\d+)\.", out)
    return int(m.group(1)) if m else 0


def gencode_flags(nvcc_major):
    """Build the -gencode flags: native SASS per architecture, PTX for the last."""
    arch = os.environ.get("PATMINTON_CUDA_ARCH", "")
    if arch.strip() == "native":
        return ["-arch=native"]
    if arch.strip():
        archs = [a.strip() for a in re.split(r"[;,\s]+", arch) if a.strip()]
    else:
        archs = DEFAULT_ARCHS.get(nvcc_major, DEFAULT_ARCHS[12])
    flags = []
    for a in archs:
        flags += ["-gencode", f"arch=compute_{a},code=sm_{a}"]
    flags += ["-gencode", f"arch=compute_{archs[-1]},code=compute_{archs[-1]}"]
    return flags


class CUDAExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class BuildLibPAT(build_ext):
    def run(self):
        nvcc = shutil.which(os.environ.get("PATMINTON_NVCC", "nvcc"))
        if nvcc is None:
            raise RuntimeError(
                "nvcc not found. Installing patminton requires the CUDA toolkit "
                "(https://developer.nvidia.com/cuda-toolkit). If nvcc is not "
                "on PATH, set the PATMINTON_NVCC environment variable."
            )

        nvcc_major = nvcc_major_version(nvcc)
        sources = sorted((CUDA_DIR / "src").glob("*.cu")) + [CUDA_DIR / "libpat.cu"]
        out_dir = Path(self.get_ext_fullpath("patminton.libpat_gpu")).resolve().parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "libpat_gpu.so"

        cmd = [
            nvcc,
            "-Xcompiler", "-fPIC",
            "-shared",
            f"-I{CUDA_DIR / 'include'}",
            "-o", str(out),
            *map(str, sources),
            "-lcufft",
            *gencode_flags(nvcc_major),
        ]
        if nvcc_major >= 13:
            cmd.append("-static-global-template-stub=false")

        print("Compiling CUDA library:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


class WheelTags(bdist_wheel or object):
    """Tag the wheel py3-none-<platform>.

    ``libpat_gpu.so`` is loaded with ctypes and links against no CPython
    symbols, so a single wheel is valid for every Python 3 interpreter. It is
    still platform-specific (and CUDA-major-specific).
    """

    def finalize_options(self):
        plat = os.environ.get("PATMINTON_PLAT_TAG")
        if plat:
            self.plat_name = plat
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        return "py3", "none", super().get_tag()[2]


cmdclass = {"build_ext": BuildLibPAT}
if bdist_wheel is not None:
    cmdclass["bdist_wheel"] = WheelTags

setup(
    ext_modules=[CUDAExtension("patminton.libpat_gpu")],
    cmdclass=cmdclass,
)
