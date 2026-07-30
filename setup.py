"""Build script: compiles the CUDA library libpat_gpu.so with nvcc.

The library is not a Python extension module — it is a plain shared library
loaded at runtime with ctypes — so we bypass the default compiler and invoke
nvcc directly, placing the .so inside the patas package.

Environment variables:
    PATAS_CUDA_ARCH  GPU architectures to compile for, e.g. "86" or
                     "70;80;86". The last one also gets embedded PTX for
                     forward compatibility. Default: "-arch=native"
                     (the GPU of the build machine; requires a visible GPU).
    PATAS_NVCC       Path to the nvcc binary (default: "nvcc" from PATH).
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parent
CUDA_DIR = ROOT / "src" / "patas" / "cuda"


def nvcc_major_version(nvcc):
    out = subprocess.run(
        [nvcc, "--version"], check=True, capture_output=True, text=True
    ).stdout
    m = re.search(r"release (\d+)\.", out)
    return int(m.group(1)) if m else 0


def gencode_flags():
    arch = os.environ.get("PATAS_CUDA_ARCH")
    if not arch:
        return ["-arch=native"]
    archs = [a.strip() for a in re.split(r"[;,\s]+", arch) if a.strip()]
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
        nvcc = shutil.which(os.environ.get("PATAS_NVCC", "nvcc"))
        if nvcc is None:
            raise RuntimeError(
                "nvcc not found. Installing patas requires the CUDA toolkit "
                "(https://developer.nvidia.com/cuda-toolkit). If nvcc is not "
                "on PATH, set the PATAS_NVCC environment variable."
            )

        sources = sorted((CUDA_DIR / "src").glob("*.cu")) + [CUDA_DIR / "libpat.cu"]
        out_dir = Path(self.get_ext_fullpath("patas.libpat_gpu")).resolve().parent
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
            *gencode_flags(),
        ]
        if nvcc_major_version(nvcc) >= 13:
            cmd.append("-static-global-template-stub=false")

        print("Compiling CUDA library:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


setup(
    ext_modules=[CUDAExtension("patas.libpat_gpu")],
    cmdclass={"build_ext": BuildLibPAT},
)
