# Installation

## Requirements

- An NVIDIA GPU with compute capability ≥ 7.5 and a driver ≥ 580 for the
  prebuilt wheel (CUDA 13); a source build with CUDA 12 also supports 7.0
  (V100) and drivers ≥ 525
- Python ≥ 3.9
- Linux x86-64 for the prebuilt wheel; any platform with the CUDA toolkit
  when building from source

Python dependencies (`numpy`, `scipy`, `torch`) are installed automatically.
PyTorch must be able to see the GPU (`torch.cuda.is_available()`).

## Install with pip

```bash
pip install patminton
```

On Linux x86-64 this installs a prebuilt wheel — no CUDA toolkit needed, only
the driver. The wheel is built with CUDA 13 and contains `libpat_gpu.so`
compiled for compute capabilities 7.5 to 12.0, plus PTX for newer GPUs.

`libpat_gpu.so` links dynamically against cuFFT (linking it statically costs
270 MB, above the PyPI file size limit). cuFFT comes from the CUDA toolkit if
one is installed, or from the cuFFT wheel that PyTorch pulls in; the default
PyTorch wheel (CUDA 13) provides the `libcufft.so.12` the prebuilt wheel
needs. Otherwise, e.g. with a CUDA 12 build of PyTorch, install it with the
extra matching the CUDA major version patminton was built with:

```bash
pip install patminton[cuda13]     # prebuilt wheel: libcufft.so.12
pip install patminton[cuda12]     # source build with CUDA 12: libcufft.so.11
```

### Building from source

On any other platform, or with `pip install --no-binary patminton`, pip
compiles the CUDA library. This needs the CUDA toolkit (≥ 11, tested with
12.x and 13.x) with `nvcc` on `PATH`; no GPU has to be visible at build time.
Expect about 70 s for the default architecture list.

For development, use an editable install from the repository root:

```bash
pip install -e .
```

## Choosing GPU architectures

By default the build targets every architecture supported by the `nvcc` in
use — 70, 75, 80, 86, 89, 90 for CUDA 12, and 75 to 120 for CUDA 13, which
dropped Volta. To build for a specific set of GPUs instead, e.g. to cut
compile time or to shrink the binary, set `PATMINTON_CUDA_ARCH` to a list of
compute capabilities:

```bash
# V100 (sm_70) + A100 (sm_80) + RTX 30xx (sm_86), with PTX for newer GPUs
PATMINTON_CUDA_ARCH="70;80;86" pip install .
```

Each listed architecture gets native SASS code; the last one also gets
embedded PTX, so the binary runs (after JIT compilation) on newer GPUs.
Use `PATMINTON_CUDA_ARCH=native` to compile for the GPU of the build machine
only — the fastest build, but it requires a visible GPU.

Other environment variables:

| Variable | Effect |
| --- | --- |
| `PATMINTON_NVCC` | Path to the `nvcc` binary if not on `PATH`. |
| `PATMINTON_LIB` | At runtime, load this `.so` instead of the one shipped with the package. |
| `PATMINTON_PLAT_TAG` | Platform tag of a wheel built with `python -m build --wheel`, e.g. `manylinux_2_28_x86_64`. |

## Manual build (without pip)

The CUDA library can also be compiled directly:

```bash
nvcc -Xcompiler -fPIC -shared \
     -Isrc/patminton/cuda/include \
     -o libpat_gpu.so src/patminton/cuda/src/*.cu src/patminton/cuda/libpat.cu \
     -lcufft -arch=native
export PATMINTON_LIB=$PWD/libpat_gpu.so
```

With CUDA ≥ 13, add `-static-global-template-stub=false`.

## Verify the installation

```python
import torch, patminton
pat = patminton.PAT(...)  # see Quickstart
```

A quick adjoint (dot-product) test validates the build: for random $p$ and
$s$, $\langle Ap, s\rangle = \langle p, A^\top s\rangle$ should hold to
~14 significant digits in double precision.

## Running the tests

```bash
pip install patminton[test]
pytest                   # CPU tests (geometry, metrics, solver utilities)
pytest -m gpu            # GPU tests (require a CUDA GPU and the compiled library)
```

The GPU tests are skipped automatically when no CUDA device or compiled
`libpat_gpu.so` is available.

## Building the documentation

```bash
pip install -r docs/requirements.txt
mkdocs serve     # live preview on http://127.0.0.1:8000
mkdocs build     # static site in site/
```

The documentation is plain Markdown and requires neither the compiled CUDA
library nor the Python dependencies.
