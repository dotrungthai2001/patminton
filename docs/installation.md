# Installation

## Requirements

- An NVIDIA GPU with compute capability ≥ 7.0
- CUDA toolkit ≥ 11 with `nvcc` on `PATH` (tested with 12.x and 13.x); cuFFT
  ships with the toolkit
- Python ≥ 3.9

Python dependencies (`numpy`, `scipy`, `torch`) are installed automatically.
PyTorch must be able to see the GPU (`torch.cuda.is_available()`).

## Install with pip

From the repository root:

```bash
pip install .
```

`pip` invokes `nvcc` to compile the CUDA library `libpat_gpu.so` and places
it inside the installed package. By default it compiles for the GPU of the
build machine (`-arch=native`), which requires a visible GPU at build time.

For development, use an editable install:

```bash
pip install -e .
```

## Choosing GPU architectures

To build for other GPUs than the build machine — e.g. compiling on a cluster
login node without a GPU, or producing one binary for several node types —
set `PATAS_CUDA_ARCH` to a list of compute capabilities:

```bash
# V100 (sm_70) + A100 (sm_80) + RTX 30xx (sm_86), with PTX for newer GPUs
PATAS_CUDA_ARCH="70;80;86" pip install .
```

Each listed architecture gets native SASS code; the last one also gets
embedded PTX, so the binary runs (after JIT compilation) on newer GPUs.

Other environment variables:

| Variable | Effect |
| --- | --- |
| `PATAS_NVCC` | Path to the `nvcc` binary if not on `PATH`. |
| `PATAS_LIB` | At runtime, load this `.so` instead of the one shipped with the package. |

## Manual build (without pip)

The CUDA library can also be compiled directly:

```bash
nvcc -Xcompiler -fPIC -shared \
     -Isrc/patas/cuda/include \
     -o libpat_gpu.so src/patas/cuda/src/*.cu src/patas/cuda/libpat.cu \
     -lcufft -arch=native
export PATAS_LIB=$PWD/libpat_gpu.so
```

With CUDA ≥ 13, add `-static-global-template-stub=false`.

## Verify the installation

```python
import torch, patas
pat = patas.PAT(...)  # see Quickstart
```

A quick adjoint (dot-product) test validates the build: for random $p$ and
$s$, $\langle Ap, s\rangle = \langle p, A^\top s\rangle$ should hold to
~14 significant digits in double precision.

## Running the tests

```bash
pip install .[test]
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
