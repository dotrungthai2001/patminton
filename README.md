<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/dotrungthai2001/patminton/main/docs/assets/logo-dark.png">
    <img src="https://raw.githubusercontent.com/dotrungthai2001/patminton/main/docs/assets/logo.png" alt="PATminton — a GPU-accelerated Python toolbox for 3D photoacoustic tomography" width="600">
  </picture>
</p>

<p align="center">
  <a href="https://patminton.readthedocs.io/"><img src="https://readthedocs.org/projects/patminton/badge/?version=latest" alt="Documentation"></a>
</p>

# patminton — GPU-Accelerated 3D Photoacoustic Tomography

**PAT**minton: **PAT** **M**odels **I**mpleme**NT**ati**ON**.

Scalable GPU implementations of 3D photoacoustic tomography (PAT) models
that account for the transducer spatial impulse response. The forward and
adjoint operators are computed on the fly as matrix-vector products, so
iterative image reconstruction never stores the system matrix. CGLS,
L-BFGS-B, PGD and Chambolle-Pock TV solvers are built on top of them.

For a 200³ grid observed by 11 520 transducer positions with 1000 time samples,
the dense system matrix would occupy 200³ × 11 520 × 1000 × 8 B ≈ 737 TB in double
precision; `patminton` applies it and its adjoint on the fly on the GPU instead.

## Installation

```bash
pip install patminton
```

Requirements: a CUDA-capable NVIDIA GPU (compute capability ≥ 7.0), an NVIDIA
driver ≥ 525, and Python ≥ 3.9. On Linux x86-64 this installs a prebuilt wheel
built with CUDA 12.8, which needs no CUDA toolkit and works with both CUDA 12
and CUDA 13 drivers and PyTorch builds. Its cuFFT dependency
(`libcufft.so.11`) is installed from PyPI.

Installing from source instead requires the CUDA toolkit (`nvcc` ≥ 11 on
PATH), which compiles the library for every supported architecture. To target
a specific set of GPUs (e.g. V100 + A100 + RTX 30xx):

```bash
PATMINTON_CUDA_ARCH="70;80;86" pip install --no-binary patminton patminton
```

## Quick example

```python
import torch
from patminton import PAT, translation_rotation_system, least_squares_CG

infos = translation_rotation_system(
    transducer_radius=25e-3, transducer_height=7.5e-3,
    transducer_width=0.250e-3, transducer_pitch=0.298e-3,
    transducer_nbr_elements=64, transducer_wavelength=1500/5e6,
    grid_size=10e-3,
)

pat = PAT(200, 200, 200, 5e-3, 5e-3, 5e-3,
          nT=1024, tStart=12.5e-6, dt=16e-9, c=1500.0,
          mode='cylinder_lut', infos_transducers=infos)

p = torch.zeros((200, 200, 200), dtype=torch.float64, device='cuda')
p[100, 100, 100] = 1.0

s = pat @ p                     # forward:  signals from initial pressure
p_bp = pat.T @ s                # adjoint:  back-propagation

u, *_ = least_squares_CG(pat, s, M_inv=None, max_iter=50, lam=1e-4)
```

## Documentation

**[patminton.readthedocs.io](https://patminton.readthedocs.io/)** — installation, quickstart, physical and
mathematical model, transducer models, reconstruction algorithms and key
parameters.

To preview it locally:

```bash
pip install -r docs/requirements.txt
mkdocs serve      # live preview on http://127.0.0.1:8000
```

## Tests

```bash
pip install .[test]
pytest              # CPU tests (no GPU needed)
pytest -m gpu       # operator tests on a CUDA GPU
```

## Citation

If you use this package, please cite:

> Trung-Thai Do, Paul Escande, Caroline Chaux, Jérôme Gateau, Hwee Kuan Lee —
> *"Scalable implementations of photoacoustic tomography models accounting for
> transducers spatial impulse response"*, 2026.
