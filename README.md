# patas — GPU-Accelerated 3D Photoacoustic Tomography

CUDA implementations of the forward and adjoint operators of 3D
photoacoustic tomography (PAT) as on-the-fly matrix-vector products, for
iterative image reconstruction without storing the system matrix, plus
CGLS, L-BFGS-B, PGD and Chambolle-Pock TV solvers built on top of them.

For a 100³ grid observed by 10⁴ transducer positions with 512 time samples,
the dense system matrix would occupy 100³ × 10⁴ × 512 × 8 B ≈ 41 TB in double
precision; `patas` applies it and its adjoint on the fly on the GPU instead.

## Installation

Requirements: a CUDA-capable NVIDIA GPU, the CUDA toolkit (`nvcc` ≥ 11 on
PATH) and Python ≥ 3.9. Then:

```bash
pip install .
```

`pip` compiles the CUDA library for the GPU of the build machine
(`-arch=native`). To target other architectures (e.g. building on a cluster
login node for V100 + A100 compute nodes):

```bash
PATAS_CUDA_ARCH="70;80;86" pip install .
```

## Quick example

```python
import torch
from patas import PAT, translation_rotation_system, least_squares_CG

infos = translation_rotation_system(
    transducer_radius=25e-3, transducer_height=7.5e-3,
    transducer_width=0.250e-3, transducer_pitch=0.289e-3,
    transducer_nbr_elements=64, transducer_wavelength=1540/5e6,
    grid_size=10e-3,
)

pat = PAT(100, 100, 100, 5e-3, 5e-3, 5e-3,
          nT=512, tStart=0.0, dt=25e-9, c=1540.0,
          mode='cylinder_lut', infos_transducers=infos)

p = torch.zeros((100, 100, 100), dtype=torch.float64, device='cuda')
p[50, 50, 50] = 1.0

s = pat @ p                     # forward:  signals from initial pressure
p_bp = pat.T @ s                # adjoint:  back-propagation

u, *_ = least_squares_CG(pat, s, M_inv=None, max_iter=50, lam=1e-4)
```

## Documentation

The documentation (installation, quickstart, theory, transducer models,
solvers) is built with MkDocs:

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

## Reference

The on-the-fly matrix-vector product strategy follows:

> Lu Ding, Daniel Razansky, Xosé Luís Deán-Ben —
> *"Model-based reconstruction of large three-dimensional optoacoustic
> datasets"*, IEEE Transactions on Medical Imaging, 2020.

If you use this package, please cite:

> Trung-Thai Do, Paul Escande, Caroline Chaux, Jérôme Gateau, Hwee Kuan Lee —
> *"Implementations of photoacoustic tomography models"*, 2026.
