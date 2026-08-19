# patminton

**GPU-accelerated forward and adjoint operators for 3D photoacoustic tomography.**

In photoacoustic tomography (PAT), a short laser pulse illuminates biological
tissue. The absorbed optical energy causes a rapid thermoelastic expansion
that generates a broadband ultrasonic wave, recorded at the surface by an
array of ultrasound transducers. The goal is to recover the 3D initial
pressure distribution $p(x)$ — which reflects the optical absorption of the
tissue — from the recorded time-domain signals $s(t)$.

`patminton` implements the forward operator (pressure → signals) and its adjoint
as **on-the-fly matrix-vector products** on the GPU, following the strategy
of Ding, Razansky and Deán-Ben (IEEE TMI, 2020). The system matrix is never
stored: for a 201³ grid, 10⁴ transducer positions and 1000 time samples it
would occupy 201³ × 10⁴ × 1000 × 8 B ≈ 800 TB in dense double precision.
This makes iterative model-based reconstruction of large 3D volumes
tractable on a single GPU.

## Features

- **Six transducer models** for cylindrical elements — from exact surface
  integrals via elliptic integrals (evaluated on-device with Carlson
  symmetric forms) to fast lookup-table and far-field approximations. See
  [Transducer models](transducer-models.md).
- **Forward model with instrument response**: convolution with the acoustic
  Green's function kernel, the laser pulse envelope and the measured
  Electronic Impulse Response (EIR), computed with batched cuFFT.
- **Adjoint consistency**: the adjoint mirrors the forward pass exactly
  (verified by dot-product tests), as required by iterative solvers.
- **Iterative solvers**: CGLS, L-BFGS-B (non-negativity constraint), projected
  gradient descent, and Chambolle-Pock with 3D total variation. See
  [Reconstruction algorithms](solvers.md).
- **PyTorch interface**: operators act on `torch.float64` CUDA tensors and
  support the `@` operator (`s = pat @ p`, `p = pat.T @ s`).

## Minimal example

```python
import torch
from patminton import PAT, translation_rotation_system, least_squares_CG

infos = translation_rotation_system(
    transducer_radius=25e-3, transducer_height=7.5e-3,
    transducer_width=0.250e-3, transducer_pitch=0.289e-3,
    transducer_nbr_elements=64, transducer_wavelength=1540/5e6,
    grid_size=10e-3,
)

pat = PAT(201, 201, 201, 5e-3, 5e-3, 5e-3,
          nT=1000, tStart=0.0, dt=25e-9, c=1540.0,
          mode='cylinder_lut', infos_transducers=infos)

p = torch.zeros((201, 201, 201), dtype=torch.float64, device='cuda')
p[100, 100, 100] = 1.0

s = pat @ p          # forward
p_bp = pat.T @ s     # adjoint
u, *_ = least_squares_CG(pat, s, M_inv=None, max_iter=50, lam=1e-4)
```

See the [Quickstart](quickstart.md) for a walk-through.

## Citation

If you use this package, please cite:

> Trung-Thai Do, Paul Escande, Caroline Chaux, Jérôme Gateau, Hwee Kuan Lee —
> *"Implementations of photoacoustic tomography models"*, 2026.

The matrix-vector product strategy follows:

> Lu Ding, Daniel Razansky, Xosé Luís Deán-Ben — *"Model-based reconstruction
> of large three-dimensional optoacoustic datasets"*, IEEE Transactions on
> Medical Imaging, 2020.
