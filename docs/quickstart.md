# Quickstart

This walk-through simulates the signals of a point absorber and reconstructs
it. A complete script is in
[`examples/point_phantom.py`](https://github.com/dotrungthai2001/patminton/blob/main/examples/point_phantom.py).

## Define the acquisition geometry

The scanner is a linear probe of cylindrical elements that rotates and
translates around the sample.
`translation_rotation_system()`
generates one 12-parameter row per element position:

```python
import numpy as np
import torch
from patminton import PAT, translation_rotation_system

c = 1540.0        # speed of sound (m/s)
Fc = 5e6          # transducer central frequency (Hz)

infos = translation_rotation_system(
    transducer_radius=25e-3,      # cylinder radius (m)
    transducer_height=7.5e-3,     # element height, arc direction (m)
    transducer_width=0.250e-3,    # element width, axis direction (m)
    transducer_pitch=0.289e-3,    # element spacing (m)
    transducer_nbr_elements=64,   # elements per probe position
    transducer_wavelength=c / Fc,
    grid_size=10e-3,              # extent to cover with translations (m)
)
print(infos.shape)                # (nTrans, 12)
```

## Instantiate the operator

```python
Nx = Ny = Nz = 100                # grid (voxels)
Lx = Ly = Lz = 5e-3               # grid half-extent (m)
nT, dt, tStart = 512, 25e-9, 0.0  # time axis

pat = PAT(
    Nx, Ny, Nz, Lx, Ly, Lz,
    nT, tStart, dt, c,
    mode="cylinder_lut",          # see Transducer models
    infos_transducers=infos,
    upsample=11,
    laser_pulse_variance=5e-9,
    eir=None,                     # or a measured EIR sampled at dt
)
```

Instantiation precomputes the frequency-domain kernels (cuFFT) and, for
`cylinder_lut`, builds the elliptic-integral lookup table.

## Forward and adjoint products

Tensors must be `torch.float64`, contiguous, and on the GPU.

```python
device = torch.device("cuda:0")

p = torch.zeros((Nx, Ny, Nz), dtype=torch.float64, device=device)
p[Nx // 2, Ny // 2, Nz // 2] = 1.0   # point absorber at the center

s = pat @ p                          # forward:  (nTrans, nT) signals
p_bp = pat.T @ s                     # adjoint:  (Nx, Ny, Nz) volume
```

The in-place variants `PAT.PMV()` /
`PAT.PMVT()`
write into preallocated tensors and avoid reallocation inside solver loops.

## Check the adjoint

```python
x = torch.rand((Nx, Ny, Nz), dtype=torch.float64, device=device)
y = torch.rand((pat.nTrans, nT), dtype=torch.float64, device=device)
lhs = torch.sum((pat @ x) * y)
rhs = torch.sum(x * (pat.T @ y))
print(f"relative gap: {abs(lhs - rhs) / abs(lhs):.2e}")   # ~1e-14
```

## Normalize the operator

In raw physical units (meters, seconds, pascals) the operator norm is tiny —
$\|A^\top A\| \sim 10^{-15}$ for the geometry above — so any usual
regularization weight would swamp the data term, and gradient step sizes are
off by orders of magnitude.
`normalize_operator()` rescales the
operator and the data to unit spectral norm; the minimizer is unchanged:

```python
from patminton import normalize_operator

pat_n, s_n, norm_A = normalize_operator(pat, s, p.shape)
```

## Reconstruct

```python
from patminton import least_squares_CG

u, F_list, SNR_list, SSIM_list, elapsed, _ = least_squares_CG(
    pat_n, s_n,
    M_inv=None,      # optional diagonal preconditioner
    max_iter=100,
    lam=1e-4,        # Tikhonov weight
    ref=p,           # ground truth -> tracks SNR/SSIM per iteration
    patience=10,     # early stopping
    verbose=True,
)
```

For non-negative reconstructions use
`least_squares_LBFGSB()` or
`least_squares_PGD()`; for total
variation use
`least_squares_CP_TV()`.
See [Reconstruction algorithms](solvers.md).
