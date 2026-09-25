"""Point-phantom example: forward simulation, adjoint test, L-BFGS-B reconstruction.

Runs on a small 64^3 grid. patminton computes in float64, so even this takes
minutes on a data-centre GPU (V100, A100, H100); consumer GPUs run float64 at a
small fraction of their float32 rate and are much slower.
"""

import torch

from patminton import (
    PAT,
    least_squares_LBFGSB,
    normalize_operator,
    translation_rotation_system,
)

# --- acquisition geometry -------------------------------------------------
c = 1500.0   # speed of sound in water (m/s)
Fc = 5e6     # transducer central frequency (Hz)

infos = translation_rotation_system(
    transducer_radius=25e-3,
    transducer_height=7.5e-3,
    transducer_width=0.250e-3,
    transducer_pitch=0.298e-3,
    transducer_nbr_elements=64,
    transducer_wavelength=c / Fc,
    grid_size=5e-3,
)
print(f"{infos.shape[0]} transducer positions")

# --- operator -------------------------------------------------------------
Nx = Ny = Nz = 64
Lx = Ly = Lz = 2.5e-3
# Grid-to-surface distances span 21.7-36.8 mm here, i.e. 14.5-24.6 us at
# c = 1500; the time axis is gated to that window (Fs = 62.5 MHz).
nT, dt, tStart = 640, 16e-9, 14.4e-6

pat = PAT(
    Nx, Ny, Nz, Lx, Ly, Lz,
    nT, tStart, dt, c,
    mode="cylinder_lut",
    infos_transducers=infos,
    upsample=11,
    laser_pulse_variance=5e-9,
)

# --- forward simulation ---------------------------------------------------
device = torch.device("cuda:0")
p = torch.zeros((Nx, Ny, Nz), dtype=torch.float64, device=device)
p[Nx // 2, Ny // 2, Nz // 2] = 1.0

s = pat @ p
print(f"signals: {tuple(s.shape)}, max |s| = {s.abs().max():.3e}")

# --- adjoint (dot-product) test -------------------------------------------
x = torch.rand((Nx, Ny, Nz), dtype=torch.float64, device=device)
y = torch.rand((pat.nTrans, nT), dtype=torch.float64, device=device)
lhs = torch.sum((pat @ x) * y)
rhs = torch.sum(x * (pat.T @ y))
print(f"adjoint relative gap: {abs(lhs - rhs) / abs(lhs):.2e}")

# --- reconstruction -------------------------------------------------------
# In raw physical units ||A|| ~ 3e-6 here, so any usual Tikhonov weight would
# swamp the data term; rescale the operator to unit norm first. The norm only
# sets the scale: 5 power iterations suffice, as in the paper's runs.
pat_n, s_n, norm_A = normalize_operator(pat, s, p.shape, n_iter=5)
print(f"operator norm: {norm_A:.3e}")

# Non-negative least squares (u >= 0) with Tikhonov weight lam, the solver
# used in the paper. Each function evaluation costs one A and one A^T.
status = {}
u, F_list, SNR_list, SSIM_list, elapsed, _ = least_squares_LBFGSB(
    pat_n, s_n, M_inv=None, max_iter=30, lam=1e-4, ref=p, verbose=True,
    status=status,
)
print(f"L-BFGS-B: {status['n_iter']} iterations, {status['nfev']} evaluations "
      f"in {elapsed:.1f} s, final SNR = {SNR_list[-1]:.2f} dB")
