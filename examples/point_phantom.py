"""Point-phantom example: forward simulation, adjoint test, CGLS reconstruction.

Runs on a small 64^3 grid so it fits on a laptop GPU in a few seconds.
"""

import torch

from patas import (
    PAT,
    least_squares_CG,
    normalize_operator,
    translation_rotation_system,
)

# --- acquisition geometry -------------------------------------------------
c = 1540.0   # speed of sound (m/s)
Fc = 5e6     # transducer central frequency (Hz)

infos = translation_rotation_system(
    transducer_radius=25e-3,
    transducer_height=7.5e-3,
    transducer_width=0.250e-3,
    transducer_pitch=0.289e-3,
    transducer_nbr_elements=64,
    transducer_wavelength=c / Fc,
    grid_size=5e-3,
)
print(f"{infos.shape[0]} transducer positions")

# --- operator -------------------------------------------------------------
Nx = Ny = Nz = 64
Lx = Ly = Lz = 2.5e-3
nT, dt, tStart = 512, 25e-9, 12e-6

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
# In raw physical units ||A^T A|| ~ 1e-15, so any usual Tikhonov weight
# would swamp the data term; rescale the operator to unit norm first.
pat_n, s_n, norm_A = normalize_operator(pat, s, p.shape)
print(f"operator norm: {norm_A:.3e}")

u, F_list, SNR_list, SSIM_list, elapsed, _ = least_squares_CG(
    pat_n, s_n, M_inv=None, max_iter=30, lam=1e-4, ref=p, patience=10, verbose=True
)
print(f"CGLS: {len(F_list)} iterations in {elapsed:.1f} s, "
      f"final SNR = {SNR_list[-1]:.2f} dB")
