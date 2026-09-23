"""Iterative reconstruction algorithms and image-quality metrics.

All solvers take a :class:`patminton.PAT` operator ``pat`` and the measured
signals ``s_meas`` (a ``(nTrans, nT)`` float64 CUDA tensor), and share the
return signature ``(u, F_list, SNR_list, SSIM_list, elapsed,
SNR_planes_history)`` where ``u`` is the reconstructed volume and the lists
track the cost and metrics per iteration (metrics require the ``ref``
ground-truth argument).

Solvers:

- :func:`least_squares_CG` — CGLS, Tikhonov, unconstrained.
- :func:`least_squares_LBFGSB` — L-BFGS-B, Tikhonov, ``u >= 0``, with
  checkpoint/resume support for long HPC runs.
- :func:`least_squares_PGD` — projected gradient descent, ``u >= 0``.
- :func:`least_squares_CP_TV` — Chambolle-Pock primal-dual with isotropic
  3D total variation, ``u >= 0``.

Metrics: :func:`SNR`, :func:`PSNR`, :func:`compute_ssim_3d`, plus per-plane
and per-region variants.
"""

import os
import torch
import time
import scipy.optimize
import numpy as np
import torch.nn.functional as F

# =============================================================================
# Metrics
# =============================================================================

def SNR(xref, xest):
    """Signal-to-Noise Ratio in dB."""
    return 10 * torch.log10(torch.sum(xref ** 2) / torch.sum((xest - xref) ** 2))


def PSNR(xref, xest, data_range=None):
    """Peak Signal-to-Noise Ratio in dB: 10 log10(data_range^2 / MSE).

    `data_range` defaults to ``xref.max() - xref.min()``.
    """
    if data_range is None:
        data_range = xref.max() - xref.min()
    mse = torch.mean((xest - xref) ** 2)
    return 10 * torch.log10(data_range ** 2 / mse)


def SNR_per_plane(xref, xest, dim=2):
    """
    Compute SNR for each plane along a given dimension.

    Args:
        xref:  Reference image (Nx, Ny, Nz)
        xest:  Estimated image (Nx, Ny, Nz)
        dim:   Dimension to slice along (0=x, 1=y, 2=z)

    Returns:
        snr_planes: Tensor of SNR values (one per plane)
    """
    if dim not in (0, 1, 2):
        raise ValueError("dim must be 0, 1, or 2")

    num_planes = xref.shape[dim]
    snr_planes = torch.zeros(num_planes, device=xref.device)

    for k in range(num_planes):
        sl = [slice(None)] * 3
        sl[dim] = k
        snr_planes[k] = SNR(xref[sl[0], sl[1], sl[2]],
                            xest[sl[0], sl[1], sl[2]])
    return snr_planes

# -----------------------------------------------------------------------------
# Per-region (per-piece) SNR
# -----------------------------------------------------------------------------
# Divide the 3-D phantom into sub-volumes and measure SNR in each one, so the
# reconstruction quality can be read off locally instead of as a single global
# number -- e.g. inside the 1 mm transducer focal zone, near the edges, in the
# corners, or over a regular grid of tiles for a spatial quality heat-map.

def SNR_masked(xref, xest, mask, eps=1e-30):
    """
    SNR (dB) restricted to the voxels selected by a boolean mask.

    Returns NaN if the mask is empty or the reference carries no energy there
    (a region with no signal has no meaningful SNR), so downstream plots/tables
    can skip it instead of showing +/-inf.

    Args:
        xref:  Reference volume (Nx, Ny, Nz)
        xest:  Estimated volume  (Nx, Ny, Nz)
        mask:  Boolean tensor (Nx, Ny, Nz); True = include the voxel
    """
    mask = mask.to(xref.device)
    if mask.dtype != torch.bool:
        mask = mask != 0
    if int(mask.sum()) == 0:
        return float('nan')

    ref = xref[mask]
    est = xest[mask]
    signal = torch.sum(ref ** 2)
    error  = torch.sum((est - ref) ** 2)
    if signal <= eps:
        return float('nan')
    return (10 * torch.log10(signal / (error + eps))).item()


def _axis_coords(shape, extent, device='cpu', dtype=torch.float64):
    """
    Physical coordinate axis for each dimension of a centered grid.

    Matches `precond_distance`: axis k spans [-L_k, L_k] with N_k samples, so
    `extent = (Lx, Ly, Lz)` are the half-extents in metres.
    """
    Nx, Ny, Nz = shape
    Lx, Ly, Lz = extent
    xs = torch.linspace(-Lx, Lx, Nx, device=device, dtype=dtype)
    ys = torch.linspace(-Ly, Ly, Ny, device=device, dtype=dtype)
    zs = torch.linspace(-Lz, Lz, Nz, device=device, dtype=dtype)
    return xs, ys, zs


def grid_tile_masks(shape, tiles=(3, 3, 3), device='cpu'):
    """
    Partition the volume into a `tiles = (tx, ty, tz)` grid of equal index
    blocks. Returns an ordered dict {'tile_i_j_k': bool mask}.

    Purely index-based (no physical units needed) -- ideal for a spatial SNR
    heat-map over the whole phantom.
    """
    Nx, Ny, Nz = shape
    tx, ty, tz = tiles
    xb = torch.linspace(0, Nx, tx + 1).round().long().tolist()
    yb = torch.linspace(0, Ny, ty + 1).round().long().tolist()
    zb = torch.linspace(0, Nz, tz + 1).round().long().tolist()

    masks = {}
    for i in range(tx):
        for j in range(ty):
            for k in range(tz):
                m = torch.zeros(shape, dtype=torch.bool, device=device)
                m[xb[i]:xb[i + 1], yb[j]:yb[j + 1], zb[k]:zb[k + 1]] = True
                masks[f'tile_{i}_{j}_{k}'] = m
    return masks


def roi_masks(shape, extent, center=(0.0, 0.0, 0.0),
              focal_radius=1e-3, center_halfsize=None,
              edge_margin=None, corner_size=None,
              per_corner=False, device='cpu', dtype=torch.float64):
    """
    Named physical regions of interest on a centered grid.

    All lengths are in metres; `extent = (Lx, Ly, Lz)` are the half-extents
    (axis k spans [-L_k, L_k]), same convention as `precond_distance`.

    Args:
        center:          (cx, cy, cz) focal point for the focal sphere, in metres.
        focal_radius:    radius of the 'focal' sphere (default 1 mm).
        center_halfsize: half-side of the central box; defaults to focal_radius.
        edge_margin:     thickness of the boundary shell; defaults to focal_radius.
        corner_size:     side of each corner cube; defaults to 2*focal_radius.
        per_corner:      if True, emit the 8 corners separately ('corner_+--',
                         signs over x,y,z) instead of a single merged 'corner'.

    Returns:
        Ordered dict {name: bool mask}:
            'focal'  : sphere of radius focal_radius around center
            'center' : central box of half-size center_halfsize
            'edge'   : outer shell within edge_margin of any face
            'corner' : union of the 8 corner cubes  (or 'corner_sss' if per_corner)
    """
    Lx, Ly, Lz = extent
    if center_halfsize is None:
        center_halfsize = focal_radius
    if edge_margin is None:
        edge_margin = focal_radius
    if corner_size is None:
        corner_size = 2.0 * focal_radius

    xs, ys, zs = _axis_coords(shape, extent, device=device, dtype=dtype)
    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing='ij')

    masks = {}

    # focal zone: sphere around the focal point
    cx, cy, cz = center
    r2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
    masks['focal'] = r2 <= focal_radius ** 2

    # central box (centered on the grid origin, not the focal point)
    masks['center'] = ((X.abs() <= center_halfsize) &
                       (Y.abs() <= center_halfsize) &
                       (Z.abs() <= center_halfsize))

    # boundary shell: within edge_margin of any face
    near_x = (X <= -Lx + edge_margin) | (X >= Lx - edge_margin)
    near_y = (Y <= -Ly + edge_margin) | (Y >= Ly - edge_margin)
    near_z = (Z <= -Lz + edge_margin) | (Z >= Lz - edge_margin)
    masks['edge'] = near_x | near_y | near_z

    # corner cubes: within corner_size of a corner along every axis
    corner_union = torch.zeros(shape, dtype=torch.bool, device=device)
    for sx, lx in ((-1, -Lx), (1, Lx)):
        for sy, ly in ((-1, -Ly), (1, Ly)):
            for sz, lz in ((-1, -Lz), (1, Lz)):
                cube = (((X - lx).abs() <= corner_size) &
                        ((Y - ly).abs() <= corner_size) &
                        ((Z - lz).abs() <= corner_size))
                if per_corner:
                    tag = ''.join('+' if s > 0 else '-' for s in (sx, sy, sz))
                    masks[f'corner_{tag}'] = cube
                else:
                    corner_union |= cube
    if not per_corner:
        masks['corner'] = corner_union

    return masks


def SNR_per_region(xref, xest, regions, return_counts=False):
    """
    Compute SNR (dB) for each named region of the phantom.

    Args:
        xref, xest:    Reference / estimated volumes (Nx, Ny, Nz)
        regions:       dict {name: bool mask}, e.g. from `roi_masks` or
                       `grid_tile_masks` (merge several with `dict(**a, **b)`).
        return_counts: also return voxel counts (small regions are noisy).

    Returns:
        snr:    dict {name: snr_dB_float}  (NaN for empty / signal-free regions)
        counts: dict {name: n_voxels}      (only if return_counts=True)
    """
    snr = {name: SNR_masked(xref, xest, mask) for name, mask in regions.items()}
    if return_counts:
        counts = {name: int(mask.to(xref.device).bool().sum())
                  for name, mask in regions.items()}
        return snr, counts
    return snr

    
def gaussian_kernel_3d(kernel_size=11, sigma=1.5, device='cpu', dtype=torch.float64):
    """Separable 3-D Gaussian kernel, normalised to sum = 1."""
    coords = torch.arange(kernel_size, dtype=dtype, device=device) - kernel_size // 2
    g      = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g      = g / g.sum()
    return g[:, None, None] * g[None, :, None] * g[None, None, :]

def compute_ssim_3d(xref, xest, K=(0.01, 0.03), kernel_size=11, sigma=1.5,
                         data_range=1.0, downsample=True):
    """
    3D SSIM following Wang et al. (with automatic downsampling)

    Parameters
    ----------
    xref, xest : (Nx,Ny,Nz) or (1,1,D,H,W)
    K : tuple (K1, K2)
    data_range : L (important -> fix to 1.0 if normalized)
    """

    # ---- reshape ----
    if xref.ndim == 3:
        xref = xref.unsqueeze(0).unsqueeze(0)
        xest = xest.unsqueeze(0).unsqueeze(0)

    device = xref.device
    dtype = xref.dtype

    _, _, D, H, W = xref.shape

    # --------------------------------------------------
    # 1. Automatic downsampling
    # --------------------------------------------------
    if downsample:
        f = max(1, round(min(D, H, W) / 256))

        if f > 1:
            # simple average filter (LPF)
            lpf = torch.ones((1, 1, f, f, f), device=device, dtype=dtype)
            lpf = lpf / lpf.sum()

            pad = f // 2
            xref = F.conv3d(F.pad(xref, (pad, pad, pad, pad, pad, pad), mode='reflect'), lpf)
            xest = F.conv3d(F.pad(xest, (pad, pad, pad, pad, pad, pad), mode='reflect'), lpf)

            # subsample
            xref = xref[:, :, ::f, ::f, ::f]
            xest = xest[:, :, ::f, ::f, ::f]
            
    # --------------------------------------------------
    # 1b. Adapt kernel to volume size
    # --------------------------------------------------
    # VALID conv requires every axis >= kernel_size; shrink the kernel and
    # rescale sigma proportionally for small volumes (e.g. coarse z-axis).
    _, _, D, H, W = xref.shape
    min_dim = min(D, H, W)
    if kernel_size > min_dim:
        new_size = min_dim if min_dim % 2 == 1 else min_dim - 1
        sigma = sigma * new_size / kernel_size
        kernel_size = new_size
        
    # --------------------------------------------------
    # 2. Constants
    # --------------------------------------------------
    K1, K2 = K
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # --------------------------------------------------
    # 3. Gaussian window
    # --------------------------------------------------
    kernel = gaussian_kernel_3d(kernel_size, sigma, device=device, dtype=dtype)
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    # --------------------------------------------------
    # 4. VALID convolution (critical difference)
    # --------------------------------------------------
    # no padding → matches MATLAB 'valid'
    mu_x = F.conv3d(xref, kernel)
    mu_y = F.conv3d(xest, kernel)

    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv3d(xref * xref, kernel) - mu_x2
    sigma_y2 = F.conv3d(xest * xest, kernel) - mu_y2
    sigma_xy = F.conv3d(xref * xest, kernel) - mu_xy

    # numerical stability
    sigma_x2 = torch.clamp(sigma_x2, min=0.0)
    sigma_y2 = torch.clamp(sigma_y2, min=0.0)

    # --------------------------------------------------
    # 5. SSIM map
    # --------------------------------------------------
    if C1 > 0 and C2 > 0:
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    else:
        numerator1 = 2 * mu_xy + C1
        numerator2 = 2 * sigma_xy + C2
        denominator1 = mu_x2 + mu_y2 + C1
        denominator2 = sigma_x2 + sigma_y2 + C2

        ssim_map = torch.ones_like(mu_x)
        mask = (denominator1 * denominator2 > 0)
        ssim_map[mask] = (numerator1[mask] * numerator2[mask]) / \
                         (denominator1[mask] * denominator2[mask])

    mssim = ssim_map.mean()

    return mssim#, ssim_map

def compute_metrics(ref, xest, track_planes=False, plane_dim=2):
    """
    Compute SNR, SSIM, and optionally per-plane SNR for a single estimate.

    Args:
        ref:          Ground-truth volume
        u_eval:       Current estimate (NOT clamped — caller decides)
        track_planes: Also compute per-plane SNR
        plane_dim:    Dimension to slice for per-plane SNR

    Returns:
        snr:         Scalar float
        ssim:        Scalar float
        snr_planes:  CPU tensor (num_planes,) or None
    """
    snr        = SNR(ref, xest).item()
    ssim       = compute_ssim_3d(ref, xest).item()
    snr_planes = SNR_per_plane(ref, xest, dim=plane_dim).cpu() if track_planes else None
    return snr, ssim, snr_planes

# =============================================================================
# Shared utilities
# =============================================================================

def apply_precond(x, M_inv, shape, device, dtype):
    """Apply diagonal preconditioner M_inv (or identity if None)."""
    if M_inv is None:
        return x
    return M_inv.reshape(shape).to(device=device, dtype=dtype) * x

def estimate_lipschitz(pat, shape, device, dtype, lam=0.0, n_iter=30, seed=0):
    """
    Estimate L = ||A^TA + lam*I|| via power iteration.
    Returns the spectral norm as a Python float.

    `seed`: if not None, the power-iteration start is deterministic, so norm_A is
    REPRODUCIBLE across runs. This matters: normalize_operator scales A by
    1/norm_A, so the solver's effective regularization is lam*norm_A^2 -- a random
    norm_A would make the same `lam` a different problem each run (different
    conditioning, speed, and result), confounding any lambda sweep.

    Note: PAT.T mutates `self.transpose` and returns `self`, so a nested
    expression `pat.T @ (pat @ v)` would read the flag set by `.T` while
    evaluating the inner forward and silently do two adjoints. Each matvec
    therefore lives on its own statement.
    """
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(int(seed))
        v = torch.randn(shape, device=device, dtype=dtype, generator=gen)
    else:
        v = torch.randn(shape, device=device, dtype=dtype)
    v = v / v.norm()
    L = torch.ones(1, device=device, dtype=dtype)
    for _ in range(n_iter):
        Av = pat @ v
        w  = pat.T @ Av + lam * v
        L = w.norm()
        v = w / L
    return L.item()


class NormalizedPAT:
    """Wrap a PAT operator A so it acts as A_tilde = A / norm.

    Pair with rescaled data b_tilde = b / norm: the minimiser of
    ||A_tilde u - b_tilde||^2 equals that of ||A u - b||^2, but the operator
    spectral norm is ~1, which puts Lipschitz step sizes and Tikhonov / TV
    regularisation weights on a sane scale instead of ~1e-15.

    Mimics PAT's transpose protocol: `.T` sets a flag and returns self, then
    `@` consumes the flag. Each matvec lives on its own statement, same as
    in estimate_lipschitz.
    """

    def __init__(self, pat, norm):
        self.pat = pat
        self.norm = float(norm)
        self._transpose = False
        # mirror common attributes so callers like the solvers can introspect
        self.Nx, self.Ny, self.Nz = pat.Nx, pat.Ny, pat.Nz
        self.nT, self.nTrans = pat.nT, pat.nTrans

    def __matmul__(self, x):
        if self._transpose:
            self._transpose = False
            out = self.pat.T @ x
        else:
            out = self.pat @ x
        return out / self.norm

    @property
    def T(self):
        self._transpose = True
        return self


def normalize_operator(pat, s_meas, shape, lam=0.0, n_iter=30, seed=0):
    """Return (A_tilde, s_rescaled, norm_A) for a unit-norm reconstruction.

    norm_A = sqrt(||A^T A||) is estimated with `estimate_lipschitz`. The
    wrapped operator A_tilde = A / norm_A and the rescaled measurement
    s / norm_A are drop-in replacements for the solvers in this module --
    the recovered u is the same, just on a well-conditioned scale.

    `seed` makes norm_A reproducible across runs (see estimate_lipschitz) so a
    lambda sweep varies only lambda, not the random operator scaling.
    """
    L_A = estimate_lipschitz(pat, shape, s_meas.device, s_meas.dtype,
                             lam=lam, n_iter=n_iter, seed=seed)
    norm_A = L_A ** 0.5
    return NormalizedPAT(pat, norm_A), s_meas / norm_A, norm_A


def _estimate_lipschitz_preconditioned(pat, M_inv_t, shape, device, dtype,
                                        lam=0.0, n_iter=30):
    """
    Estimate L = ||D^-1(A^TA + lamI)D^-1|| where D^-2 = M_inv (so D^-1 = sqrt(M_inv)).

    This is the spectral norm of the *preconditioned* operator that PGD/FISTA
    descend along when M_inv != None -- using it sets the correct step size
    1/L for variable-metric proximal-gradient steps.
    """
    sqrt_M_inv = torch.sqrt(M_inv_t)
    v = torch.randn(shape, device=device, dtype=dtype)
    v = v / (v.norm() + 1e-30)
    L = torch.ones(1, device=device, dtype=dtype)
    for _ in range(n_iter):
        w     = sqrt_M_inv * v
        Aw    = pat @ w
        AtAw  = pat.T @ Aw + lam * w
        out   = sqrt_M_inv * AtAw
        L     = out.norm()
        v     = out / (L + 1e-30)
    return L.item()
    
def _init_history(track_planes):
    """Return empty history containers."""
    return [], [], [], ([] if track_planes else None)


def _append_metrics(snr, ssim, snr_planes, SNR_list, SSIM_list, SNR_planes_history):
    """Push one iteration's metrics into the history lists."""
    SNR_list.append(snr)
    SSIM_list.append(ssim)
    if SNR_planes_history is not None and snr_planes is not None:
        SNR_planes_history.append(snr_planes)

def _check_early_stop(snr, best_snr, best_u, u, best_it, snr_no_improve, patience, i, verbose, tag):
    """
    Update best-SNR bookkeeping and check patience criterion.

    Returns (best_snr, best_u, best_it, snr_no_improve, stop)
    """
    if snr > best_snr:
        best_snr, best_u, best_it, snr_no_improve = snr, u.clone(), i + 1, 0
    else:
        snr_no_improve += 1

    stop = patience is not None and snr_no_improve >= patience
    if stop and verbose:
        print(f"[{tag}] Early stop at iter {i+1} | Best SNR={best_snr:.3f} dB at iter {best_it}")

    return best_snr, best_u, best_it, snr_no_improve, stop

    
def precond_distance(points, Nx, Ny, Nz, Lx, Ly, Lz, device):
    """
    Diagonal Jacobi-style preconditioner M_inv ~= diag(A^TA)^-1 for PAT.

    Approximates (A^TA)_{ii} via inverse-square distance summed over detectors
    (because pressure decays as 1/r), then returns its reciprocal so that
    apply_precond(r, M_inv) computes  M_inv * r  ~=  diag(A^TA)^-1 r.

    Returns:
        M_inv: flat tensor of length Nx*Ny*Nz (to be passed as `M_inv` to a
        solver). Values are large for voxels far from any detector.
    """
    xs_v = torch.linspace(-Lx, Lx, Nx, device=device)
    ys_v = torch.linspace(-Ly, Ly, Ny, device=device)
    zs_v = torch.linspace(-Lz, Lz, Nz, device=device)

    X, Y, Z = torch.meshgrid(xs_v, ys_v, zs_v, indexing='ij')

    coords = torch.tensor(points[:, 0, :], dtype=torch.float64, device=device)
    xd, yd, zd = coords[:, 0], coords[:, 1], coords[:, 2]

    diag_AtA = torch.zeros((Nx, Ny, Nz), device=device)
    for i in range(len(xd)):
        r2 = (X - xd[i])**2 + (Y - yd[i])**2 + (Z - zd[i])**2
        diag_AtA += 1.0 / (r2 + 1e-6)

    M_inv = 1.0 / (diag_AtA + 1e-30)
    return M_inv.reshape(-1)
    
class _EarlyStop(Exception):
    """Raised inside scipy callbacks to interrupt optimization early."""
    pass

# =============================================================================
# Algorithms
# =============================================================================

# Conjugate gradient with Tikhonov regu
def least_squares_CG(pat, s_meas, M_inv, max_iter=300, tol=1e-6, lam=0.0,
                     ref=None, verbose=True, patience=None,
                     track_planes=False, plane_dim=2):
    """
    Conjugate Gradient Least Squares (CGLS) with Tikhonov regularization.
    Solves:  min  0.5*||Au - s||^2  +  0.5*lam*||u||^2

    Returns:
        u, F_list, SNR_list, SSIM_list, elapsed, SNR_planes_history
    """
    device = s_meas.device
    dtype = torch.float64
    
    # Compute b = A.T s
    b = pat.T @ s_meas
    
    # Initialization
    u = torch.zeros_like(b)
    r = b.clone()
    
    z = apply_precond(r, M_inv, u.shape, device, dtype)
    p = z.clone()
    rz_old = torch.sum(r * z)
    
    # History tracking
    F_list, SNR_list, SSIM_list, SNR_planes_history = _init_history(track_planes)

    # Early stopping
    best_snr, best_u, best_it, snr_no_improve = -float('inf'), None, 0, 0

    def apply_AtA_reg(x):
        """Apply (A^T A + lam I) to vector x."""
        Ax = pat @ x
        AtAx = pat.T @ Ax
        if lam > 0:
            AtAx = AtAx + lam * x
        return AtAx
        
    # Initial cost (proxy: squared gradient = ||b - A^TAu||2; at u=0 reduces to ||b||2)
    F_list.append(0.5 * torch.sum(r * r).item())
    
    if ref is not None:
        snr, ssim, snr_planes = compute_metrics(ref, u, track_planes, plane_dim)
        _append_metrics(snr, ssim, snr_planes, SNR_list, SSIM_list, SNR_planes_history)
        best_snr, best_u = snr, u.clone()

    norm_b = torch.norm(b)
    tic    = time.time()
    
    for i in range(max_iter):
        # CG step
        AtAp  = apply_AtA_reg(p)
        alpha = rz_old / torch.sum(p * AtAp)
        
        u = u + alpha * p
        r = r - alpha * AtAp
        
        # Convergence check
        rel_residual = torch.norm(r) / norm_b
        if rel_residual < tol:
            if verbose:
                print(f"[CGLS] Converged at iteration {i+1}: rel_residual={rel_residual:.3e}")
            break
        
        # Preconditioning
        z = apply_precond(r, M_inv, u.shape, device, dtype)        
        rz_new = torch.sum(r * z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
        
        # Cost:  0.5*(||r_normal||^2 + lam*||u||^2) as proxy
        cost = 0.5 * torch.sum(r * r).item()
        if lam > 0:
            cost += 0.5 * lam * torch.sum(u * u).item()
        F_list.append(cost)
        
        # Metrics
        if ref is not None:
            snr, ssim, snr_planes = compute_metrics(ref, u, track_planes, plane_dim)

            _append_metrics(snr, ssim, snr_planes, SNR_list, SSIM_list, SNR_planes_history)
            best_snr, best_u, best_it, snr_no_improve, stop = _check_early_stop(
                snr, best_snr, best_u, u, best_it, snr_no_improve, patience, i, verbose, 'CGLS')
            if stop:
                break

        # Progress print
        if verbose and (i % 10 == 0 or i == max_iter - 1):
            msg = f"[CGLS] Iter {i+1:03d}/{max_iter} | rel_res={rel_residual:.3e}"
            if ref is not None:
                msg += f" | SNR={SNR_list[-1]:.3f} dB | SSIM={SSIM_list[-1]:.4f}"
            print(msg)

    elapsed = time.time() - tic

    if patience is not None and best_u is not None and ref is not None:
        return best_u, F_list, SNR_list, SSIM_list, elapsed, SNR_planes_history
    return u, F_list, SNR_list, SSIM_list, elapsed, SNR_planes_history

def save_lbfgsb_checkpoint(ckpt_dir, x_np, shape, it, F_list, SNR_list, SSIM_list,
                           elapsed, best_x_np=None, best_iter=-1,
                           best_snr=float('nan')):
    """Atomically write an L-BFGS-B resume checkpoint to <ckpt_dir>/state.npz.

    Writes to a temp file then os.replace() so a kill mid-write cannot leave a
    truncated/corrupt checkpoint behind. Also persists the best-SNR iterate so
    early-stopping is correct across chained windows.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    tmp = os.path.join(ckpt_dir, "_state_tmp.npz")   # np.savez keeps the .npz suffix
    final = os.path.join(ckpt_dir, "state.npz")
    np.savez(
        tmp,
        x=np.asarray(x_np, dtype=np.float64).reshape(-1),
        shape=np.asarray(shape, dtype=np.int64),
        iter=np.int64(it),
        F_list=np.asarray(F_list, dtype=np.float64),
        SNR_list=np.asarray(SNR_list, dtype=np.float64),
        SSIM_list=np.asarray(SSIM_list, dtype=np.float64),
        elapsed=np.float64(elapsed),
        best_x=np.asarray([] if best_x_np is None else best_x_np,
                          dtype=np.float64).reshape(-1),
        best_iter=np.int64(best_iter),
        best_snr=np.float64(best_snr),
    )
    os.replace(tmp, final)


def load_lbfgsb_checkpoint(ckpt_dir):
    """Return the saved checkpoint dict, or None if none exists."""
    path = os.path.join(ckpt_dir, "state.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    best_x = d["best_x"] if "best_x" in d else np.array([])
    return {
        "x": d["x"],
        "shape": tuple(int(s) for s in d["shape"]),
        "iter": int(d["iter"]),
        "F_list": d["F_list"].tolist(),
        "SNR_list": d["SNR_list"].tolist(),
        "SSIM_list": d["SSIM_list"].tolist(),
        "elapsed": float(d["elapsed"]),
        "best_x": (best_x if best_x.size else None),
        "best_iter": int(d["best_iter"]) if "best_iter" in d else -1,
        "best_snr": float(d["best_snr"]) if "best_snr" in d else float('nan'),
    }


# L-BFGS-B with Tikhonov regu
def least_squares_LBFGSB(pat, s_meas, M_inv, max_iter=300, ftol=2.22e-9, gtol=2.22e-9,
                         maxcor=10, lam=1e-20, bound=None,
                         ref=None, verbose=True, patience=None,
                         track_planes=False, plane_dim=2,
                         x0=None, ckpt_dir=None, ckpt_every=1,
                         iter_offset=0, elapsed_offset=0.0,
                         F_init=None, SNR_init=None, SSIM_init=None,
                         best_x_init=None, best_snr_init=None, best_iter_init=0,
                         should_stop=None, status=None):
    """
    Non-Negative Least Squares via L-BFGS-B with Tikhonov regularization.
    Solves: min_{x>=0} 0.5*||Ax-s||^2 + 0.5*lam*||x||^2

    L-BFGS-B (Limited-memory BFGS for Bound-Constrained problems) is a
    quasi-Newton method that handles box constraints — here x >= 0.

    Checkpoint / resume (for long SLURM runs split across walltime windows):

    - ``x0``: warm-start iterate (np.ndarray or torch.Tensor); None -> zeros.
    - ``ckpt_dir``: if set, every ``ckpt_every`` iters atomically write
      ``<ckpt_dir>/state.npz`` holding x, the global iteration count and
      the metric histories, so a killed job can resume from there.
    - ``iter_offset``: global iteration number already completed before this call.
    - ``F_init``/``SNR_init``/``SSIM_init``: prior histories to prepend
      (resume continuity).
    - ``should_stop``: ``callable() -> bool`` checked each iter; True -> stop
      cleanly (used by the SIGUSR1/walltime trap to flush before the kill).
    - ``status``: optional dict, filled with ``{'converged', 'message', 'n_iter'}``.

    Returns: (u, F_list, SNR_list, SSIM_list, elapsed, SNR_planes_history)
    """
    resuming = iter_offset > 0 or F_init is not None
    prior_elapsed = elapsed_offset
    device = s_meas.device
    dtype  = torch.float64

    b     = pat.T @ s_meas
    shape = b.shape
    n     = b.numel()

    # --- diagonal preconditioning ---
    if M_inv is not None:
        M_inv_t = M_inv.to(device=device, dtype=dtype).reshape(shape)
    else:
        M_inv_t = None

    # --- history (seed from prior windows when resuming) ---
    F_list, SNR_list, SSIM_list, SNR_planes_history = _init_history(track_planes)
    if F_init is not None:
        F_list.extend(list(F_init))
    if SNR_init is not None:
        SNR_list.extend(list(SNR_init))
    if SSIM_init is not None:
        SSIM_list.extend(list(SSIM_init))

    best_snr, best_x, best_it, snr_no_improve = -float('inf'), None, 0, 0
    iter_count = [iter_offset]
    last_f     = [0.0]

    x_prev = [None]  
    
    def fg(x_np):
        """Returns (f, grad) in x-space."""
        x_t = torch.from_numpy(x_np).to(device=device, dtype=dtype).reshape(shape)

        # residual
        r = pat @ x_t - s_meas

        # objective
        f = 0.5 * torch.sum(r**2) + 0.5 * lam * torch.sum(x_t**2)

        # gradient
        grad = pat.T @ r + lam * x_t

        # apply diagonal preconditioning (left scaling)
        if M_inv_t is not None:
            grad = M_inv_t * grad

        last_f[0] = f.item()

        return last_f[0], grad.reshape(-1).cpu().numpy().astype(np.float64)

    def callback(x_np):
        iter_count[0] += 1
        it = iter_count[0]

        F_list.append(last_f[0])

        x_t = torch.from_numpy(x_np).to(device=device, dtype=dtype).reshape(shape)

        # --- step monitoring (debug stability) ---
        if x_prev[0] is not None:
            step = torch.norm(x_t - x_prev[0]).item()
        else:
            step = 0.0
        x_prev[0] = x_t.clone()

        if ref is not None:
            nonlocal best_snr, best_x, best_it, snr_no_improve

            snr, ssim, snr_planes = compute_metrics(ref, x_t, track_planes, plane_dim)

            _append_metrics(snr, ssim, snr_planes,
                            SNR_list, SSIM_list, SNR_planes_history)

            best_snr, best_x, best_it, snr_no_improve, stop = _check_early_stop(
                snr, best_snr, best_x, x_t,
                best_it, snr_no_improve, patience, it - 1,
                verbose, 'L-BFGS-B')

            if stop:
                raise _EarlyStop()

        if verbose and (it % 10 == 0 or it <= 3):
            msg = f"[L-BFGS-B] Iter {it:03d} | obj={last_f[0]:.4e} | step={step:.2e}"
            if ref is not None:
                msg += f" | SNR={SNR_list[-1]:.3f} dB"
            print(msg, flush=True)

        bx = None if best_x is None else best_x.reshape(-1).cpu().numpy()

        # --- checkpoint (atomic) for resume across SLURM windows ---
        if ckpt_dir is not None and (it % ckpt_every == 0):
            save_lbfgsb_checkpoint(ckpt_dir, x_np, shape, it,
                                   F_list, SNR_list, SSIM_list,
                                   prior_elapsed + (time.time() - tic),
                                   best_x_np=bx, best_iter=best_it, best_snr=best_snr)

        # --- cooperative stop (walltime / SIGUSR1 trap) ---
        if should_stop is not None and should_stop():
            if ckpt_dir is not None:
                save_lbfgsb_checkpoint(ckpt_dir, x_np, shape, it,
                                       F_list, SNR_list, SSIM_list,
                                       prior_elapsed + (time.time() - tic),
                                       best_x_np=bx, best_iter=best_it, best_snr=best_snr)
            if verbose:
                print(f"[L-BFGS-B] Stop requested at iter {it} — checkpoint flushed",
                      flush=True)
            raise _EarlyStop()

    # --- bounds: x >= 0 ---
    bounds = [(0.0, bound)] * n

    # --- initialization (warm-start from x0 when resuming) ---
    if x0 is not None:
        x0_t = (x0 if torch.is_tensor(x0)
                else torch.from_numpy(np.asarray(x0))).to(device=device, dtype=dtype)
        x0_t = x0_t.reshape(shape).contiguous()
    else:
        x0_t = torch.clamp(b, min=0) * 0.0
    x0_np = x0_t.reshape(-1).cpu().numpy().astype(np.float64)

    # seed the histories only on a fresh start; on resume they are already filled
    if not resuming:
        f0 = 0.5 * torch.sum(s_meas ** 2).item()
        F_list.append(f0)
        if ref is not None:
            snr, ssim, snr_planes = compute_metrics(ref, x0_t, track_planes, plane_dim)
            _append_metrics(snr, ssim, snr_planes,
                            SNR_list, SSIM_list, SNR_planes_history)
            best_snr, best_x = snr, x0_t.clone()
    elif ref is not None:
        # resume best-tracking from the persisted global best, not the (later,
        # possibly over-fitted) resumed iterate
        if best_x_init is not None:
            best_x = (best_x_init if torch.is_tensor(best_x_init)
                      else torch.from_numpy(np.asarray(best_x_init))
                      ).to(device=device, dtype=dtype).reshape(shape).contiguous()
            best_snr = best_snr_init if best_snr_init is not None else -float('inf')
            best_it = int(best_iter_init)
        else:
            snr, ssim, _ = compute_metrics(ref, x0_t, track_planes, plane_dim)
            best_snr, best_x = snr, x0_t.clone()

    # --- run optimizer ---
    tic = time.time()
    stopped_early = False

    try:
        result = scipy.optimize.minimize(
            fg, x0_np,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            callback=callback,
            options={
                'maxiter': max_iter,
                'maxfun':  max(15000, max_iter * 100),  # don't cap before maxiter/walltime
                'ftol':    ftol,
                'gtol':    gtol,
                'maxcor':  maxcor,
                'maxls':   20,
                'iprint':  -1,
            }
        )
    except _EarlyStop:
        stopped_early = True
        result = None

    elapsed = prior_elapsed + (time.time() - tic)

    # converged = optimizer reported success (status 0), not a maxiter/walltime stop
    converged = (not stopped_early) and result is not None and result.status == 0
    if status is not None:
        status['converged'] = bool(converged)
        status['n_iter'] = int(iter_count[0])
        status['message'] = (str(result.message) if result is not None
                             else 'stopped (window budget / signal)')
        status['stopped_early'] = bool(stopped_early)
        status['best_x'] = best_x                       # best-SNR iterate (tensor or None)
        status['best_iter'] = int(best_it)
        status['best_snr'] = float(best_snr)
        # nfev = gradient evals (forward+adjoint pairs) = the true cost driver;
        # nfev/n_iter is the per-iteration line-search count (grows when ill-conditioned)
        status['nfev'] = int(result.nfev) if result is not None else -1

    if verbose:
        if stopped_early:
            print(f"[L-BFGS-B] Early stopped at iter {iter_count[0]} | time={elapsed:.2f}s",
                  flush=True)
        else:
            print(f"[L-BFGS-B] Done — {result.message} | "
                  f"iters={iter_count[0]} | obj={result.fun:.4e} | time={elapsed:.2f}s",
                  flush=True)

    # final checkpoint so a resume picks up exactly where this window ended.
    # resume x = the LATEST iterate (to continue the trajectory); the best-SNR
    # iterate is persisted separately for early-stopping.
    if ckpt_dir is not None:
        if result is not None:
            resume_x = torch.tensor(result.x, dtype=dtype, device=device).reshape(shape)
        elif x_prev[0] is not None:
            resume_x = x_prev[0]
        else:
            resume_x = x0_t
        bx = None if best_x is None else best_x.reshape(-1).cpu().numpy()
        save_lbfgsb_checkpoint(ckpt_dir, resume_x.reshape(-1).cpu().numpy(), shape,
                               iter_count[0], F_list, SNR_list, SSIM_list, elapsed,
                               best_x_np=bx, best_iter=best_it, best_snr=best_snr)

    if stopped_early:
        # window cut short (patience / walltime trap): return the best iterate if
        # we have one, else the last seen iterate (x_prev), else the warm start.
        u = (best_x if best_x is not None
             else x_prev[0] if x_prev[0] is not None
             else x0_t)
        return u, F_list, SNR_list, SSIM_list, elapsed, SNR_planes_history

    # --- final solution ---
    u = torch.tensor(result.x, dtype=dtype, device=device).reshape(shape)

    if patience is not None and best_x is not None and ref is not None:
        return best_x, F_list, SNR_list, SSIM_list, elapsed, SNR_planes_history

    return u, F_list, SNR_list, SSIM_list, elapsed, SNR_planes_history


# Projected Gradient Decent with Tikhonov regu

def least_squares_PGD(pat, s_meas, M_inv, max_iter=500, tol=1e-6, ref=None,
                      verbose=True, patience=None, track_planes=False, plane_dim=2,
                      step_size=None, lam=0.0):
    """
    Non-Negative Least Squares via Projected Gradient Descent with Tikhonov regularization.
    Solves:  min_{x>=0}  0.5*||Ax-s||^2  +  0.5*lam*||x||^2

    `tol`: relative gradient ||grad||/||b||. Scale-invariant -- works in
    physical and rescaled signal modes. 
    `step_size=None` auto-picks the Lipschitz step 1/L; for PAT in raw units this 
    gives lr ~ 1e15 which cancels the tiny gradient ~ 1e-15 to a sensible step ~ 1.

    For PAT with ||A^T A|| ~ 1e-15, a Morozov-like Tikhonov default is
        lam ~ ||A^T A|| * (sigma_meas)^2

    Returns:
        u, F_list, SNR_list, SSIM_list, elapsed, SNR_planes_history
    """
    device = s_meas.device
    dtype  = torch.float64

    b = pat.T @ s_meas

    M_inv_t = (M_inv.to(device=device, dtype=dtype).reshape(b.shape)
               if M_inv is not None else None)

    if step_size is None:
        if verbose:
            print("[PGD] Estimating step size via power iteration...")
        if M_inv_t is None:
            L = estimate_lipschitz(pat, b.shape, device, dtype, lam=lam)
        else:
            L = _estimate_lipschitz_preconditioned(pat, M_inv_t, b.shape,
                                                   device, dtype, lam=lam)
        lr = 1.0 / (L + 1e-30)
        if verbose:
            tag = "preconditioned" if M_inv_t is not None else "plain"
            print(f"[PGD] L_{tag} = {L:.4e}  ->  lr = {lr:.4e}")
    else:
        lr = float(step_size)

    u  = torch.zeros_like(b)
    Au = torch.zeros_like(s_meas)
    norm_b = b.norm()

    F_list, SNR_list, SSIM_list, SNR_planes_history = _init_history(track_planes)
    best_snr, best_u, best_it, snr_no_improve = -float('inf'), None, 0, 0

    # Initial cost / metrics at u₀
    r0 = Au - s_meas
    F_list.append(0.5 * r0.square().sum().item() + 0.5 * lam * u.square().sum().item())
    if ref is not None:
        snr, ssim, snr_planes = compute_metrics(ref, u, track_planes, plane_dim)
        _append_metrics(snr, ssim, snr_planes, SNR_list, SSIM_list, SNR_planes_history)
        best_snr, best_u = snr, u.clone()

    tic = time.time()

    for i in range(max_iter):
        grad      = pat.T @ Au - b + lam * u    # 1 adjoint matvec (uses cached Au)
        direction = M_inv_t * grad if M_inv_t is not None else grad
        u         = torch.clamp(u - lr * direction, min=0.0)

        Au   = pat @ u                          # 1 forward matvec (reused next iter)
        r    = Au - s_meas
        cost = 0.5 * r.square().sum().item() + 0.5 * lam * u.square().sum().item()
        F_list.append(cost)

        rel_grad = grad.norm() / (norm_b + 1e-30)
        if rel_grad < tol:
            if verbose:
                print(f"[PGD] Converged at iter {i+1}: rel_grad={rel_grad:.3e}")
            break

        if ref is not None:
            snr, ssim, snr_planes = compute_metrics(ref, u, track_planes, plane_dim)
            _append_metrics(snr, ssim, snr_planes, SNR_list, SSIM_list, SNR_planes_history)
            best_snr, best_u, best_it, snr_no_improve, stop = _check_early_stop(
                snr, best_snr, best_u, u, best_it, snr_no_improve, patience, i, verbose, 'PGD')
            if stop:
                break

        if verbose and (i % 10 == 0 or i == max_iter - 1):
            msg = f"[PGD] Iter {i+1:03d}/{max_iter} | cost={cost:.4e} | rel_grad={rel_grad:.3e}"
            if ref is not None:
                msg += f" | SNR={SNR_list[-1]:.3f} dB | SSIM={SSIM_list[-1]:.4f}"
            print(msg)

    elapsed = time.time() - tic

    if patience is not None and best_u is not None and ref is not None:
        return best_u, F_list, SNR_list, SSIM_list, elapsed, SNR_planes_history
    return u, F_list, SNR_list, SSIM_list, elapsed, SNR_planes_history

# Chambolle-Pock with TV regu
def _grad_3d(u):
    """Forward-difference gradient, shape (Nx,Ny,Nz) -> (Nx,Ny,Nz,3)."""
    g = torch.zeros((*u.shape, 3), device=u.device, dtype=u.dtype)
    g[:-1, :, :, 0] = u[1:,  :,  :] - u[:-1, :,   :]
    g[:, :-1, :, 1] = u[:,  1:,  :] - u[:,  :-1,  :]
    g[:, :, :-1, 2] = u[:,   :, 1:] - u[:,   :, :-1]
    return g


def _div_3d(g):
    """∇.T adjoint of `_grad_3d`. shape (Nx,Ny,Nz,3) -> (Nx,Ny,Nz)."""
    d = torch.zeros(g.shape[:-1], device=g.device, dtype=g.dtype)
    # x
    d[1:,   :, :] += g[:-1, :, :, 0]
    d[:-1,  :, :] -= g[:-1, :, :, 0]
    # y
    d[:, 1:,  :] += g[:, :-1, :, 1]
    d[:, :-1, :] -= g[:, :-1, :, 1]
    # z
    d[:, :, 1:]  += g[:, :, :-1, 2]
    d[:, :, :-1] -= g[:, :, :-1, 2]
    return d

def least_squares_CP_TV(pat, s_meas, lam, beta=None, max_iter=1000,
                        sigma=None, tau=None, theta=1.0,
                        ref=None, verbose=True, patience=None,
                        track_planes=False, plane_dim=2, tol=1e-6):
    """
    Non-Negative Least Squares + isotropic 3D TV via the Chambolle-Pock
    primal-dual algorithm using the augmented reformulation with an
    explicit scaling `beta` on the TV block.

    LS problem:
        min_{u >= 0}  0.5*||A u - s||^2  +  lam*||grad(u)||_{2,1}

    `beta` scales the TV block of K = [A; beta*grad]. Every beta > 0 gives the
    same minimiser; it only changes the step sizes. None -> ||A|| / ||grad||
    = sqrt(||A^T A|| / 12), which balances the two blocks.

    Returns:
        u, F_list, SNR_list, SSIM_list, elapsed, SNR_planes_history
    """
    if lam is None:
        raise ValueError("least_squares_CP_TV: `lam` is required (no sensible default).")

    device = s_meas.device
    dtype  = torch.float64

    # primal init: u = 0
    b     = pat.T @ s_meas
    shape = b.shape
    u     = torch.zeros_like(b)
    ub    = u.clone()

    # dual init: y1 (data block), y2 (TV block, 3-vector field)
    y1 = torch.zeros_like(s_meas)
    y2 = torch.zeros((*shape, 3), device=device, dtype=dtype)

    # ---- operator norms and beta default ---------------------------------
    if verbose:
        print("[CP-TV] Estimating ||A||^2 via power iteration...")
    L_A    = estimate_lipschitz(pat, shape, device, dtype, lam=0.0)
    norm_A = L_A ** 0.5
    # pat = pat / norm_A
    if beta is None:
        beta = (L_A / 12.0) ** 0.5

    # ||grad||^2 <= 4 per axis for forward differences, so 12 in 3D
    K2     = L_A + 12 * beta ** 2
    norm_K = K2 ** 0.5

    if sigma is None: sigma = 0.99 / norm_K
    if tau   is None: tau   = 0.99 / norm_K

    if verbose:
        print(f"[CP-TV] ||A||^2 = {L_A:.3e} | beta = {beta:.3e} | ||K||^2 = {K2:.3e} "
              f"| sigma = {sigma:.3e} | tau = {tau:.3e} | theta = {theta} | lam = {lam:.3e}")

    F_list, SNR_list, SSIM_list, SNR_planes_history = _init_history(track_planes)
    best_snr, best_u, best_it, snr_no_improve = -float('inf'), None, 0, 0

    # initial cost at u=0:  0.5*||s||^2 (TV term is 0)
    F_list.append(0.5 * (s_meas ** 2).sum().item())
    if ref is not None:
        snr, ssim, snr_planes = compute_metrics(ref, u, track_planes, plane_dim)
        _append_metrics(snr, ssim, snr_planes, SNR_list, SSIM_list, SNR_planes_history)
        best_snr, best_u = snr, u.clone()

    # cache A@u and A@ub: linearity saves one forward matvec per iter
    Au_old = torch.zeros_like(s_meas)   # A @ u    (u  = 0 initially)
    Au_b   = torch.zeros_like(s_meas)   # A @ ub   (ub = 0 initially)

    inv_lam = (1.0 / lam) if lam > 0 else 0.0   # for the TV ball radius

    tic = time.time()

    for k in range(max_iter):
        # ---- dual data:  y1 <- prox_{sigma f1*}( y1 + sigma * A ub )
        # using cached Au_b  (Au_b == A @ ub by linearity of the extrapolation)
        y1 = (y1 + sigma * (Au_b - s_meas)) / (1.0 + sigma)

        # ---- dual TV:  y2 <- proj_{B(0, lam/beta)}( y2 + sigma * beta * grad(ub) )
        if lam > 0 and beta > 0:
            y2     = y2 + (sigma * beta) * _grad_3d(ub)
            norm_y = torch.sqrt((y2 ** 2).sum(dim=-1, keepdim=True))
            y2     = y2 / torch.clamp((beta * inv_lam) * norm_y, min=1.0)
        # else: y2 stays zero (no TV)

        # ---- primal:  u <- clamp_{>=0}( u - tau*(A^T y1 + beta * grad^T y2) )
        # In this codebase _div_3d == grad^T (positive-sign adjoint convention).
        u_new  = torch.clamp(u - tau * (pat.T @ y1 + beta * _div_3d(y2)), min=0.0)
        Au_new = pat @ u_new                          # 1 forward matvec

        # extrapolation; A @ ub by linearity (no extra matvec)
        ub   = u_new + theta * (u_new - u)
        Au_b = Au_new + theta * (Au_new - Au_old)

        # primal objective in physical units
        r    = Au_new - s_meas
        gu   = _grad_3d(u_new)
        tv   = torch.sqrt((gu ** 2).sum(dim=-1)).sum().item()
        cost = 0.5 * r.square().sum().item() + lam * tv
        F_list.append(cost)

        Au_old = Au_new
        u      = u_new

        # stop on relative primal-objective change
        if tol > 0 and len(F_list) > 1:
            f_prev, f_curr = F_list[-2], F_list[-1]
            denom = max(abs(f_prev), abs(f_curr), 1e-30)
            if abs(f_prev - f_curr) / denom < tol:
                if verbose:
                    print(f"[CP-TV] Converged at iter {k+1}: "
                          f"|dF|/F = {abs(f_prev - f_curr)/denom:.3e}")
                break

        if ref is not None:
            snr, ssim, snr_planes = compute_metrics(ref, u, track_planes, plane_dim)
            _append_metrics(snr, ssim, snr_planes, SNR_list, SSIM_list, SNR_planes_history)
            best_snr, best_u, best_it, snr_no_improve, stop = _check_early_stop(
                snr, best_snr, best_u, u, best_it, snr_no_improve, patience, k, verbose, 'CP-TV')
            if stop:
                break

        if verbose and (k % 10 == 0 or k == max_iter - 1):
            msg = f"[CP-TV] Iter {k+1:03d}/{max_iter} | cost={cost:.4e}"
            if ref is not None:
                msg += f" | SNR={SNR_list[-1]:.3f} dB | SSIM={SSIM_list[-1]:.4f}"
            print(msg)

    elapsed = time.time() - tic

    if patience is not None and best_u is not None and ref is not None:
        return best_u, F_list, SNR_list, SSIM_list, elapsed, SNR_planes_history
    return u, F_list, SNR_list, SSIM_list, elapsed, SNR_planes_history
