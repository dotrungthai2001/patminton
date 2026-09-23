"""CPU tests for metrics, solvers and solver utilities (no GPU required)."""

import math

import numpy as np
import pytest
import scipy.optimize
import torch

from patminton.algorithms import (
    PSNR,
    SNR,
    SNR_masked,
    SNR_per_plane,
    NormalizedPAT,
    _div_3d,
    _grad_3d,
    compute_ssim_3d,
    estimate_lipschitz,
    least_squares_CG,
    least_squares_CP_TV,
    least_squares_LBFGSB,
    least_squares_PGD,
    normalize_operator,
)

torch.manual_seed(0)


class MatrixOperator:
    """Dense-matrix stand-in for PAT, mimicking its transpose protocol:
    `.T` sets a flag consumed by the next `@` product."""

    def __init__(self, M, vol_shape):
        self.M = M
        self.vol_shape = vol_shape
        self.transpose = False
        # attributes mirrored by NormalizedPAT
        self.Nx, self.Ny, self.Nz = vol_shape
        self.nTrans, self.nT = 1, M.shape[0]

    def __matmul__(self, x):
        if self.transpose:
            self.transpose = False
            return (self.M.T @ x.reshape(-1)).reshape(self.vol_shape)
        return self.M @ x.reshape(-1)

    @property
    def T(self):
        self.transpose = True
        return self


@pytest.fixture(scope="module")
def matrix_op():
    vol_shape = (4, 4, 4)
    n = 64
    m = 100
    # controlled spectrum: top singular value 3, well separated from the rest
    gen = torch.Generator().manual_seed(1)
    U, _ = torch.linalg.qr(torch.randn(m, n, dtype=torch.float64, generator=gen))
    V, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64, generator=gen))
    svals = torch.cat([torch.tensor([3.0]), torch.linspace(1.0, 0.1, n - 1)])
    M = U @ torch.diag(svals.to(torch.float64)) @ V.T
    return MatrixOperator(M, vol_shape)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def test_snr_known_value():
    xref = torch.ones(8, 8, 8, dtype=torch.float64)
    xest = xref + 0.1  # error energy = 1% of signal energy
    assert math.isclose(SNR(xref, xest).item(), 20.0, rel_tol=1e-12)


def test_snr_per_plane_shape():
    xref = torch.rand(4, 5, 6, dtype=torch.float64)
    xest = xref + 0.01 * torch.rand(4, 5, 6, dtype=torch.float64)
    for dim, n in [(0, 4), (1, 5), (2, 6)]:
        assert SNR_per_plane(xref, xest, dim=dim).shape == (n,)


def test_snr_masked():
    xref = torch.rand(6, 6, 6, dtype=torch.float64)
    xest = xref.clone()
    mask = torch.zeros(6, 6, 6, dtype=torch.bool)
    # empty mask -> NaN, not +-inf
    assert math.isnan(SNR_masked(xref, xest, mask))
    # corrupt only the unmasked half: masked SNR unaffected
    mask[:3] = True
    xest[3:] += 1.0
    xest[:3] += 0.1
    expected = 10 * math.log10(
        (xref[:3] ** 2).sum().item() / (0.01 * mask.sum().item())
    )
    assert math.isclose(SNR_masked(xref, xest, mask), expected, rel_tol=1e-9)


def test_psnr_known_value():
    xref = torch.linspace(0, 2, 512, dtype=torch.float64).reshape(8, 8, 8)
    xest = xref + 0.02  # MSE = 4e-4, reference range = 2
    assert math.isclose(PSNR(xref, xest).item(), 40.0, rel_tol=1e-12)
    # an explicit data_range overrides the reference range
    assert math.isclose(PSNR(xref, xest, data_range=0.2).item(), 20.0, rel_tol=1e-12)


def test_psnr_is_snr_plus_constant():
    xref = torch.rand(6, 6, 6, dtype=torch.float64)
    offset = 10 * math.log10(
        (xref.max() - xref.min()).item() ** 2 / (xref ** 2).mean().item()
    )
    for scale in (0.01, 0.1, 1.0):
        xest = xref + scale * torch.rand_like(xref)
        assert math.isclose(
            PSNR(xref, xest).item(), SNR(xref, xest).item() + offset, rel_tol=1e-12
        )


def test_ssim_identity_and_degradation():
    x = torch.rand(16, 16, 16, dtype=torch.float64)
    assert math.isclose(compute_ssim_3d(x, x).item(), 1.0, abs_tol=1e-10)
    noisy = x + 0.5 * torch.rand_like(x)
    assert compute_ssim_3d(x, noisy).item() < 0.99


# --------------------------------------------------------------------------
# TV building blocks
# --------------------------------------------------------------------------

def test_grad_div_adjointness():
    # _div_3d is the transpose of _grad_3d (positive-sign convention):
    # <grad u, g> = <u, div g>
    u = torch.rand(5, 6, 7, dtype=torch.float64)
    g = torch.rand(5, 6, 7, 3, dtype=torch.float64)
    lhs = torch.sum(_grad_3d(u) * g)
    rhs = torch.sum(u * _div_3d(g))
    assert math.isclose(lhs.item(), rhs.item(), rel_tol=1e-12)


# --------------------------------------------------------------------------
# Operator utilities (with a dense-matrix mock of PAT)
# --------------------------------------------------------------------------

def test_matrix_op_adjoint(matrix_op):
    x = torch.rand(*matrix_op.vol_shape, dtype=torch.float64)
    y = torch.rand(matrix_op.M.shape[0], dtype=torch.float64)
    lhs = torch.sum((matrix_op @ x) * y)
    rhs = torch.sum(x * (matrix_op.T @ y))
    assert math.isclose(lhs.item(), rhs.item(), rel_tol=1e-12)


def test_estimate_lipschitz(matrix_op):
    # ||A^T A|| = sigma_max^2 = 9 for the controlled spectrum
    L = estimate_lipschitz(matrix_op, matrix_op.vol_shape, "cpu", torch.float64)
    assert math.isclose(L, 9.0, rel_tol=1e-6)
    # Tikhonov shift: ||A^T A + lam I|| = 9 + lam
    L_reg = estimate_lipschitz(
        matrix_op, matrix_op.vol_shape, "cpu", torch.float64, lam=0.5
    )
    assert math.isclose(L_reg, 9.5, rel_tol=1e-6)


def test_estimate_lipschitz_reproducible(matrix_op):
    args = (matrix_op, matrix_op.vol_shape, "cpu", torch.float64)
    assert estimate_lipschitz(*args, seed=3) == estimate_lipschitz(*args, seed=3)


def test_normalize_operator(matrix_op):
    s = torch.rand(matrix_op.M.shape[0], dtype=torch.float64)
    A_n, s_n, norm_A = normalize_operator(matrix_op, s, matrix_op.vol_shape)
    assert isinstance(A_n, NormalizedPAT)
    assert math.isclose(norm_A, 3.0, rel_tol=1e-6)
    assert torch.allclose(s_n, s / norm_A)
    # the wrapped operator has unit spectral norm
    L_n = estimate_lipschitz(A_n, matrix_op.vol_shape, "cpu", torch.float64)
    assert math.isclose(L_n, 1.0, rel_tol=1e-6)
    # and stays adjoint-consistent
    x = torch.rand(*matrix_op.vol_shape, dtype=torch.float64)
    y = torch.rand(matrix_op.M.shape[0], dtype=torch.float64)
    lhs = torch.sum((A_n @ x) * y)
    rhs = torch.sum(x * (A_n.T @ y))
    assert math.isclose(lhs.item(), rhs.item(), rel_tol=1e-12)


# --------------------------------------------------------------------------
# Solvers, on dense-matrix problems whose minimiser is known exactly
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def signal(matrix_op):
    """Data whose unconstrained solution has negative entries, so the
    non-negativity constraint is active in the constrained problems."""
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(matrix_op.M.shape[1], dtype=torch.float64, generator=gen)
    return matrix_op.M @ x


@pytest.fixture(scope="module")
def tv_problem(matrix_op):
    """Data from a non-negative volume, with the best constant fit c* and a
    TV weight lam_ub above which c* * 1 is the exact TV minimiser.

    c* 1 is optimal iff g = M^T (s - M c* 1) equals grad^T p for a field p
    with max_i |p_i| <= lam. The minimum-norm p gives such a lam_ub.
    """
    M, shape = matrix_op.M, matrix_op.vol_shape
    n = M.shape[1]
    gen = torch.Generator().manual_seed(3)
    s = M @ torch.rand(n, dtype=torch.float64, generator=gen)
    M1 = M @ torch.ones(n, dtype=torch.float64)
    c_star = (M1 @ s / (M1 @ M1)).item()
    g = M.T @ (s - c_star * M1)
    D = torch.stack([_grad_3d(e.reshape(shape)).reshape(-1)
                     for e in torch.eye(n, dtype=torch.float64)], dim=1)
    p = torch.linalg.pinv(D.T) @ g
    assert c_star > 0 and torch.allclose(D.T @ p, g)
    lam_ub = p.reshape(*shape, 3).norm(dim=-1).max().item()
    return s, c_star, lam_ub


def _rel_err(u, ref):
    return ((u.reshape(-1) - ref.reshape(-1)).norm() / ref.norm()).item()


def _nnls(M, s, lam=0.0):
    """argmin_{x >= 0} ||M x - s||^2 + lam ||x||^2, by active set (exact)."""
    n = M.shape[1]
    A = np.vstack([M.numpy(), np.sqrt(lam) * np.eye(n)])
    b = np.concatenate([s.numpy(), np.zeros(n)])
    return torch.from_numpy(scipy.optimize.nnls(A, b)[0])


def _tv_objective(M, u, s, lam):
    return (0.5 * ((M @ u.reshape(-1) - s) ** 2).sum()
            + lam * _grad_3d(u).norm(dim=-1).sum()).item()


@pytest.mark.parametrize("lam", [0.0, 0.1])
def test_cgls_matches_closed_form(matrix_op, signal, lam):
    M = matrix_op.M
    ref = torch.linalg.solve(
        M.T @ M + lam * torch.eye(M.shape[1], dtype=torch.float64), M.T @ signal
    )
    u, *_ = least_squares_CG(matrix_op, signal, None, max_iter=500, tol=1e-12,
                             lam=lam, verbose=False)
    assert _rel_err(u, ref) < 1e-9


# L-BFGS-B stops on its default ftol (~3e-6 here); PGD runs to max_iter
@pytest.mark.parametrize("solver, rtol", [(least_squares_LBFGSB, 1e-4),
                                          (least_squares_PGD, 1e-9)])
def test_nonneg_solvers_match_nnls(matrix_op, signal, solver, rtol):
    lam = 0.5
    ref = _nnls(matrix_op.M, signal, lam)
    assert (ref == 0).sum() > 10  # the constraint is active
    u, *_ = solver(matrix_op, signal, None, max_iter=500, lam=lam, verbose=False)
    assert u.min() >= 0
    assert _rel_err(u, ref) < rtol


def test_cp_tv_without_tv_matches_nnls(matrix_op, signal):
    # lam = 0 leaves the data term and the constraint: plain NNLS
    ref = _nnls(matrix_op.M, signal)
    u, *_ = least_squares_CP_TV(matrix_op, signal, lam=0.0, max_iter=3000,
                                tol=0, verbose=False)
    assert _rel_err(u, ref) < 1e-9


def test_cp_tv_large_lam_gives_constant(matrix_op, tv_problem):
    s, c_star, lam_ub = tv_problem
    u, *_ = least_squares_CP_TV(matrix_op, s, lam=2 * lam_ub, max_iter=1000,
                                tol=0, verbose=False)
    const = torch.full(matrix_op.vol_shape, c_star, dtype=torch.float64)
    assert _rel_err(u, const) < 1e-10


def test_cp_tv_objective_below_both_limits(matrix_op, tv_problem):
    # between the two limits, the TV minimiser beats both the NNLS solution
    # (optimal as lam -> 0) and the constant (optimal for lam >= lam_ub)
    s, c_star, lam_ub = tv_problem
    lam = lam_ub / 10
    M = matrix_op.M
    u, *_ = least_squares_CP_TV(matrix_op, s, lam=lam, max_iter=5000, tol=0,
                                verbose=False)
    const = torch.full(matrix_op.vol_shape, c_star, dtype=torch.float64)
    f = _tv_objective(M, u, s, lam)
    assert u.min() >= 0
    assert f < _tv_objective(M, _nnls(M, s).reshape(matrix_op.vol_shape), s, lam)
    assert f < _tv_objective(M, const, s, lam)
