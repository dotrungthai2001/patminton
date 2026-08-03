"""CPU tests for metrics and solver utilities (no GPU required)."""

import math

import pytest
import torch

from patminton.algorithms import (
    SNR,
    SNR_masked,
    SNR_per_plane,
    NormalizedPAT,
    _div_3d,
    _grad_3d,
    compute_ssim_3d,
    estimate_lipschitz,
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
