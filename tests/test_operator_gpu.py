"""GPU tests for the PAT forward/adjoint operator.

These require a CUDA GPU and the compiled ``libpat_gpu.so``; they are
skipped automatically otherwise. Run them with ``pytest -m gpu``.

The geometry is deliberately small (32^3 grid, 8 transducers, 256 time
samples) so the whole file runs in well under a minute on any GPU.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

import patminton


def _lib_available():
    env = os.environ.get("PATMINTON_LIB")
    if env:
        return Path(env).exists()
    return (Path(patminton.__file__).resolve().parent / "libpat_gpu.so").exists()


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not (torch.cuda.is_available() and _lib_available()),
        reason="requires a CUDA GPU and the compiled libpat_gpu.so",
    ),
]

# --- small test geometry ---------------------------------------------------
N = 32                      # grid voxels per axis
L = 2e-3                    # grid half-extent (m)
NT = 256
DT = 25e-9
TSTART = 17.5e-6            # window 17.5-23.9 us covers every arrival (18.4-23.2 us)
C = 1540.0
RADIUS = 25e-3
HEIGHT = 7.5e-3
WIDTH = 0.250e-3
DEVICE = "cuda:0"


def _probe(element):
    infos = patminton.translation_rotation_system(
        transducer_radius=RADIUS,
        transducer_height=HEIGHT,
        transducer_width=WIDTH,
        transducer_pitch=0.289e-3,
        transducer_nbr_elements=64,
        transducer_wavelength=C / 5e6,
        grid_size=2 * L,
        element=element,
    )
    return np.ascontiguousarray(infos[:8])


@pytest.fixture(scope="module")
def infos():
    return _probe("cylinder")


@pytest.fixture(scope="module")
def plane_infos():
    return _probe("plane")


def make_pat(mode, infos, points=None, **kwargs):
    """`points`: (locPoints, area) for mode='points'; default: the
    cylindrical elements of `infos` sampled on 15 x 3 points."""
    common = dict(nT=NT, tStart=TSTART, dt=DT, c=C, upsample=5,
                  laser_pulse_variance=5e-9)
    common.update(kwargs)
    if mode == "points":
        if points is None:
            points = patminton.discretize_cylindrical_transducers(infos, 15, 3)
        return patminton.PAT(N, N, N, L, L, L, mode="points",
                         locPoints=points[0], area=points[1], **common)
    return patminton.PAT(N, N, N, L, L, L, mode=mode,
                     infos_transducers=infos, **common)


ALL_MODES = ["cylinder_lut", "cylinder_exact", "cylinder_trapezoidal",
             "plane", "points"]


@pytest.mark.parametrize("mode", ALL_MODES)
def test_adjoint_dot_product(mode, infos, plane_infos):
    """<A x, y> == <x, A^T y> to double precision, for every transducer model."""
    pat = make_pat(mode, plane_infos if mode == "plane" else infos)
    torch.manual_seed(0)
    x = torch.rand((N, N, N), dtype=torch.float64, device=DEVICE)
    y = torch.rand((pat.nTrans, NT), dtype=torch.float64, device=DEVICE)
    lhs = torch.sum((pat @ x) * y).item()
    rhs = torch.sum(x * (pat.T @ y)).item()
    assert lhs != 0.0
    assert abs(lhs - rhs) / abs(lhs) < 1e-10


def test_shapes_and_transpose_flag(infos):
    pat = make_pat("cylinder_lut", infos)
    p = torch.rand((N, N, N), dtype=torch.float64, device=DEVICE)
    s = pat @ p
    assert s.shape == (pat.nTrans, NT)
    p_bp = pat.T @ s
    assert p_bp.shape == (N, N, N)
    # the transpose flag is consumed: the next product is a forward again
    assert pat.transpose is False
    assert (pat @ p).shape == (pat.nTrans, NT)


def test_linearity(infos):
    pat = make_pat("cylinder_lut", infos)
    torch.manual_seed(1)
    x1 = torch.rand((N, N, N), dtype=torch.float64, device=DEVICE)
    x2 = torch.rand((N, N, N), dtype=torch.float64, device=DEVICE)
    s = pat @ (2.0 * x1 - 0.5 * x2)
    s_lin = 2.0 * (pat @ x1) - 0.5 * (pat @ x2)
    assert torch.allclose(s, s_lin, rtol=1e-12, atol=s_lin.abs().max() * 1e-12)


def test_point_source_arrival_time(infos):
    """The signal of a central point source peaks at the acoustic
    time-of-flight between the grid center and the element surface."""
    pat = make_pat("cylinder_lut", infos)
    p = torch.zeros((N, N, N), dtype=torch.float64, device=DEVICE)
    p[N // 2, N // 2, N // 2] = 1.0
    s = (pat @ p).cpu().numpy()
    assert np.abs(s).max() > 0

    points, _ = patminton.discretize_cylindrical_transducers(infos, 31, 5)
    for i in range(s.shape[0]):
        d = np.linalg.norm(points[i], axis=1)  # source at the origin
        t_peak = TSTART + np.argmax(np.abs(s[i])) * DT
        # allow 3 samples of slack for the laser-pulse/kernel widening
        assert d.min() / C - 3 * DT <= t_peak <= d.max() / C + 3 * DT


def test_cylinder_modes_agree(infos):
    """LUT and exact elliptic integrals give nearly identical signals."""
    p = torch.zeros((N, N, N), dtype=torch.float64, device=DEVICE)
    p[N // 2, N // 2, N // 2] = 1.0
    s_lut = make_pat("cylinder_lut", infos) @ p
    s_exact = make_pat("cylinder_exact", infos) @ p
    rel = (s_lut - s_exact).norm() / s_exact.norm()
    assert rel.item() < 1e-2


def test_plane_matches_points(plane_infos):
    """Point quadrature of the same flat faces converges to the plane model.

    The faces are 7.5 mm x 0.25 mm, so swapping their two extents, or a
    wrong scale, gives an O(1) error rather than a quadrature error.
    """
    p = torch.zeros((N, N, N), dtype=torch.float64, device=DEVICE)
    p[N // 2, N // 2, N // 2] = 1.0
    s_plane = make_pat("plane", plane_infos) @ p

    def rel_err(n_h, n_w):
        points = patminton.discretize_planar_transducers(plane_infos, n_h, n_w)
        s_points = make_pat("points", None, points=points) @ p
        return ((s_points - s_plane).norm() / s_plane.norm()).item()

    coarse, fine = rel_err(15, 3), rel_err(151, 7)
    assert fine < coarse
    assert fine < 0.2


@pytest.mark.parametrize("mode", ["cylinder_lut", "cylinder_exact", "plane"])
def test_models_match_point_quadrature(mode, infos, plane_infos):
    """With one SIR evaluation per upsampled sample (steps=1), the analytic
    models match a fine point quadrature of the same element surfaces."""
    p = torch.zeros((N, N, N), dtype=torch.float64, device=DEVICE)
    p[N // 2, N // 2, N // 2] = 1.0
    if mode == "plane":
        elements, discretize = plane_infos, patminton.discretize_planar_transducers
    else:
        elements, discretize = infos, patminton.discretize_cylindrical_transducers
    s = make_pat(mode, elements, steps=1, steps_border=1) @ p
    s_ref = make_pat("points", None, points=discretize(elements, 151, 7)) @ p
    assert ((s - s_ref).norm() / s_ref.norm()).item() < 1e-2
