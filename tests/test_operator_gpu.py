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


# --- steps maps and access to the upsampled signal ------------------------
ELLIPTIC_MODES = ["cylinder_lut", "cylinder_exact", "cylinder_trapezoidal"]


def _local_coords(infos, element):
    """Voxel centers of the test grid in the frame of one element, as in
    gridToCylinderCoords: x along e2, y along e1 x e2, z along e1, origin at
    the cylinder center. Three (N, N, N) arrays."""
    g = -L + (np.arange(N) + 0.5) * (2 * L / N)
    P = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1) - infos[element, 0:3]
    e1, e2 = infos[element, 3:6], infos[element, 6:9]
    return P @ e2, P @ np.cross(e1, e2), P @ e1


def _random_map(infos, rng, n_nodes=5, max_step=12):
    """Random steps in 1..max_step on n_nodes^3 nodes spanning the grid in
    the frame of element 0."""
    coords = tuple(np.linspace(c.min(), c.max(), n_nodes) for c in _local_coords(infos, 0))
    return rng.integers(1, max_step + 1, size=(n_nodes,) * 3), coords


def _impulse(i, j, k):
    p = torch.zeros((N, N, N), dtype=torch.float64, device=DEVICE)
    p[i, j, k] = 1.0
    return p


@pytest.mark.parametrize("mode", ELLIPTIC_MODES)
def test_constant_steps_map_matches_scalar_steps(mode, infos):
    """A map holding the same step at every node gives the scalar-step operator."""
    coords = tuple(np.linspace(-L, L, 3) for _ in range(3))
    torch.manual_seed(3)
    p = torch.rand((N, N, N), dtype=torch.float64, device=DEVICE)
    s_map = make_pat(mode, infos, steps=np.full((3, 3, 3), 4), steps_coords=coords) @ p
    s_ref = make_pat(mode, infos, steps=4, steps_border=4) @ p
    assert torch.allclose(s_map, s_ref, rtol=0, atol=1e-10 * s_ref.abs().max().item())


def test_steps_map_lookup(infos):
    """Each voxel uses the step of the nearest map node in the element frame:
    its signal equals that of the scalar-step operator with this step."""
    one = np.ascontiguousarray(infos[:1])
    steps_map, coords = _random_map(one, np.random.default_rng(0))
    local = _local_coords(one, 0)
    spacing = [c[1] - c[0] for c in coords]
    # voxels whose nearest node is unambiguous along every axis
    clear = np.ones((N, N, N), dtype=bool)
    for c, v, h in zip(coords, local, spacing):
        clear &= np.abs(v[..., None] - c).min(-1) < 0.3 * h
    voxels = np.argwhere(clear)[:: max(1, int(clear.sum()) // 6)][:6]
    pat_map = make_pat("cylinder_lut", one, steps=steps_map, steps_coords=coords)
    steps_seen = set()
    for i, j, k in voxels:
        node = tuple(int(np.argmin(np.abs(c - v[i, j, k]))) for c, v in zip(coords, local))
        step = int(steps_map[node])
        steps_seen.add(step)
        p = _impulse(i, j, k)
        u_map = pat_map.compute_signal_u(p)
        u_ref = make_pat("cylinder_lut", one, steps=step, steps_border=step).compute_signal_u(p)
        assert u_ref.abs().max() > 0
        assert torch.allclose(u_map, u_ref, rtol=0, atol=1e-12 * u_ref.abs().max().item())
    assert len(steps_seen) > 1


@pytest.mark.parametrize("mode", ELLIPTIC_MODES)
def test_adjoint_dot_product_steps_map(mode, infos):
    """<A x, y> == <x, A^T y> with a different step for each node of the map."""
    steps_map, coords = _random_map(infos, np.random.default_rng(1))
    pat = make_pat(mode, infos, steps=steps_map, steps_coords=coords)
    torch.manual_seed(4)
    x = torch.rand((N, N, N), dtype=torch.float64, device=DEVICE)
    y = torch.rand((pat.nTrans, NT), dtype=torch.float64, device=DEVICE)
    lhs = torch.sum((pat @ x) * y).item()
    rhs = torch.sum(x * (pat.T @ y)).item()
    assert lhs != 0.0
    assert abs(lhs - rhs) / abs(lhs) < 1e-10


@pytest.mark.parametrize("mode", ALL_MODES)
def test_signal_u_and_kernel_reproduce_forward(mode, infos, plane_infos):
    """compute_signal_u, then the convolution with get_kernel_fft and the
    downsampling, give pat @ p; d_su_ then holds the unscaled convolved signal."""
    pat = make_pat(mode, plane_infos if mode == "plane" else infos)
    assert pat.nTu == 5 * NT
    torch.manual_seed(5)
    p = torch.rand((N, N, N), dtype=torch.float64, device=DEVICE)
    u = pat.compute_signal_u(p)
    s = torch.fft.irfft(torch.fft.rfft(u, dim=-1) * pat.get_kernel_fft(), n=pat.nTu, dim=-1)
    s = s.reshape(pat.nTrans, NT, -1).mean(-1)
    s_ref = pat @ p
    atol = 1e-10 * s_ref.abs().max().item()
    assert torch.allclose(s, s_ref, rtol=0, atol=atol)
    su = pat.get_signal_u()
    assert torch.allclose(su.reshape(pat.nTrans, NT, -1).mean(-1) / pat.nTu, s_ref, rtol=0, atol=atol)


def test_last_step_is_its_own_block(infos):
    """With a step longer than the SIR, a voxel whose patch is not split
    (|z| >= h and its angle outside the aperture) gets two blocks: the steps
    l_min .. l_max - 1 share one value and the last step l_max has its own."""
    one = np.ascontiguousarray(infos[:1])
    xl, yl, zl = _local_coords(one, 0)
    a = np.abs(np.arctan2(yl, xl))
    theta_max, h = one[0, 10], one[0, 11]
    single = (np.abs(zl) > h) & (a > theta_max + 0.05) & (a < np.pi - theta_max - 0.05)
    i, j, k = np.argwhere(single)[0]
    u = make_pat("cylinder_lut", one, steps=10**6, steps_border=10**6).compute_signal_u(
        _impulse(i, j, k))[0].cpu().numpy()
    support = np.flatnonzero(np.abs(u) > 1e-12 * np.abs(u).max())
    assert support.size >= 3 and np.all(np.diff(support) == 1)
    first, last = u[support[:-1]], u[support[-1]]
    assert np.allclose(first, first[0], rtol=1e-12, atol=0)
    assert not np.isclose(last, first[0], rtol=1e-6, atol=0)


def test_steps_map_rejected_for_plane(plane_infos):
    coords = tuple(np.linspace(-L, L, 2) for _ in range(3))
    steps_map = np.ones((2, 2, 2), dtype=int)
    with pytest.raises(ValueError):
        make_pat("plane", plane_infos, steps=steps_map, steps_coords=coords)
    with pytest.raises(ValueError):
        make_pat("plane", plane_infos).set_steps_map(steps_map, coords)
