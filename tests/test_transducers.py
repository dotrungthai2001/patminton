"""CPU tests for the transducer geometry (no GPU required)."""

import numpy as np
import pytest

from patminton import discretize_cylindrical_transducers, translation_rotation_system

RADIUS = 25e-3
HEIGHT = 7.5e-3
WIDTH = 0.250e-3
C = 1540.0
FC = 5e6


@pytest.fixture(scope="module")
def infos():
    return translation_rotation_system(
        transducer_radius=RADIUS,
        transducer_height=HEIGHT,
        transducer_width=WIDTH,
        transducer_pitch=0.289e-3,
        transducer_nbr_elements=64,
        transducer_wavelength=C / FC,
        grid_size=2e-3,
    )


def test_infos_shape_and_constants(infos):
    assert infos.ndim == 2 and infos.shape[1] == 12
    assert infos.shape[0] > 0
    # constant geometric parameters
    assert np.allclose(infos[:, 9], RADIUS)
    assert np.allclose(infos[:, 10], np.arctan(HEIGHT / (2 * RADIUS)))
    assert np.allclose(infos[:, 11], WIDTH / 2)


def test_infos_unit_orthogonal_axes(infos):
    e1 = infos[:, 3:6]
    e2 = infos[:, 6:9]
    assert np.allclose(np.linalg.norm(e1, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(e2, axis=1), 1.0)
    assert np.allclose(np.sum(e1 * e2, axis=1), 0.0, atol=1e-12)


def test_non_unit_axis_raises():
    with pytest.raises(ValueError):
        translation_rotation_system(
            transducer_radius=RADIUS,
            transducer_height=HEIGHT,
            transducer_width=WIDTH,
            transducer_pitch=0.289e-3,
            transducer_nbr_elements=64,
            transducer_wavelength=C / FC,
            grid_size=2e-3,
            rotation_axis=np.array([0, 2, 0]),
        )


def test_discretization_geometry():
    theta_max = np.arctan(HEIGHT / (2 * RADIUS))
    half_width = WIDTH / 2
    center = np.array([1e-3, -2e-3, 0.5e-3])
    e1 = np.array([0.0, 1.0, 0.0])
    e2 = np.array([0.0, 0.0, 1.0])

    infos = np.zeros((1, 12))
    infos[0, 0:3] = center
    infos[0, 3:6] = e1
    infos[0, 6:9] = e2
    infos[0, 9] = RADIUS
    infos[0, 10] = theta_max
    infos[0, 11] = half_width

    n_h, n_w = 9, 4
    points, area = discretize_cylindrical_transducers(infos, n_h, n_w)

    assert points.shape == (1, n_h * n_w, 3)
    assert area.shape == (1, n_h * n_w)

    # total area equals arc length x width of the element
    expected_area = (2 * theta_max * RADIUS) * (2 * half_width)
    assert np.isclose(area.sum(), expected_area, rtol=1e-12)

    # every point lies at distance R from the element axis (line through the
    # center of curvature along e1) and within the half-width along e1
    d = points[0] - center
    along = d @ e1
    radial = d - np.outer(along, e1)
    assert np.allclose(np.linalg.norm(radial, axis=1), RADIUS, rtol=1e-12)
    assert np.all(np.abs(along) <= half_width)

    # the arc is centered on the e2 direction
    assert np.all(radial @ e2 > 0)
