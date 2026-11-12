"""Checks of the steps-map arguments of PAT (no GPU needed)."""

import numpy as np
import pytest

from patminton.operator import _steps_map_arrays

COORDS = (np.arange(2.0), np.arange(3.0), np.arange(4.0))


def test_steps_map_arrays_converts():
    steps_map, coords = _steps_map_arrays(np.full((2, 3, 4), 2.0), COORDS)
    assert steps_map.dtype == np.int32 and steps_map.flags.c_contiguous
    assert steps_map.shape == (2, 3, 4)
    assert [x.dtype for x in coords] == [np.float64] * 3


@pytest.mark.parametrize(
    "steps_map, coords",
    [
        (np.ones((2, 3)), COORDS[:2]),                       # not 3D
        (np.ones((2, 3, 5)), COORDS),                        # sizes do not match
        (np.ones((2, 3, 4)), (COORDS[0], COORDS[1][::-1], COORDS[2])),  # decreasing
        (np.zeros((2, 3, 4)), COORDS),                       # step < 1
    ],
)
def test_steps_map_arrays_rejects(steps_map, coords):
    with pytest.raises(ValueError):
        _steps_map_arrays(steps_map, coords)
