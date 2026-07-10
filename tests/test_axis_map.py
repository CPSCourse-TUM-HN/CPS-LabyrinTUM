import numpy as np
import pytest

from cps_maze.control.axis_map import (
    normalized_response_to_axis_map,
    snap_response_to_axis_map,
)


def test_snap_response_to_axis_map_keeps_aligned_axes_identity():
    response = np.array([
        [139.3, -3.4],
        [9.3, 123.1],
    ])

    axis_map = snap_response_to_axis_map(response)

    assert np.allclose(axis_map.matrix, np.eye(2))


def test_normalized_response_to_axis_map_decouples_measured_response():
    response = np.array([
        [139.3, -3.4],
        [9.3, 123.1],
    ])
    scale = float(np.median(np.linalg.norm(response, axis=0)))

    axis_map = normalized_response_to_axis_map(response)

    assert np.allclose(response @ axis_map.matrix, scale * np.eye(2))


def test_normalized_response_to_axis_map_handles_swap_and_sign():
    response = np.array([
        [0.0, -100.0],
        [80.0, 0.0],
    ])
    scale = 90.0

    axis_map = normalized_response_to_axis_map(
        response,
        response_scale_mm_per_unit=scale,
    )

    assert np.allclose(response @ axis_map.matrix, scale * np.eye(2))


def test_normalized_response_to_axis_map_rejects_singular_response():
    response = np.array([
        [100.0, 50.0],
        [0.0, 0.0],
    ])

    with pytest.raises(ValueError, match="singular"):
        normalized_response_to_axis_map(response)
