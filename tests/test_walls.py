import numpy as np

from cps_maze.planning.walls import WallMap


def test_escape_direction_points_toward_more_wall_clearance():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[:, :5] = 1  # vertical wall on the left
    wall_map = WallMap(mask, origin_mm=np.array([0.0, 0.0]), scale_px_per_mm=1.0)

    direction = wall_map.escape_direction_mm(np.array([6.0, 10.0]))

    assert direction[0] > 0.8
    assert abs(direction[1]) < 0.3


def test_escape_direction_returns_zero_in_flat_open_space():
    mask = np.zeros((20, 20), dtype=np.uint8)
    wall_map = WallMap(mask, origin_mm=np.array([0.0, 0.0]), scale_px_per_mm=1.0)

    direction = wall_map.escape_direction_mm(np.array([10.0, 10.0]))

    assert np.allclose(direction, np.zeros(2))
