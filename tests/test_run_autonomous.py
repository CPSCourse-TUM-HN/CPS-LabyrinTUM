import numpy as np

from cps_maze.logging.run_logger import CsvRunLogger
from cps_maze.planning.path import WaypointPath
from scripts.run_autonomous import (
    RUN_LOG_FIELDS,
    choose_carrot_point,
    slew_limit_command,
)


def test_slew_limits_increasing_drive_to_slow_rate():
    # increasing magnitude uses the slow rate: 0 -> at most slow*dt in one step
    out = slew_limit_command(
        np.array([0.5, 0.0]), np.array([0.0, 0.0]), dt_s=0.016,
        braking=False, slow_per_s=1.5, fast_per_s=12.0)
    assert np.isclose(out[0], 1.5 * 0.016)  # 0.024, not the full 0.5


def test_stalled_kick_tilt_is_not_fast_collapsed():
    # A breakaway kick that drops (reducing magnitude) must HOLD when the ball
    # is stalled (fast_reduce=False): it unwinds at the slow rate, not fast.
    prev = np.array([0.55, 0.0])
    target = np.array([0.08, 0.0])  # kick toggled off
    slow = slew_limit_command(target, prev, dt_s=0.016, braking=False,
                              slow_per_s=1.5, fast_per_s=12.0, fast_reduce=False)
    fast = slew_limit_command(target, prev, dt_s=0.016, braking=False,
                              slow_per_s=1.5, fast_per_s=12.0, fast_reduce=True)
    # stalled: barely drops (holds the tilt); moving: collapses much faster
    assert np.isclose(slow[0], 0.55 - 1.5 * 0.016)
    assert fast[0] < slow[0] - 0.1


def test_braking_always_uses_fast_rate():
    # a command opposing motion (braking) still unwinds fast regardless
    out = slew_limit_command(
        np.array([0.0, 0.0]), np.array([0.5, 0.0]), dt_s=0.016,
        braking=True, slow_per_s=1.5, fast_per_s=12.0, fast_reduce=False)
    assert np.isclose(out[0], 0.5 - 12.0 * 0.016)


class XLimitWallMap:
    def __init__(self, max_clear_x: float):
        self.max_clear_x = max_clear_x

    def line_blocked(self, _a_mm: np.ndarray, b_mm: np.ndarray) -> bool:
        return bool(b_mm[0] > self.max_clear_x)


def test_choose_carrot_point_without_wall_map_uses_full_lookahead():
    path = WaypointPath(np.array([[0.0, 0.0], [50.0, 0.0]]))

    carrot, lookahead = choose_carrot_point(
        path=path,
        position_mm=np.array([0.0, 0.0]),
        progress_mm=0.0,
        lookahead_mm=30.0,
        min_lookahead_mm=10.0,
        wall_map=None,
    )

    assert np.allclose(carrot, [30.0, 0.0])
    assert np.isclose(lookahead, 30.0)


def test_choose_carrot_point_backs_down_to_clear_line_of_sight():
    path = WaypointPath(np.array([[0.0, 0.0], [50.0, 0.0]]))

    carrot, lookahead = choose_carrot_point(
        path=path,
        position_mm=np.array([0.0, 0.0]),
        progress_mm=0.0,
        lookahead_mm=30.0,
        min_lookahead_mm=10.0,
        wall_map=XLimitWallMap(max_clear_x=12.0),
        step_mm=5.0,
    )

    assert np.allclose(carrot, [10.0, 0.0])
    assert np.isclose(lookahead, 10.0)


def test_run_logger_accepts_lost_ball_row_with_hole_fields(tmp_path):
    log_path = tmp_path / "run.csv"
    row = {
        "timestamp_s": 1.25, "found": False,
        "x_mm": "", "y_mm": "", "vx_mm_s": "", "vy_mm_s": "",
        "target_x_mm": "", "target_y_mm": "", "progress_mm": "",
        "carrot_x_mm": "", "carrot_y_mm": "",
        "desired_vx_mm_s": "", "desired_vy_mm_s": "",
        "cross_track_mm": "", "turn_deg": "",
        "wall_speed_scale": "",
        "hole_brake": "",
        "wall_distance_mm": "",
        "hole_hazard_distance_mm": "",
        "hole_speed_cap_mm_s": "",
        "wall_escape_x": "", "wall_escape_y": "",
        "board_cmd_x": "", "board_cmd_y": "",
        "yaw_command": 0.0, "pitch_command": 0.0,
    }

    with CsvRunLogger(log_path, RUN_LOG_FIELDS) as logger:
        logger.write(row)

    text = log_path.read_text(encoding="utf-8")
    assert "hole_hazard_distance_mm" in text
    assert "hole_speed_cap_mm_s" in text
