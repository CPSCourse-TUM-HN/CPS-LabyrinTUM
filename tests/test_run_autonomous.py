import numpy as np

from cps_maze.logging.run_logger import CsvRunLogger
from cps_maze.planning.path import WaypointPath
from scripts.run_autonomous import RUN_LOG_FIELDS, choose_carrot_point


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
