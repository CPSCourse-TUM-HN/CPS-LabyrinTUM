"""Static wall map for control-time wall awareness.

The maze walls never move, so wall knowledge is captured ONCE as a rasterized
obstacle mask in board-mm space (scripts/build_wall_mask.py) and used at
runtime for O(1) lookups:

- line_blocked(a, b): does the straight segment cross a wall? Used to reject
  path associations that point through a wall (adjacent corridors in
  chicanes sit within the association window).
- wall_distance_mm(p): clearance to the nearest wall, from a precomputed
  distance transform. Used to slow the ball down near walls.

Regenerate the mask whenever the homography changes (same rule as the
path/hole CSVs).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class WallMap:
    def __init__(self, mask: np.ndarray, origin_mm: np.ndarray, scale_px_per_mm: float):
        self.mask = mask.astype(bool)  # True = wall/obstacle
        self.origin_mm = np.asarray(origin_mm, dtype=float)
        self.scale = float(scale_px_per_mm)
        free = (~self.mask).astype(np.uint8)
        # distance (px) from every free cell to the nearest wall cell
        self._dist_px = cv2.distanceTransform(free, cv2.DIST_L2, 3)

    def _to_px(self, p_mm: np.ndarray) -> tuple[int, int]:
        q = (np.asarray(p_mm, dtype=float) - self.origin_mm) * self.scale
        return int(round(q[0])), int(round(q[1]))

    def _inside(self, x: int, y: int) -> bool:
        h, w = self.mask.shape
        return 0 <= x < w and 0 <= y < h

    def is_wall(self, p_mm: np.ndarray) -> bool:
        x, y = self._to_px(p_mm)
        if not self._inside(x, y):
            return True  # off the mapped area counts as blocked
        return bool(self.mask[y, x])

    def wall_distance_mm(self, p_mm: np.ndarray) -> float:
        x, y = self._to_px(p_mm)
        if not self._inside(x, y):
            return 0.0
        return float(self._dist_px[y, x]) / self.scale

    def escape_direction_mm(self, p_mm: np.ndarray,
                            sample_mm: float = 2.0) -> np.ndarray:
        """Unit vector pointing toward increasing wall clearance.

        This is used only as a local unstick hint when the ball is stalled
        while close to a wall. It is intentionally geometry-only: it does not
        choose a target, it just says which board direction moves away from
        the nearest obstacle according to the distance transform.
        """
        p = np.asarray(p_mm, dtype=float)
        d = max(float(sample_mm), 1e-6)
        grad = np.array([
            self.wall_distance_mm(p + np.array([d, 0.0]))
            - self.wall_distance_mm(p - np.array([d, 0.0])),
            self.wall_distance_mm(p + np.array([0.0, d]))
            - self.wall_distance_mm(p - np.array([0.0, d])),
        ])
        norm = float(np.linalg.norm(grad))
        if norm > 1e-9:
            return grad / norm

        # Flat/quantized local neighborhood: pick the best of eight nearby
        # directions. This also helps if the estimated ball center is on a
        # wall pixel where the distance transform is exactly zero.
        dirs = []
        for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
            dirs.append(np.array([np.cos(angle), np.sin(angle)]))
        best_dir = np.zeros(2)
        best_dist = self.wall_distance_mm(p)
        for direction in dirs:
            dist = self.wall_distance_mm(p + d * direction)
            if dist > best_dist:
                best_dist = dist
                best_dir = direction
        best_norm = float(np.linalg.norm(best_dir))
        return best_dir / best_norm if best_norm > 1e-9 else np.zeros(2)

    def line_blocked(self, a_mm: np.ndarray, b_mm: np.ndarray,
                     step_mm: float = 2.0) -> bool:
        a = np.asarray(a_mm, dtype=float)
        b = np.asarray(b_mm, dtype=float)
        length = float(np.linalg.norm(b - a))
        steps = max(int(length / step_mm), 1)
        for i in range(steps + 1):
            if self.is_wall(a + (b - a) * (i / steps)):
                return True
        return False

    def speed_scale(self, p_mm: np.ndarray, slow_start_mm: float = 14.0,
                    floor_mm: float = 5.0, min_frac: float = 0.35) -> float:
        """1.0 in open space, ramping down to min_frac when hugging a wall."""
        d = self.wall_distance_mm(p_mm)
        if d >= slow_start_mm:
            return 1.0
        if d <= floor_mm:
            return min_frac
        t = (d - floor_mm) / (slow_start_mm - floor_mm)
        return min_frac + t * (1.0 - min_frac)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(Path(path), mask=self.mask.astype(np.uint8),
                            origin_mm=self.origin_mm, scale=self.scale)

    @classmethod
    def load(cls, path: str | Path) -> "WallMap":
        data = np.load(Path(path))
        return cls(mask=data["mask"], origin_mm=data["origin_mm"],
                   scale_px_per_mm=float(data["scale"]))
