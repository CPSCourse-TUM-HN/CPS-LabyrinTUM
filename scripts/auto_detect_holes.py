#!/usr/bin/env python3
"""Auto-detect maze holes by thresholding a top-down rectified board view.

Uses the saved homography to warp the camera frame into board-mm space, then
finds dark, circular blobs of hole-like size. Walls (elongated), the printed
guide line (thin), and printed numbers (small) are rejected by area and
circularity filters. Review the result and fix any mistakes by clicking before
saving.

Controls:
  threshold trackbar : adjust until holes are solid dark blobs
  left click         : remove the detection nearest the click
  right click        : add a hole at the click
  SPACE              : grab a fresh frame and re-detect
  s                  : save holes CSV
  q/Esc              : quit
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from cps_maze.calibration.homography import Homography
from cps_maze.camera import CameraCapture
from cps_maze.config import load_config

WINDOW = "auto detect holes"
SCALE_PX_PER_MM = 2.0


def warp_topdown(image: np.ndarray, homography: Homography,
                 width_mm: float, height_mm: float) -> np.ndarray:
    scale = np.array([[SCALE_PX_PER_MM, 0, 0], [0, SCALE_PX_PER_MM, 0], [0, 0, 1]])
    matrix = scale @ homography.image_to_board
    size = (int(width_mm * SCALE_PX_PER_MM), int(height_mm * SCALE_PX_PER_MM))
    return cv2.warpPerspective(image, matrix, size)


def detect_holes(
    topdown_bgr: np.ndarray,
    threshold: int,
    min_radius_mm: float,
    max_radius_mm: float,
) -> tuple[list[tuple[float, float, float]], np.ndarray]:
    """Returns (holes as (x_mm, y_mm, r_mm), binary debug mask)."""
    gray = cv2.cvtColor(topdown_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
    # close small gaps (e.g. glare inside a hole) without merging walls
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    min_area = np.pi * (min_radius_mm * SCALE_PX_PER_MM) ** 2
    max_area = np.pi * (max_radius_mm * SCALE_PX_PER_MM) ** 2

    holes: list[tuple[float, float, float]] = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if not min_area <= area <= max_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.65:  # walls/bars and line fragments are elongated
            continue
        (x_px, y_px), r_px = cv2.minEnclosingCircle(contour)
        holes.append((x_px / SCALE_PX_PER_MM, y_px / SCALE_PX_PER_MM,
                      r_px / SCALE_PX_PER_MM))
    return holes, mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--homography", default="calibration/board_homography.npz")
    parser.add_argument("--output", default="configs/maze_holes.csv")
    parser.add_argument("--min-radius-mm", type=float, default=4.0)
    parser.add_argument("--max-radius-mm", type=float, default=12.0)
    parser.add_argument("--threshold", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    homography = Homography.load(args.homography)
    width_mm = float(config.maze["width_mm"])
    height_mm = float(config.maze["height_mm"])

    holes: list[tuple[float, float, float]] = []
    manual_added: list[tuple[float, float, float]] = []
    default_r_mm = (args.min_radius_mm + args.max_radius_mm) / 2.0

    def on_mouse(event: int, x: int, y: int, *_rest) -> None:
        x_mm, y_mm = x / SCALE_PX_PER_MM, y / SCALE_PX_PER_MM
        if event == cv2.EVENT_LBUTTONDOWN:
            merged = holes + manual_added
            if not merged:
                return
            dists = [np.hypot(h[0] - x_mm, h[1] - y_mm) for h in merged]
            i = int(np.argmin(dists))
            if dists[i] < 20.0:
                target = merged[i]
                (holes if target in holes else manual_added).remove(target)
                print(f"removed hole at ({target[0]:.0f}, {target[1]:.0f}) mm")
        elif event == cv2.EVENT_RBUTTONDOWN:
            manual_added.append((x_mm, y_mm, default_r_mm))
            print(f"added hole at ({x_mm:.0f}, {y_mm:.0f}) mm")

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    cv2.createTrackbar("threshold", WINDOW, args.threshold, 255, lambda _v: None)
    print(__doc__)

    with CameraCapture(config.camera) as camera:
        frame = camera.read().image
        topdown = warp_topdown(frame, homography, width_mm, height_mm)
        last_threshold = -1

        while True:
            threshold = cv2.getTrackbarPos("threshold", WINDOW)
            if threshold != last_threshold:
                holes, mask = detect_holes(topdown, threshold,
                                           args.min_radius_mm, args.max_radius_mm)
                last_threshold = threshold

            view = topdown.copy()
            for x_mm, y_mm, r_mm in holes:
                c = (int(x_mm * SCALE_PX_PER_MM), int(y_mm * SCALE_PX_PER_MM))
                cv2.circle(view, c, int(r_mm * SCALE_PX_PER_MM), (0, 0, 255), 2)
            for x_mm, y_mm, r_mm in manual_added:
                c = (int(x_mm * SCALE_PX_PER_MM), int(y_mm * SCALE_PX_PER_MM))
                cv2.circle(view, c, int(r_mm * SCALE_PX_PER_MM), (255, 0, 255), 2)
            count = len(holes) + len(manual_added)
            cv2.putText(view, f"holes: {count}  (auto {len(holes)} + manual "
                        f"{len(manual_added)})  s=save", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(WINDOW, view)
            cv2.imshow("threshold mask", mask)

            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord(" "):
                frame = camera.read().image
                topdown = warp_topdown(frame, homography, width_mm, height_mm)
                last_threshold = -1  # force re-detection
                print("grabbed fresh frame")
            elif key == ord("s"):
                merged = holes + manual_added
                out = Path(args.output)
                out.parent.mkdir(parents=True, exist_ok=True)
                with out.open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["x_mm", "y_mm", "radius_mm"])
                    for x_mm, y_mm, r_mm in merged:
                        writer.writerow([f"{x_mm:.1f}", f"{y_mm:.1f}", f"{r_mm:.1f}"])
                print(f"saved {len(merged)} holes -> {out}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
