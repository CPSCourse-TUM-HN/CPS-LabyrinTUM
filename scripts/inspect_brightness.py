#!/usr/bin/env python3
"""Inspect local brightness in a video/image frame.

Use this before changing vision.min_specular. Hover over the ball and the
worst glare/hole/text spot; the overlay shows the peak grayscale value in a
small patch. Left click prints the current measurement.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


WINDOW = "brightness inspector"


def read_frame(source: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(source))
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            return frame

    frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not read image/video frame from {source}")
    return frame


def patch_stats(gray: np.ndarray, x: int, y: int, radius: int) -> tuple[int, float]:
    h, w = gray.shape
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return 0, 0.0
    return int(patch.max()), float(patch.mean())


def draw_overlay(
    frame: np.ndarray,
    frame_index: int,
    mouse: tuple[int, int],
    radius: int,
    source: Path,
) -> np.ndarray:
    out = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x, y = mouse
    peak, mean = patch_stats(gray, x, y, radius)
    cv2.circle(out, (x, y), radius, (0, 255, 255), 1, cv2.LINE_AA)
    lines = [
        f"{source}  frame={frame_index}",
        f"cursor=({x},{y})  patch_radius={radius}px  peak={peak}  mean={mean:.1f}",
        "left click prints | n/p frame | +/- radius | q/esc quit",
    ]
    for idx, line in enumerate(lines):
        yy = 28 + idx * 26
        cv2.putText(out, line, (16, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(out, line, (16, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="video or image to inspect")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--patch-radius", type=int, default=10)
    args = parser.parse_args()

    source = Path(args.source)
    frame_index = max(0, args.frame_index)
    radius = max(1, args.patch_radius)
    mouse = [0, 0]
    frame = read_frame(source, frame_index)

    def on_mouse(event: int, x: int, y: int, *_rest: object) -> None:
        mouse[0], mouse[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            peak, mean = patch_stats(gray, x, y, radius)
            print(
                f"frame={frame_index}, x={x}, y={y}, "
                f"patch_radius={radius}, peak={peak}, mean={mean:.1f}"
            )

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        cv2.imshow(WINDOW, draw_overlay(frame, frame_index, tuple(mouse), radius, source))
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
        if key in (ord("n"), ord(".")):
            frame_index += 1
            frame = read_frame(source, frame_index)
        elif key in (ord("p"), ord(",")):
            frame_index = max(0, frame_index - 1)
            frame = read_frame(source, frame_index)
        elif key in (ord("+"), ord("=")):
            radius += 1
        elif key in (ord("-"), ord("_")):
            radius = max(1, radius - 1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
