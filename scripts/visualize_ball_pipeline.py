#!/usr/bin/env python3
"""Offline visualizer for the ball-tracking pipeline.

This is the non-camera counterpart to scripts/debug_tracking.py. It loads a
saved image or video, shows the image-processing stages used by the live
pipeline, and runs the same PipelineBallTracker so candidate rejection,
prediction, template rescue, ROI, and static-confuser behavior can be
inspected without connecting the camera.

Controls:
  click        seed/reseed the tracker at the cursor
  n / right    next frame
  b / left     previous frame (video sources only)
  space        play/pause (video sources only)
  r            reset tracker state and return to the start frame
  s            save the current tiled view
  q / Esc      quit
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cps_maze.config import load_config
from cps_maze.vision.ball_pipeline import (
    PipelineBallTracker,
    highlight_candidates,
    in_roi,
    motion_candidates,
    specular_peak,
)

WINDOW = "offline ball pipeline visualizer"


@dataclass(frozen=True)
class SourceFrame:
    image: np.ndarray
    index: int
    count: int
    is_video: bool


class OfflineSource:
    def __init__(self, path: Path):
        self.path = path
        self.cap: cv2.VideoCapture | None = None
        self.image: np.ndarray | None = None
        self.is_video = self._looks_like_video(path)
        if self.is_video:
            self.cap = cv2.VideoCapture(str(path))
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video source {path}")
            self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.count <= 0:
                self.count = 1
        else:
            self.image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if self.image is None:
                raise RuntimeError(f"Could not read image source {path}")
            self.count = 1

    @staticmethod
    def _looks_like_video(path: Path) -> bool:
        return path.suffix.lower() in {
            ".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm",
        }

    def read(self, index: int) -> SourceFrame:
        index = int(np.clip(index, 0, self.count - 1))
        if not self.is_video:
            assert self.image is not None
            return SourceFrame(self.image.copy(), 0, 1, False)
        assert self.cap is not None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = self.cap.read()
        if not ok or image is None:
            raise RuntimeError(f"Could not read frame {index} from {self.path}")
        return SourceFrame(image, index, self.count, True)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


def resolve_source(config_path: Path, source_arg: str | None) -> Path:
    config = load_config(config_path)
    source = source_arg or config.camera.get(
        "reference_image", "calibration/CURRENT_FIXED_CAMERA_VIEW.png")
    path = Path(source)
    if path.is_absolute():
        return path
    return (config_path.parent.parent / path).resolve()


def to_bgr(gray: np.ndarray) -> np.ndarray:
    if gray.ndim == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray.copy()


def text_block(image: np.ndarray, lines: list[str],
               color: tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    out = image.copy()
    scale = max(image.shape[1] / 420.0, 1.0)
    font_scale = 0.58 * scale
    shadow = max(int(round(3 * scale)), 2)
    thickness = max(int(round(scale)), 1)
    y = int(round(24 * scale))
    step = int(round(24 * scale))
    for line in lines:
        x = int(round(10 * scale))
        cv2.putText(out, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), shadow, cv2.LINE_AA)
        cv2.putText(out, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)
        y += step
    return out


def label_panel(image: np.ndarray, title: str,
                color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    return text_block(image, [title], color)


def draw_roi_and_confusers(image: np.ndarray, tracker: PipelineBallTracker) -> np.ndarray:
    out = image.copy()
    if tracker.roi:
        pts = np.array(tracker.roi, dtype=np.int32)
        cv2.polylines(out, [pts], True, (255, 255, 0), 2)
    for sx, sy, sr in tracker.confusers:
        cv2.circle(out, (int(sx), int(sy)), int(sr), (0, 0, 255), 1)
    return out


def draw_candidates(image: np.ndarray, candidates: list[tuple],
                    color: tuple[int, int, int], label: str | None = None) -> np.ndarray:
    out = image.copy()
    for cand in candidates:
        x, y, r = cand[0], cand[1], cand[2]
        cv2.circle(out, (int(x), int(y)), max(int(r), 3), color, 2)
        if label:
            cv2.putText(out, label, (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return out


def motion_stage(prev_gray: np.ndarray | None, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
    if prev_gray is None:
        blank = np.zeros_like(gray)
        return blank, blank, []
    diff = cv2.GaussianBlur(cv2.absdiff(prev_gray, gray), (5, 5), 0)
    _, mask = cv2.threshold(diff, 14, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return diff, mask, motion_candidates(prev_gray, gray)


def highlight_stage(gray: np.ndarray, min_specular: int) -> tuple[np.ndarray, list]:
    mask = (gray >= min_specular).astype(np.uint8) * 255
    return mask, highlight_candidates(gray, thresh=min_specular)


def filtered_candidates(
    raw_candidates: list[tuple],
    tracker: PipelineBallTracker,
) -> tuple[list[tuple], list[tuple]]:
    kept = []
    rejected = []
    for cand in raw_candidates:
        x, y = cand[0], cand[1]
        rejected_by_roi = not in_roi(x, y, tracker.roi)
        rejected_by_confuser = any(
            np.hypot(x - sx, y - sy) <= sr for (sx, sy, sr) in tracker.confusers)
        if rejected_by_roi or rejected_by_confuser:
            rejected.append(cand)
        else:
            kept.append(cand)
    return kept, rejected


def draw_tracker_overlay(
    image: np.ndarray,
    tracker: PipelineBallTracker,
    detection,
) -> np.ndarray:
    out = draw_roi_and_confusers(image, tracker)
    internal = tracker.tracker
    dbg = dict(getattr(internal, "debug", {}) or {}) if internal is not None else {}
    for cand in dbg.get("candidates", []):
        source = cand[3] if len(cand) > 3 else ""
        color = (0, 165, 255) if source == "motion" else (255, 255, 0)
        out = draw_candidates(out, [cand], color)
    if internal is not None:
        predicted = internal.pos + internal._bounded_velocity()
        search_r = float(dbg.get("search_r", 0.0))
        cv2.circle(out, (int(predicted[0]), int(predicted[1])),
                   max(int(search_r), 1), (255, 0, 255), 1)
        cv2.drawMarker(out, (int(predicted[0]), int(predicted[1])),
                       (255, 0, 255), cv2.MARKER_CROSS, 14, 1)
    if detection.found:
        center = (int(detection.x_px), int(detection.y_px))
        cv2.circle(out, center, int(detection.radius_px or 6), (0, 255, 0), 2)
        cv2.circle(out, center, 2, (0, 255, 0), -1)
    status = dbg.get("status", "unseeded")
    color = {"detected": (0, 255, 0), "predicted": (0, 200, 255),
             "lost": (0, 0, 255), "seed": (255, 255, 255)}.get(
                 status, (200, 200, 200))
    return text_block(out, [
        f"tracker: {status}",
        f"motion {dbg.get('n_motion', '-')}  highlight {dbg.get('n_highlight', '-')}",
        f"reject roi/conf {dbg.get('n_rej_roi_confuser', '-')}  "
        f"jump {dbg.get('n_rej_jump', '-')}  spec {dbg.get('n_rej_specular', '-')}",
        "magenta circle = prediction/search gate",
    ], color)


def make_montage(
    frame: SourceFrame,
    prev_gray: np.ndarray | None,
    tracker: PipelineBallTracker,
    detection,
    min_specular: int,
) -> np.ndarray:
    image = frame.image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    diff, motion_mask, motion = motion_stage(prev_gray, gray)
    highlight_mask, highlights = highlight_stage(gray, min_specular)
    raw = ([(x, y, r, "motion") for x, y, r in motion]
           + [(x, y, r, "highlight") for x, y, r in highlights])
    kept, rejected = filtered_candidates(raw, tracker)

    input_panel = text_block(draw_roi_and_confusers(image, tracker), [
        f"input frame {frame.index + 1}/{frame.count}",
        "cyan = ROI, red = confusers",
    ])
    gray_panel = label_panel(to_bgr(gray), "1 grayscale")
    diff_panel = text_block(to_bgr(diff), [
        "2 motion diff",
        "needs previous frame" if prev_gray is None else "absdiff + blur",
    ])
    motion_panel = draw_candidates(to_bgr(motion_mask), motion, (0, 165, 255), "M")
    motion_panel = text_block(motion_panel, [
        f"3 motion mask/candidates: {len(motion)}",
        "area + circularity filtered",
    ], (0, 165, 255))
    highlight_panel = draw_candidates(to_bgr(highlight_mask), highlights,
                                      (255, 255, 0), "H")
    highlight_panel = text_block(highlight_panel, [
        f"4 specular mask/candidates: {len(highlights)}",
        f"gray >= {min_specular}",
    ], (255, 255, 0))
    filtered_panel = image.copy()
    filtered_panel = draw_candidates(filtered_panel, rejected, (0, 0, 255), "X")
    filtered_panel = draw_candidates(filtered_panel, kept, (0, 255, 255), "K")
    filtered_panel = text_block(draw_roi_and_confusers(filtered_panel, tracker), [
        f"5 ROI/confuser gate: kept {len(kept)} rejected {len(rejected)}",
        "yellow K = usable, red X = rejected",
    ])
    tracking_panel = draw_tracker_overlay(image, tracker, detection)
    status_panel = np.full_like(image, 24)
    internal = tracker.tracker
    dbg = dict(getattr(internal, "debug", {}) or {}) if internal is not None else {}
    glint = dbg.get("peak_at_track", "-")
    template = "-"
    if not np.isnan(tracker.last_template_score):
        template = f"{tracker.last_template_score:.2f}"
    status_lines = [
        "6 tracker state",
        f"found: {bool(detection.found)}",
        f"position: {detection.x_px:.1f}, {detection.y_px:.1f}"
        if detection.found else "position: -",
        f"glint at track: {glint}",
        f"template score: {template}",
        f"template rescued: {tracker.template_rescued}",
        "click seed | n/b step | space play | s save",
    ]
    status_panel = text_block(status_panel, status_lines)

    panels = [
        input_panel, gray_panel, diff_panel, motion_panel,
        highlight_panel, filtered_panel, tracking_panel, status_panel,
    ]
    target_w = 420
    target_h = int(round(target_w * image.shape[0] / image.shape[1]))
    resized = [cv2.resize(p, (target_w, target_h), interpolation=cv2.INTER_AREA)
               for p in panels]
    rows = [
        np.hstack(resized[0:2]),
        np.hstack(resized[2:4]),
        np.hstack(resized[4:6]),
        np.hstack(resized[6:8]),
    ]
    return np.vstack(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize each offline stage of the ball tracking pipeline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", default=None,
                        help="image/video source; defaults to camera.reference_image")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--seed-x", type=float, default=None)
    parser.add_argument("--seed-y", type=float, default=None)
    parser.add_argument("--auto-seed", action="store_true",
                        help="use the tracker auto-seed path from config defaults")
    parser.add_argument("--save", default=None,
                        help="write one montage image and exit instead of opening UI")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    source_path = resolve_source(config_path, args.source)
    source = OfflineSource(source_path)

    vision = dict(config.vision)
    if args.auto_seed:
        vision["auto_seed"] = True
    tracker = PipelineBallTracker(vision)
    min_specular = int(vision.get("min_specular", tracker.min_specular))
    tracker.min_specular = min_specular

    state = {
        "index": int(np.clip(args.start_frame, 0, source.count - 1)),
        "prev_gray": None,
        "play": False,
        "seed": None,
    }

    def previous_gray_for(index: int) -> np.ndarray | None:
        if not source.is_video or index <= 0:
            return None
        previous = source.read(index - 1)
        return cv2.cvtColor(previous.image, cv2.COLOR_BGR2GRAY)

    state["prev_gray"] = previous_gray_for(state["index"])

    if args.seed_x is not None and args.seed_y is not None:
        state["seed"] = (float(args.seed_x), float(args.seed_y))

    def render() -> np.ndarray:
        frame = source.read(state["index"])
        if state["seed"] is not None:
            # Prime _last_gray so seed() can capture a template from this
            # offline frame, matching click-to-seed behavior in the live tool.
            tracker._last_gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
            tracker.seed(*state["seed"])
            state["seed"] = None
        detection = tracker.detect(frame.image)
        montage = make_montage(
            frame, state["prev_gray"], tracker, detection, min_specular)
        state["prev_gray"] = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        return montage

    if args.save:
        out = render()
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), out)
        print(f"saved {save_path}")
        source.close()
        return

    mouse: dict[str, tuple[int, int]] = {}

    def on_mouse(event: int, x: int, y: int, *_rest) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["seed"] = (x, y)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print(__doc__)
    print(f"source: {source_path}")

    try:
        while True:
            seed = mouse.pop("seed", None)
            if seed is not None:
                state["seed"] = seed
                state["prev_gray"] = None
                print(f"seeded at {seed}")
            montage = render()
            cv2.imshow(WINDOW, montage)
            delay = 30 if state["play"] and source.is_video else 0
            key = cv2.waitKey(delay) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("n"), 83, 3):
                if source.is_video:
                    state["index"] = min(state["index"] + 1, source.count - 1)
            elif key in (ord("b"), 81, 2):
                if source.is_video:
                    state["index"] = max(state["index"] - 1, 0)
                    state["prev_gray"] = previous_gray_for(state["index"])
            elif key == ord(" "):
                state["play"] = not state["play"]
            elif key == ord("r"):
                tracker = PipelineBallTracker(vision)
                state["index"] = int(np.clip(args.start_frame, 0, source.count - 1))
                state["prev_gray"] = previous_gray_for(state["index"])
                state["play"] = False
            elif key == ord("s"):
                out = Path("data/raw/ball_pipeline_visualizer.png")
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), montage)
                print(f"saved {out}")
            elif source.is_video and state["play"]:
                state["index"] = min(state["index"] + 1, source.count - 1)
                if state["index"] >= source.count - 1:
                    state["play"] = False
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
