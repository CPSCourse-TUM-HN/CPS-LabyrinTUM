#!/usr/bin/env python3
"""Hold the camera open once and publish frames to shared memory.

Start this ONCE at the beginning of a calibration/lab session:

    python scripts/camera_server.py

It pays the MSMF open cost a single time (10-30 s), then streams the latest
frame into shared memory. Every other script (calibration tools, axis_check,
run_autonomous, debug tools) automatically attaches to it and opens instantly
- CameraCapture detects the running server. Leave this window running; press
Ctrl-C (or 'q' in the preview window) to stop it and release the camera.

Only one process can hold a USB camera at a time, so do NOT also let another
script open the device directly while the server runs - they all share this.
"""
from __future__ import annotations

import argparse
import sys
from time import monotonic, sleep

import cv2

from cps_maze.camera import CameraCapture
from cps_maze.camera_share import SharedFrameWriter
from cps_maze.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", type=int, default=None,
                        help="override camera.device_index")
    parser.add_argument("--preview", action="store_true",
                        help="show a live preview window (costs a little fps)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.device is not None:
        config.camera["device_index"] = args.device
    # The server publishes RAW frames; each client applies its own flips, so
    # semantics match opening the device directly. Force flips off here.
    server_camera_config = dict(config.camera)
    server_camera_config["flip_horizontal"] = False
    server_camera_config["flip_vertical"] = False
    # Tell CameraCapture NOT to try attaching to a server (it IS the server).
    server_camera_config["use_shared_server"] = False

    width = int(config.camera["width"])
    height = int(config.camera["height"])

    print("Opening camera (MSMF can take 10-30 s the first time)...")
    writer: SharedFrameWriter | None = None
    frames = 0
    report_at = monotonic() + 2.0
    try:
        with CameraCapture(server_camera_config) as camera:
            obs = camera.observed_settings()
            print(f"camera open: {obs['width']}x{obs['height']} "
                  f"fourcc={obs['fourcc']} driver_fps={obs['fps']:.0f}")
            # A frame may negotiate to a different size than requested; size the
            # shared block to the ACTUAL frame.
            first = camera.read()
            h, w = first.image.shape[:2]
            channels = first.image.shape[2] if first.image.ndim == 3 else 1
            writer = SharedFrameWriter(w, h, channels)
            writer.write(first.image, first.timestamp_s)
            print(f"serving frames on shared memory as '{writer.shm.name}'. "
                  "Leave this running; Ctrl-C to stop.")
            while True:
                frame = camera.read()
                # Guard against a resolution hiccup mid-stream.
                if frame.image.shape[:2] != (h, w):
                    frame_img = cv2.resize(frame.image, (w, h))
                else:
                    frame_img = frame.image
                writer.write(frame_img, frame.timestamp_s)
                frames += 1
                now = monotonic()
                if now >= report_at:
                    fps = frames / (now - (report_at - 2.0))
                    print(f"  serving ~{fps:.0f} fps", end="\r", flush=True)
                    frames = 0
                    report_at = now + 2.0
                if args.preview:
                    cv2.imshow("camera_server (q to quit)", frame_img)
                    if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                        break
    except KeyboardInterrupt:
        print("\nstopping camera server.")
    finally:
        if writer is not None:
            writer.close(unlink=True)
        cv2.destroyAllWindows()
    print("camera released.")


if __name__ == "__main__":
    main()
