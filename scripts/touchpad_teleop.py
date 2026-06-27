#!/usr/bin/env python3
"""Touchpad / mouse joystick control of the maze servos.

The cursor acts as a joystick gimbal:
  - Slide finger on trackpad → board tilts proportionally to finger offset from center
  - Lift finger (or stop moving) → cursor stays at center → board returns to neutral
  - The cursor is warped back to screen center every tick so it never drifts

Press q or Esc to quit.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time

from cps_maze.config import load_config
from cps_maze.hardware.serial_link import ArduinoServoLink, ServoCommand

try:
    from pynput import keyboard as kb
    from pynput.mouse import Controller as MouseController
except ImportError:
    sys.exit("pynput is required: pip3 install pynput")


HELP = """
Touchpad joystick control
--------------------------
  Slide finger  : tilt board (proportional to distance from center)
  Lift finger   : board returns to neutral
  q / Esc       : quit

Tip: --sensitivity controls how many pixels of movement = full tilt.
     Lower = more sensitive. --smooth controls lag vs jitter trade-off.
"""


def _get_screen_center(cx_override: int | None, cy_override: int | None) -> tuple[int, int]:
    if cx_override is not None and cy_override is not None:
        return cx_override, cy_override
    try:
        from AppKit import NSScreen  # type: ignore[import]
        frame = NSScreen.mainScreen().frame()
        return int(frame.size.width // 2), int(frame.size.height // 2)
    except Exception:
        pass
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        cx = root.winfo_screenwidth() // 2
        cy = root.winfo_screenheight() // 2
        root.destroy()
        return cx, cy
    except Exception:
        pass
    # Hard fallback — common MacBook resolution
    return 960, 600


def main() -> None:
    parser = argparse.ArgumentParser(description="Touchpad joystick control for the maze servos.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--port", default=None)
    parser.add_argument(
        "--sensitivity", type=float, default=150.0,
        help="Pixels of finger movement from center for full tilt. Default 150.",
    )
    parser.add_argument(
        "--deadzone", type=float, default=6.0,
        help="Pixel radius around center treated as neutral. Default 6.",
    )
    parser.add_argument(
        "--smooth", type=float, default=0.6,
        help="Exponential smoothing factor 0-1 (0=frozen, 1=raw). Default 0.6.",
    )
    parser.add_argument(
        "--max-tilt", type=float, default=0.9,
        help="Maximum tilt angle sent to servos (0-1). Default 0.9.",
    )
    parser.add_argument(
        "--rate-hz", type=float, default=200.0,
        help="Command stream rate in Hz. Default 200.",
    )
    parser.add_argument("--cx", type=int, default=None, help="Screen center X override (px).")
    parser.add_argument("--cy", type=int, default=None, help="Screen center Y override (px).")
    parser.add_argument("--invert-yaw",   action="store_true", help="Flip front/back direction.")
    parser.add_argument("--invert-pitch", action="store_true", help="Flip left/right direction.")
    args = parser.parse_args()
    args.yaw_sign   = -1.0 if args.invert_yaw   else 1.0
    args.pitch_sign = 1.0 if args.invert_pitch else -1.0

    if not 0.0 < args.smooth <= 1.0:
        sys.exit("--smooth must be in (0, 1].")
    if not 0.0 < args.max_tilt <= 1.0:
        sys.exit("--max-tilt must be in (0, 1].")

    config = load_config(args.config)
    port = args.port or config.serial["port"]
    period = 1.0 / args.rate_hz

    cx, cy = _get_screen_center(args.cx, args.cy)

    quit_flag = threading.Event()

    def on_press(key):
        char = getattr(key, "char", None)
        if char in ("q", "Q") or key == kb.Key.esc:
            quit_flag.set()
            return False

    print(HELP)
    print(f"Screen center detected: ({cx}, {cy})")
    print(f"Connecting on {port}  sensitivity={args.sensitivity}px  deadzone={args.deadzone}px  "
          f"smooth={args.smooth}  max_tilt=±{args.max_tilt}  {args.rate_hz:.0f} Hz ...")

    mc = MouseController()

    with ArduinoServoLink(
        port=port,
        baudrate=int(config.serial["baudrate"]),
        timeout_s=float(config.serial["timeout_s"]),
    ) as link:
        time.sleep(2.0)
        link.neutral()

        # Warp cursor to center before starting so first delta is clean
        mc.position = (cx, cy)
        time.sleep(0.05)

        print("Ready. Slide finger on trackpad to tilt. Lift to neutral. q / Esc = quit.\n")

        listener = kb.Listener(on_press=on_press)
        listener.start()

        smoothed_yaw   = 0.0
        smoothed_pitch = 0.0
        alpha = args.smooth

        next_tick = time.monotonic()
        try:
            while not quit_flag.is_set():
                # Read cursor offset from center
                x, y = mc.position
                dx = x - cx
                dy = y - cy

                # Apply deadzone
                if abs(dx) < args.deadzone:
                    dx = 0.0
                if abs(dy) < args.deadzone:
                    dy = 0.0

                # Map to tilt: Y axis → yaw (front/back), X axis → pitch (left/right)
                raw_yaw   = max(-args.max_tilt, min(args.max_tilt,  dy / args.sensitivity)) * args.yaw_sign
                raw_pitch = max(-args.max_tilt, min(args.max_tilt,  dx / args.sensitivity)) * args.pitch_sign

                # Exponential smoothing to filter jitter
                smoothed_yaw   = alpha * raw_yaw   + (1.0 - alpha) * smoothed_yaw
                smoothed_pitch = alpha * raw_pitch + (1.0 - alpha) * smoothed_pitch

                # Warp cursor back to center — makes it act like a gimbal
                mc.position = (cx, cy)

                print(
                    f"\ryaw={smoothed_yaw:+.3f}  pitch={smoothed_pitch:+.3f}  "
                    f"(raw dx={dx:+.0f} dy={dy:+.0f})   ",
                    end="", flush=True,
                )

                link.send(ServoCommand(yaw=smoothed_yaw, pitch=smoothed_pitch))

                next_tick += period
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)

        finally:
            listener.stop()
            mc.position = (cx, cy)
            link.neutral()
            print("\nReturned to neutral. Bye.")


if __name__ == "__main__":
    main()
