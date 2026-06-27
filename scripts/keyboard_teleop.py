#!/usr/bin/env python3
"""Manual keyboard control of the maze servos.

Streams commands continuously so the firmware watchdog keeps holding the
current tilt instead of snapping back to neutral. Windows only (uses msvcrt).
"""
from __future__ import annotations

import argparse
import sys
import time

from cps_maze.config import load_config
from cps_maze.hardware.serial_link import ArduinoServoLink, ServoCommand

try:
    import msvcrt  # Windows-only non-blocking key reads
except ImportError:  # pragma: no cover - non-Windows
    msvcrt = None


HELP = """
Keyboard teleop for the maze servos
-----------------------------------
  Arrow Left/Right : yaw   -/+   (channel 0)
  Arrow Up/Down    : pitch -/+   (channel 1)
  Space            : return to neutral (0, 0)
  + / -            : bigger / smaller step per press
  q or Esc         : quit (returns to neutral first)

Each arrow press nudges the tilt; the script keeps streaming the command
so the board holds position against the 500 ms firmware watchdog.
"""


def read_key():
    """Non-blocking key read. Returns a token string or None if no key."""
    if not msvcrt.kbhit():
        return None
    ch = msvcrt.getch()
    if ch in (b"\x00", b"\xe0"):  # arrow / function key: a second byte follows
        ch2 = msvcrt.getch()
        return {b"H": "up", b"P": "down", b"K": "left", b"M": "right"}.get(ch2)
    if ch in (b"q", b"Q", b"\x1b"):
        return "quit"
    if ch == b" ":
        return "neutral"
    if ch == b"+":
        return "step_up"
    if ch == b"-":
        return "step_down"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--port", default=None, help="Override serial port, e.g. COM10")
    parser.add_argument("--step", type=float, default=0.05, help="Tilt change per key press")
    parser.add_argument("--rate-hz", type=float, default=20.0, help="Command stream rate")
    parser.add_argument(
        "--limit",
        type=float,
        default=1.0,
        help="Max |tilt| this script will command (0-1). Lower = safer while testing.",
    )
    parser.add_argument("--invert-yaw", action="store_true", help="Flip left/right direction")
    parser.add_argument("--invert-pitch", action="store_true", help="Flip up/down direction")
    parser.add_argument(
        "--swap-axes",
        action="store_true",
        help="Swap which arrow pair drives yaw vs pitch (use if servo channels are swapped)",
    )
    args = parser.parse_args()

    if msvcrt is None:
        sys.exit("keyboard_teleop currently supports Windows only (needs msvcrt).")
    if not 0.0 < args.limit <= 1.0:
        sys.exit("--limit must be in (0, 1].")

    config = load_config(args.config)
    port = args.port or config.serial["port"]

    yaw = 0.0
    pitch = 0.0
    step = args.step
    limit = args.limit
    period = 1.0 / args.rate_hz
    yaw_sign = -1.0 if args.invert_yaw else 1.0
    pitch_sign = -1.0 if args.invert_pitch else 1.0

    def clamp(value: float) -> float:
        return max(-limit, min(limit, value))

    print(HELP)
    print(f"Connecting on {port} (limit=+/-{limit:.2f}) ...")

    with ArduinoServoLink(
        port=port,
        baudrate=int(config.serial["baudrate"]),
        timeout_s=float(config.serial["timeout_s"]),
    ) as link:
        time.sleep(2.0)  # let the Arduino finish its reset after the port opens
        link.neutral()
        print("Connected. Use the arrow keys. Press q or Esc to quit.\n")
        try:
            while True:
                key = read_key()
                if key == "quit":
                    break
                elif key == "neutral":
                    yaw, pitch = 0.0, 0.0
                elif key == "step_up":
                    step = min(0.50, step + 0.01)
                elif key == "step_down":
                    step = max(0.01, step - 0.01)
                elif key in ("up", "down", "left", "right"):
                    if args.swap_axes:  # swap which arrow pair drives each channel
                        key = {"up": "left", "down": "right",
                               "left": "up", "right": "down"}[key]
                    if key == "left":
                        yaw = clamp(yaw - step)
                    elif key == "right":
                        yaw = clamp(yaw + step)
                    elif key == "up":
                        pitch = clamp(pitch - step)
                    elif key == "down":
                        pitch = clamp(pitch + step)

                if key is not None:
                    print(
                        f"\ryaw={yaw:+.2f}  pitch={pitch:+.2f}  step={step:.2f}   ",
                        end="",
                        flush=True,
                    )

                link.send(ServoCommand(yaw=yaw * yaw_sign, pitch=pitch * pitch_sign))
                time.sleep(period)
        finally:
            link.neutral()
            print("\nReturned to neutral. Bye.")


if __name__ == "__main__":
    main()
