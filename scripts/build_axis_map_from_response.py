#!/usr/bin/env python3
"""Build an axis-map calibration file from a measured response matrix.

Use this when `scripts/axis_check.py` already printed a response matrix and
you want to create a calibration file without repeating the hardware pulses.

Example:

    PYTHONPATH=src python3 scripts/build_axis_map_from_response.py \
      139.3 -3.4 9.3 123.1 \
      --output calibration/axis_map_response_estimate.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cps_maze.control.axis_map import (
    normalized_response_to_axis_map,
    snap_response_to_axis_map,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "response",
        nargs=4,
        type=float,
        metavar=("R00", "R01", "R10", "R11"),
        help="Measured response matrix in row-major order: [[R00, R01], [R10, R11]]",
    )
    parser.add_argument("--output", default="calibration/axis_map_response_estimate.npz")
    parser.add_argument("--mode", choices=["normalized-response", "snap"],
                        default="normalized-response")
    parser.add_argument("--response-scale-mm-per-unit", type=float, default=None,
                        help="Scale for normalized-response mode. Defaults to the "
                             "median measured one-axis response.")
    args = parser.parse_args()

    response = np.array(args.response, dtype=float).reshape(2, 2)
    if args.mode == "snap":
        axis_map = snap_response_to_axis_map(response)
    else:
        axis_map = normalized_response_to_axis_map(
            response,
            response_scale_mm_per_unit=args.response_scale_mm_per_unit,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    axis_map.save(output)

    print("response matrix (board mm per unit servo command):")
    print(response.round(3))
    print("\nboard->servo matrix saved to", output)
    print(axis_map.matrix.round(4))
    print("\neffective response after board->servo map:")
    print((response @ axis_map.matrix).round(3))


if __name__ == "__main__":
    main()
