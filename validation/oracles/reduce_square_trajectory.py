#!/usr/bin/env python3
"""Deterministically reduce a raw Square output directory to the pinned CSV."""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

from validation.oracles.check_square_trajectory import check_trajectory

FIELDNAMES = (
    "time",
    "position_x_max",
    "position_y_max",
    "velocity_max",
    "internal_energy_max",
    "density_relative_max",
    "smoothing_relative_max",
    "pressure_relative_span",
    "effective_neighbors_min",
    "effective_neighbors_max",
    "mass",
    "momentum_x",
    "momentum_y",
    "total_energy",
)


def reduce_trajectory(output_directory: Path, fixture_path: Path) -> bytes:
    """Return canonical UTF-8 CSV bytes for all strictly checked snapshots."""
    reports = check_trajectory(output_directory, fixture_path)["snapshots"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    for metrics in reports:
        writer.writerow(
            {
                "time": metrics["time"],
                "position_x_max": metrics["position_component_maximum"][0],
                "position_y_max": metrics["position_component_maximum"][1],
                "velocity_max": max(metrics["velocity_component_maximum"]),
                "internal_energy_max": metrics["internal_energy_maximum"],
                "density_relative_max": metrics["density_relative_maximum"],
                "smoothing_relative_max": metrics[
                    "smoothing_length_relative_maximum"
                ],
                "pressure_relative_span": metrics["pressure_relative_span"],
                "effective_neighbors_min": metrics["effective_neighbor_range"][0],
                "effective_neighbors_max": metrics["effective_neighbor_range"][1],
                "mass": metrics["mass"],
                "momentum_x": metrics["momentum"][0],
                "momentum_y": metrics["momentum"][1],
                "total_energy": metrics["total_energy"],
            }
        )
    return stream.getvalue().encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(__file__).with_name("square") / "square_ics.hdf5",
    )
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--check", type=Path)
    args = parser.parse_args()

    reduced = reduce_trajectory(args.output_directory, args.fixture)
    if args.check is not None:
        expected = args.check.read_bytes()
        if reduced != expected:
            parser.error(f"reduced bytes differ from {args.check}")
        return 0
    args.output.write_bytes(reduced)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
