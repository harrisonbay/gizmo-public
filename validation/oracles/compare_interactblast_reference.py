#!/usr/bin/env python3
"""Compare an ID-sorted interacting-blast state with the public t=0.038 table."""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
from pathlib import Path
from typing import Callable


GAMMA = 1.4


def read_state(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_reference(path: Path) -> list[tuple[float, ...]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            rows.append(tuple(float(value) for value in line.split()))
    if len(rows) < 2:
        raise ValueError("reference table must contain at least two rows")
    return rows


def linear_interpolate(
    coordinates: list[float], values: list[float], position: float
) -> float:
    index = bisect.bisect_left(coordinates, position)
    if index == 0:
        left, right = 0, 1
    elif index == len(coordinates):
        left, right = len(coordinates) - 2, len(coordinates) - 1
    else:
        left, right = index - 1, index
    fraction = (position - coordinates[left]) / (
        coordinates[right] - coordinates[left]
    )
    return values[left] + fraction * (values[right] - values[left])


def volume_weighted_l1(
    state: list[dict[str, str]],
    reference: list[tuple[float, ...]],
    value: Callable[[dict[str, str]], float],
    reference_value: Callable[[tuple[float, ...]], float],
) -> float:
    weighted_error = 0.0
    total_volume = 0.0
    coordinates = [row[1] for row in reference]
    reference_values = [reference_value(row) for row in reference]
    for particle in state:
        density = float(particle["density"])
        volume = float(particle["mass"]) / density
        position = float(particle["x"])
        expected = linear_interpolate(coordinates, reference_values, position)
        weighted_error += volume * abs(value(particle) - expected)
        total_volume += volume
    return weighted_error / total_volume


def metrics(
    state: list[dict[str, str]], reference: list[tuple[float, ...]]
) -> dict[str, float]:
    density = lambda row: float(row["density"])
    pressure = lambda row: (GAMMA - 1.0) * float(
        row["density"]
    ) * float(row["specific_internal_energy"])
    entropy = lambda row: pressure(row) / density(row) ** GAMMA
    velocity = lambda row: float(row["velocity_x"])
    return {
        "density": volume_weighted_l1(state, reference, density, lambda row: row[2]),
        "pressure": volume_weighted_l1(state, reference, pressure, lambda row: row[6]),
        "entropy": volume_weighted_l1(
            state,
            reference,
            entropy,
            lambda row: row[6] / row[2] ** GAMMA,
        ),
        "velocity_x": volume_weighted_l1(
            state, reference, velocity, lambda row: row[3]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--check-manifest", type=Path)
    args = parser.parse_args()
    measured = metrics(read_state(args.state), read_reference(args.reference))
    print(json.dumps(measured, indent=2, sort_keys=True))
    if args.check_manifest:
        manifest = json.loads(args.check_manifest.read_text(encoding="utf-8"))
        expected = manifest["public_reference"]["volume_weighted_l1"]
        for field, value in measured.items():
            if abs(value - expected[field]) > 1.0e-12:
                raise SystemExit(
                    f"{field} L1 mismatch: measured={value:.17g}, "
                    f"manifest={expected[field]:.17g}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
