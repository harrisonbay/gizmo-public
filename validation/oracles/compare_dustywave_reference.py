#!/usr/bin/env python3
"""Compare an ID-sorted dusty-wave state with the public t=1.2 velocity table."""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
import math
from pathlib import Path


VELOCITY_SCALE = 1.0e-4


def read_state(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_reference(path: Path) -> list[tuple[float, float, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            x, grain_velocity, gas_velocity = map(float, line.split())
            rows.append(
                (x, VELOCITY_SCALE * grain_velocity, VELOCITY_SCALE * gas_velocity)
            )
    if len(rows) < 2:
        raise ValueError("reference table must contain at least two rows")
    return sorted(rows)


def periodic_interpolate(
    reference: list[tuple[float, float, float]], position: float, column: int
) -> float:
    coordinates = [row[0] for row in reference]
    wrapped = position % 1.0
    index = bisect.bisect_left(coordinates, wrapped)
    left = (index - 1) % len(reference)
    right = index % len(reference)
    x_left = coordinates[left]
    x_right = coordinates[right]
    x = wrapped
    if right == 0:
        x_right += 1.0
    if index == 0:
        x += 1.0
    fraction = (x - x_left) / (x_right - x_left)
    return reference[left][column] + fraction * (
        reference[right][column] - reference[left][column]
    )


def metrics(
    state: list[dict[str, str]], reference: list[tuple[float, float, float]]
) -> dict[str, float]:
    result = {}
    for particle_type, column, name in (
        (3, 1, "grain_velocity_x"),
        (0, 2, "gas_velocity_x"),
    ):
        errors = []
        for particle in state:
            if int(particle["particle_type"]) != particle_type:
                continue
            expected = periodic_interpolate(
                reference, float(particle["x"]), column
            )
            errors.append(float(particle["velocity_x"]) - expected)
        if not errors:
            raise ValueError(f"state has no particle type {particle_type}")
        rms = math.sqrt(sum(error * error for error in errors) / len(errors))
        peak = max(abs(row[column]) for row in reference)
        result[f"{name}_absolute_rms"] = rms
        result[f"{name}_peak_normalized_rms"] = rms / peak
    return result


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
        expected = manifest["public_reference"]
        for phase in ("absolute_rms", "peak_normalized_rms"):
            for species in ("grain_velocity_x", "gas_velocity_x"):
                key = f"{species}_{phase}"
                if abs(measured[key] - expected[phase][species]) > 1.0e-14:
                    raise SystemExit(
                        f"{key} mismatch: measured={measured[key]:.17g}, "
                        f"manifest={expected[phase][species]:.17g}"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
