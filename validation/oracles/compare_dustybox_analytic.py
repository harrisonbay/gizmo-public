#!/usr/bin/env python3
"""Compare an ID-sorted dusty-box state with the analytic Epstein solution."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path


EPSTEIN_ALPHA = 15.0 * math.pi / 128.0


def read_state(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def analytic_velocities(time: float) -> tuple[float, float]:
    """Return `(gas, grain)` for the public equal-density dusty-box profile."""
    if not math.isfinite(time) or time < 0.0:
        raise ValueError("time must be finite and non-negative")
    psi = math.exp(-2.0 * time) / (1.0 + math.sqrt(1.0 + EPSTEIN_ALPHA))
    relative_velocity = 2.0 * psi / (1.0 - EPSTEIN_ALPHA * psi * psi)
    return (
        0.5 * (1.0 - relative_velocity),
        0.5 * (1.0 + relative_velocity),
    )


def metrics(state: list[dict[str, str]], time: float) -> dict[str, float]:
    expected = dict(zip((0, 3), analytic_velocities(time), strict=True))
    result: dict[str, float] = {}
    for particle_type, name in ((0, "gas"), (3, "grain")):
        velocities = [
            float(row["velocity_x"])
            for row in state
            if int(row["particle_type"]) == particle_type
        ]
        if not velocities:
            raise ValueError(f"state has no particle type {particle_type}")
        mean = math.fsum(velocities) / len(velocities)
        errors = [velocity - expected[particle_type] for velocity in velocities]
        result[f"{name}_mean_velocity_x"] = mean
        result[f"{name}_analytic_velocity_x"] = expected[particle_type]
        result[f"{name}_absolute_rms"] = math.sqrt(
            math.fsum(error * error for error in errors) / len(errors)
        )
        result[f"{name}_spatial_stddev"] = math.sqrt(
            math.fsum((velocity - mean) ** 2 for velocity in velocities)
            / len(velocities)
        )
    result["total_momentum_x"] = math.fsum(
        float(row["mass"]) * float(row["velocity_x"]) for row in state
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--check-manifest", type=Path)
    args = parser.parse_args()
    measured = metrics(read_state(args.state), args.time)
    print(json.dumps(measured, indent=2, sort_keys=True))
    if args.check_manifest:
        manifest = json.loads(args.check_manifest.read_text(encoding="utf-8"))
        matching = [
            snapshot
            for snapshot in manifest["snapshots"]
            if abs(snapshot["time"] - args.time) <= 2.0e-15
        ]
        if len(matching) != 1:
            raise SystemExit(
                f"expected one manifest snapshot at time {args.time:.17g}, "
                f"found {len(matching)}"
            )
        expected = matching[0]["analytic_metrics"]
        if set(measured) != set(expected):
            raise SystemExit("analytic metric keys do not match the manifest")
        for key, value in measured.items():
            if abs(value - expected[key]) > 1.0e-14:
                raise SystemExit(
                    f"{key} mismatch: measured={value:.17g}, "
                    f"manifest={expected[key]:.17g}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
