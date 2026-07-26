#!/usr/bin/env python3
"""Gate the full public Square trajectory against exact per-ID advection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from validation.oracles.analyze_square import (
    SnapshotError,
    analyze_snapshot,
    load_fixture,
)

EXPECTED_TIMES = tuple(0.5 * index for index in range(21))
MAXIMUM_POSITION_ERROR = 1.0e-9
MAXIMUM_VELOCITY_ERROR = 2.0e-9
MAXIMUM_INTERNAL_ENERGY_ERROR = 1.0e-9
MAXIMUM_RELATIVE_DENSITY_ERROR = 1.0e-9
MAXIMUM_RELATIVE_SMOOTHING_ERROR = 1.0e-9
MAXIMUM_RELATIVE_PRESSURE_SPAN = 2.0e-9
MINIMUM_EFFECTIVE_NEIGHBORS = 11.999
MAXIMUM_EFFECTIVE_NEIGHBORS = 12.001
MAXIMUM_RELATIVE_MASS_DRIFT = 1.0e-15
MAXIMUM_ABSOLUTE_MOMENTUM_DRIFT = 1.0e-10
MAXIMUM_RELATIVE_ENERGY_DRIFT = 2.0e-12


def trajectory_failures(reports: list[dict[str, Any]]) -> list[str]:
    """Return every scientific failure in an already reduced trajectory."""
    failures: list[str] = []
    if len(reports) != len(EXPECTED_TIMES):
        return [f"expected {len(EXPECTED_TIMES)} reports, found {len(reports)}"]
    numeric_fields = (
        "time",
        "expected_wrapped_shift",
        "position_component_maximum",
        "velocity_component_maximum",
        "internal_energy_maximum",
        "density_relative_maximum",
        "density_absolute_relative_maximum",
        "smoothing_length_relative_maximum",
        "smoothing_length_absolute_relative_maximum",
        "pressure_relative_span",
        "effective_neighbor_range",
        "mass",
        "momentum",
        "total_energy",
    )
    for index, metrics in enumerate(reports):
        for field in numeric_fields:
            value = np.asarray(metrics[field], dtype=np.float64)
            if not np.all(np.isfinite(value)):
                failures.append(f"snapshot {index}: {field} contains a non-finite value")
    if failures:
        return failures
    initial = reports[0]
    for index, (metrics, expected_time) in enumerate(
        zip(reports, EXPECTED_TIMES, strict=True)
    ):
        time = float(metrics["time"])
        if abs(time - expected_time) > 2.0e-15:
            failures.append(
                f"snapshot {index}: time {time:.17g}, expected {expected_time:.17g}"
            )
        if max(metrics["position_component_maximum"]) > MAXIMUM_POSITION_ERROR:
            failures.append(f"t={time}: periodic position error exceeds limit")
        if max(metrics["velocity_component_maximum"]) > MAXIMUM_VELOCITY_ERROR:
            failures.append(f"t={time}: velocity error exceeds limit")
        if metrics["internal_energy_maximum"] > MAXIMUM_INTERNAL_ENERGY_ERROR:
            failures.append(f"t={time}: internal-energy error exceeds limit")
        if metrics["density_relative_maximum"] > MAXIMUM_RELATIVE_DENSITY_ERROR:
            failures.append(f"t={time}: density drift exceeds limit")
        if (
            metrics["density_absolute_relative_maximum"]
            > MAXIMUM_RELATIVE_DENSITY_ERROR
        ):
            failures.append(f"t={time}: absolute initialized density is incorrect")
        if (
            metrics["smoothing_length_relative_maximum"]
            > MAXIMUM_RELATIVE_SMOOTHING_ERROR
        ):
            failures.append(f"t={time}: smoothing-length drift exceeds limit")
        if (
            metrics["smoothing_length_absolute_relative_maximum"]
            > MAXIMUM_RELATIVE_SMOOTHING_ERROR
        ):
            failures.append(
                f"t={time}: absolute initialized smoothing length is incorrect"
            )
        if metrics["pressure_relative_span"] > MAXIMUM_RELATIVE_PRESSURE_SPAN:
            failures.append(f"t={time}: pressure is not spatially uniform")
        neighbors = metrics["effective_neighbor_range"]
        if (
            neighbors[0] < MINIMUM_EFFECTIVE_NEIGHBORS
            or neighbors[1] > MAXIMUM_EFFECTIVE_NEIGHBORS
        ):
            failures.append(f"t={time}: effective-neighbor range {neighbors} is invalid")
        mass_drift = abs(metrics["mass"] / initial["mass"] - 1.0)
        momentum_drift = max(
            abs(
                float(metrics["momentum"][component])
                - float(initial["momentum"][component])
            )
            for component in range(3)
        )
        energy_drift = abs(metrics["total_energy"] / initial["total_energy"] - 1.0)
        metrics["relative_mass_drift"] = mass_drift
        metrics["maximum_absolute_component_momentum_drift"] = momentum_drift
        metrics["relative_total_energy_drift"] = energy_drift
        if mass_drift > MAXIMUM_RELATIVE_MASS_DRIFT:
            failures.append(f"t={time}: mass drift exceeds limit")
        if momentum_drift > MAXIMUM_ABSOLUTE_MOMENTUM_DRIFT:
            failures.append(f"t={time}: momentum drift exceeds limit")
        if energy_drift > MAXIMUM_RELATIVE_ENERGY_DRIFT:
            failures.append(f"t={time}: energy drift exceeds limit")
        if index % 2 == 1:
            expected_shift = np.asarray(metrics["expected_wrapped_shift"])
            if abs(expected_shift[0] - 0.5) > 1.0e-15:
                failures.append(f"t={time}: odd snapshot lacks half-box x translation")
    return failures


def check_trajectory(output_directory: Path, fixture_path: Path) -> dict[str, Any]:
    """Check all 21 public outputs and return their diagnostics."""
    fixture = load_fixture(fixture_path)
    baseline = None
    reports: list[dict[str, Any]] = []
    failures: list[str] = []
    for index in range(len(EXPECTED_TIMES)):
        path = output_directory / f"snapshot_{index:03}.hdf5"
        if not path.is_file():
            failures.append(f"missing {path.name}")
            continue
        metrics, columns = analyze_snapshot(path, fixture, baseline)
        if baseline is None:
            baseline = columns
        reports.append(metrics)
    extras = sorted(output_directory.glob("snapshot_*.hdf5"))[len(EXPECTED_TIMES) :]
    if extras:
        failures.append(f"unexpected extra snapshots: {[path.name for path in extras]}")
    failures.extend(trajectory_failures(reports))
    if failures:
        raise SnapshotError("; ".join(failures))
    return {"snapshots": reports}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(__file__).with_name("square") / "square_ics.hdf5",
    )
    args = parser.parse_args()
    try:
        report = check_trajectory(args.output_directory, args.fixture)
    except SnapshotError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
