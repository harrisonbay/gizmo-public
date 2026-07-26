#!/usr/bin/env python3
"""Gate a seven-snapshot Rust Gresho trajectory against physics and corrected C."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from validation.oracles.compare_gresho_analytic import (
    OFFICIAL_PARTICLE_COUNT,
    SnapshotError,
    comparison_metrics,
    global_diagnostics,
    read_snapshot,
)

EXPECTED_TIMES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)

# Analytic regression ceilings. These are deliberately wider than corrected-C
# measurements and do not substitute for a future resolution-convergence study.
ANALYTIC_LIMITS = {
    "density_l1": 0.012,
    "pressure_normalized_l1": 0.08,
    "specific_internal_energy_normalized_l1": 0.045,
    "radial_velocity_rms": 0.04,
    "tangential_velocity_l1": 0.05,
    "velocity_vector_l1": 0.065,
}
MINIMUM_PEAK_TANGENTIAL_VELOCITY = 0.85
MAXIMUM_TERMINAL_C_POSITION_L1 = 0.08
MAXIMUM_TERMINAL_C_VELOCITY_L1 = 0.12
MAXIMUM_C_DENSITY_L1 = 0.01
MAXIMUM_C_INTERNAL_ENERGY_NORMALIZED_L1 = 0.04
MAXIMUM_C_SMOOTHING_LENGTH_MEAN_RELATIVE_ERROR = 0.01

# The port may take a different valid retry branch locally. Requiring its
# aggregate analytic error to stay near corrected C still catches no-op,
# wrong-phase, wrong-gamma, and materially more diffusive implementations.
CORRECTED_C_RATIO = 1.20
CORRECTED_C_FLOORS = {
    "density_l1": 1.0e-3,
    "pressure_normalized_l1": 5.0e-3,
    "specific_internal_energy_normalized_l1": 5.0e-3,
    "radial_velocity_rms": 2.0e-3,
    "tangential_velocity_l1": 2.0e-3,
    "velocity_vector_l1": 3.0e-3,
}


def _read_corrected_c_table(path: Path) -> dict[str, np.ndarray]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != OFFICIAL_PARTICLE_COUNT:
        raise SnapshotError(
            f"{path} has {len(rows)} corrected-C rows, "
            f"expected {OFFICIAL_PARTICLE_COUNT}"
        )
    ids = np.array([int(row["particle_id"]) for row in rows], dtype=np.uint64)
    if not np.array_equal(ids, np.arange(OFFICIAL_PARTICLE_COUNT, dtype=np.uint64)):
        raise SnapshotError(f"{path} corrected-C IDs are not exactly 0..4091")
    return {
        name: np.array([float(row[name]) for row in rows], dtype=np.float64)
        for name in (
            "x",
            "y",
            "velocity_x",
            "velocity_y",
            "density",
            "specific_internal_energy",
            "smoothing_length",
            "mass",
        )
    }


def corrected_c_differentials(
    state: dict[str, Any], reference: dict[str, np.ndarray]
) -> dict[str, float]:
    """Return mass-weighted per-ID differences for localization and review."""
    masses = np.asarray(state["masses"], dtype=np.float64)
    weights = masses / math.fsum(float(value) for value in masses)

    def mean(values: np.ndarray) -> float:
        return float(np.sum(weights * values))

    coordinates = np.asarray(state["coordinates"], dtype=np.float64)
    velocities = np.asarray(state["velocities"], dtype=np.float64)
    dx = (coordinates[:, 0] - reference["x"] + 0.5) % 1.0 - 0.5
    dy = (coordinates[:, 1] - reference["y"] + 0.5) % 1.0 - 0.5
    velocity_delta = np.hypot(
        velocities[:, 0] - reference["velocity_x"],
        velocities[:, 1] - reference["velocity_y"],
    )
    internal_scale = 4.0 * math.log(2.0) - 2.0
    internal_scale /= float(state["gamma"]) - 1.0
    return {
        "position_vector_l1": mean(np.hypot(dx, dy)),
        "velocity_vector_l1": mean(velocity_delta),
        "density_l1": mean(np.abs(state["density"] - reference["density"])),
        "specific_internal_energy_normalized_l1": mean(
            np.abs(state["internal_energy"] - reference["specific_internal_energy"])
        )
        / internal_scale,
        "smoothing_length_mean_relative_error": mean(
            np.abs(state["smoothing_length"] - reference["smoothing_length"])
            / reference["smoothing_length"]
        ),
    }


def check_trajectory(
    output_directory: Path, manifest_path: Path
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["snapshots"]
    if [record["time"] for record in records] != list(EXPECTED_TIMES):
        raise SnapshotError("corrected-C manifest does not contain the seven public times")

    reports: list[dict[str, Any]] = []
    initial_masses: np.ndarray | None = None
    initial_diagnostics: dict[str, Any] | None = None
    for index, (expected_time, record) in enumerate(zip(EXPECTED_TIMES, records)):
        snapshot_path = output_directory / f"snapshot_{index:03}.hdf5"
        state = read_snapshot(snapshot_path)
        if float(state["time"]) != expected_time:
            raise SnapshotError(
                f"{snapshot_path} has time {state['time']}, expected {expected_time}"
            )
        expected_ids = np.arange(OFFICIAL_PARTICLE_COUNT, dtype=np.uint64)
        if not np.array_equal(state["ids"], expected_ids):
            raise SnapshotError(f"{snapshot_path} IDs are not exactly 0..4091")

        masses = np.asarray(state["masses"], dtype=np.float64)
        if initial_masses is None:
            initial_masses = masses.copy()
        elif not np.array_equal(masses, initial_masses):
            raise SnapshotError(f"{snapshot_path} changed one or more per-ID masses")

        metrics = comparison_metrics(state)
        failures = [
            f"t={expected_time}: {name}={metrics[name]:.8g} exceeds "
            f"independent limit {limit:.8g}"
            for name, limit in ANALYTIC_LIMITS.items()
            if metrics[name] > limit
        ]
        corrected_metrics = record["analytic"]
        for name, floor in CORRECTED_C_FLOORS.items():
            limit = CORRECTED_C_RATIO * float(corrected_metrics[name]) + floor
            if metrics[name] > limit:
                failures.append(
                    f"t={expected_time}: {name}={metrics[name]:.8g} exceeds "
                    f"corrected-C proximity limit {limit:.8g}"
                )

        radius = np.asarray(state["radius"])
        annulus = (radius >= 0.15) & (radius <= 0.25)
        peak = float(np.max(np.asarray(state["tangential_velocity"])[annulus]))
        if peak < MINIMUM_PEAK_TANGENTIAL_VELOCITY:
            failures.append(
                f"t={expected_time}: peak tangential velocity {peak:.8g} is below "
                f"{MINIMUM_PEAK_TANGENTIAL_VELOCITY:.8g}"
            )

        diagnostics = global_diagnostics(state)
        if initial_diagnostics is None:
            initial_diagnostics = diagnostics
        else:
            mass_drift = abs(diagnostics["mass"] - initial_diagnostics["mass"])
            energy_drift = abs(diagnostics["energy"] - initial_diagnostics["energy"])
            momentum_drift = max(
                abs(value - initial)
                for value, initial in zip(
                    diagnostics["momentum"], initial_diagnostics["momentum"]
                )
            )
            angular_drift = abs(
                diagnostics["angular_momentum_z"]
                - initial_diagnostics["angular_momentum_z"]
            ) / abs(initial_diagnostics["angular_momentum_z"])
            if mass_drift > 5.0e-15 * abs(initial_diagnostics["mass"]):
                failures.append(f"t={expected_time}: mass drift is {mass_drift:.8g}")
            # Public snapshots store half-step velocities and predicted thermal
            # energy. Their naive diagnostic energy is therefore staggered;
            # corrected C varies by 3.3e-6 relative over this trajectory.
            if energy_drift > 1.0e-5 * abs(initial_diagnostics["energy"]):
                failures.append(f"t={expected_time}: energy drift is {energy_drift:.8g}")
            if momentum_drift > 1.0e-10:
                failures.append(
                    f"t={expected_time}: component momentum drift is {momentum_drift:.8g}"
                )
            if angular_drift > 0.02:
                failures.append(
                    f"t={expected_time}: relative angular-momentum drift is "
                    f"{angular_drift:.8g}"
                )
        if failures:
            raise SnapshotError("; ".join(failures))

        reference = _read_corrected_c_table(manifest_path.parent / record["table"])
        if not np.array_equal(masses, reference["mass"]):
            raise SnapshotError(
                f"t={expected_time}: Rust masses differ from corrected-C per-ID masses"
            )
        differentials = corrected_c_differentials(state, reference)
        if (
            differentials["density_l1"] > MAXIMUM_C_DENSITY_L1
            or differentials["specific_internal_energy_normalized_l1"]
            > MAXIMUM_C_INTERNAL_ENERGY_NORMALIZED_L1
            or differentials["smoothing_length_mean_relative_error"]
            > MAXIMUM_C_SMOOTHING_LENGTH_MEAN_RELATIVE_ERROR
        ):
            raise SnapshotError(
                f"t={expected_time}: density/internal-energy/smoothing-length "
                "differential is too far from corrected C"
            )
        if index == 0 and (
            differentials["position_vector_l1"] > 1.0e-14
            or differentials["velocity_vector_l1"] > 5.0e-5
        ):
            raise SnapshotError(
                "t=0 does not reproduce corrected C's initial half-kick output phase"
            )
        if index == len(EXPECTED_TIMES) - 1 and (
            differentials["position_vector_l1"] > MAXIMUM_TERMINAL_C_POSITION_L1
            or differentials["velocity_vector_l1"] > MAXIMUM_TERMINAL_C_VELOCITY_L1
        ):
            raise SnapshotError(
                "terminal per-ID displacement/velocity is too far from corrected C; "
                "the evolution may be absent or use the wrong phase"
            )
        reports.append(
            {
                "time": expected_time,
                "metrics": metrics,
                "peak_tangential_velocity": peak,
                "diagnostics": diagnostics,
                "corrected_c_differentials": differentials,
            }
        )
    return {"snapshots": reports}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("gresho") / "evolution-manifest.json",
    )
    args = parser.parse_args()
    try:
        report = check_trajectory(args.output_directory, args.manifest)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
