#!/usr/bin/env python3
"""Reduce a KH snapshot sequence to deterministic physics diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from validation.oracles.analyze_kh_mcnally import (
    GAMMA,
    OFFICIAL_PARTICLE_COUNT,
    read_snapshot,
    snapshot_diagnostics,
)

EXPECTED_TIMES = tuple(index / 10.0 for index in range(16))
FIELDS = (
    "time",
    "particle_count",
    "mode_amplitude",
    "maximum_vertical_kinetic_energy_density",
    "mass",
    "momentum_x",
    "momentum_y",
    "diagnostic_total_energy",
    "density_minimum",
    "density_maximum",
    "pressure_minimum",
    "pressure_maximum",
    "smoothing_length_minimum",
    "smoothing_length_maximum",
    "smoothing_length_mean",
    "effective_neighbors_minimum",
    "effective_neighbors_maximum",
    "effective_neighbors_mean",
)


def reduce_trajectory(output_directory: Path) -> list[dict[str, float | int]]:
    """Read the first sixteen benchmark snapshots through t=1.5."""
    snapshots = [
        output_directory / f"snapshot_{index:03}.hdf5"
        for index in range(len(EXPECTED_TIMES))
    ]
    missing = [str(path) for path in snapshots if not path.is_file()]
    if missing:
        raise ValueError(f"missing required KH snapshots: {missing}")
    rows: list[dict[str, float | int]] = []
    initial_ids: np.ndarray | None = None
    initial_masses: np.ndarray | None = None
    for path, expected_time in zip(snapshots, EXPECTED_TIMES, strict=True):
        state = read_snapshot(path)
        time = float(state["time"])
        if not math.isclose(time, expected_time, rel_tol=0.0, abs_tol=2.0e-15):
            raise ValueError(f"{path}: time {time} != expected {expected_time}")
        ids = np.asarray(state["particle_ids"])
        masses = np.asarray(state["masses"], dtype=np.float64)
        if not np.array_equal(
            ids, np.arange(1, OFFICIAL_PARTICLE_COUNT + 1, dtype=ids.dtype)
        ):
            raise ValueError(
                f"{path}: ParticleIDs are not exactly 1..{OFFICIAL_PARTICLE_COUNT}"
            )
        if initial_ids is None:
            initial_ids = ids.copy()
            initial_masses = masses.copy()
        elif not np.array_equal(ids, initial_ids) or not np.array_equal(
            masses, initial_masses
        ):
            raise ValueError(f"{path}: particle IDs or per-ID masses changed")

        coordinates = np.asarray(state["coordinates"], dtype=np.float64)
        velocities = np.asarray(state["velocities"], dtype=np.float64)
        density = np.asarray(state["density"], dtype=np.float64)
        internal_energy = np.asarray(state["internal_energy"], dtype=np.float64)
        smoothing_length = np.asarray(state["smoothing_length"], dtype=np.float64)
        pressure = (GAMMA - 1.0) * density * internal_energy
        effective_neighbors = math.pi * smoothing_length**2 * density / masses
        kinetic = 0.5 * masses * np.sum(velocities * velocities, axis=1)
        thermal = masses * internal_energy
        diagnostics = snapshot_diagnostics(state)
        rows.append(
            {
                "time": time,
                "particle_count": int(density.size),
                "mode_amplitude": float(diagnostics["mode_amplitude"]),
                "maximum_vertical_kinetic_energy_density": float(
                    diagnostics["maximum_vertical_kinetic_energy_density"]
                ),
                "mass": math.fsum(float(value) for value in masses),
                "momentum_x": math.fsum(
                    float(value) for value in masses * velocities[:, 0]
                ),
                "momentum_y": math.fsum(
                    float(value) for value in masses * velocities[:, 1]
                ),
                "diagnostic_total_energy": math.fsum(
                    float(value) for value in kinetic + thermal
                ),
                "density_minimum": float(np.min(density)),
                "density_maximum": float(np.max(density)),
                "pressure_minimum": float(np.min(pressure)),
                "pressure_maximum": float(np.max(pressure)),
                "smoothing_length_minimum": float(np.min(smoothing_length)),
                "smoothing_length_maximum": float(np.max(smoothing_length)),
                "smoothing_length_mean": float(np.mean(smoothing_length)),
                "effective_neighbors_minimum": float(np.min(effective_neighbors)),
                "effective_neighbors_maximum": float(np.max(effective_neighbors)),
                "effective_neighbors_mean": float(np.mean(effective_neighbors)),
            }
        )
        if np.any(coordinates[:, 2] != 0.0):
            raise ValueError(f"{path}: nonzero z coordinate")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    rows = reduce_trajectory(args.output_directory)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
