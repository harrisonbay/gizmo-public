#!/usr/bin/env python3
"""Analyze one public advected-Square snapshot against its exact translation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np

PARTICLE_COUNT = 16_384
BOX_SIZE = 1.0
GAMMA = 1.4
BOOST = np.array([1_243.0, -358.0])
INITIALIZED_DENSITY_LOW = 0.007_814_580_197_309_413
INITIALIZED_DENSITY_HIGH = 0.031_258_320_789_237_65
INITIALIZED_SMOOTHING_LENGTH = 0.015_266_824_288_944_47


class SnapshotError(ValueError):
    """A Square fixture or output snapshot violates its pinned schema."""


def _sorted_gas(path: Path) -> tuple[float, dict[str, np.ndarray]]:
    with h5py.File(path, "r") as handle:
        if "PartType0" not in handle:
            raise SnapshotError(f"{path}: missing PartType0")
        header = handle["Header"].attrs
        time = float(header["Time"])
        if not math.isfinite(time):
            raise SnapshotError(f"{path}: Header/Time must be finite")
        gas = handle["PartType0"]
        required = {
            "Coordinates",
            "Velocities",
            "ParticleIDs",
            "Masses",
            "InternalEnergy",
            "Density",
            "SmoothingLength",
        }
        missing = sorted(required - set(gas))
        if missing:
            raise SnapshotError(f"{path}: missing datasets {missing}")
        columns = {name: np.asarray(gas[name]) for name in required}
    ids = columns["ParticleIDs"].astype(np.uint64, copy=False)
    if ids.shape != (PARTICLE_COUNT,):
        raise SnapshotError(f"{path}: expected {PARTICLE_COUNT} particles")
    order = np.argsort(ids)
    columns = {name: values[order] for name, values in columns.items()}
    expected_ids = np.arange(PARTICLE_COUNT, dtype=np.uint64)
    if not np.array_equal(columns["ParticleIDs"], expected_ids):
        raise SnapshotError(f"{path}: ParticleIDs must be exactly 0..16383")
    if columns["Coordinates"].shape != (PARTICLE_COUNT, 3) or columns[
        "Velocities"
    ].shape != (PARTICLE_COUNT, 3):
        raise SnapshotError(f"{path}: vector columns must have shape (16384,3)")
    numeric = (
        columns["Coordinates"],
        columns["Velocities"],
        columns["Masses"],
        columns["InternalEnergy"],
        columns["Density"],
        columns["SmoothingLength"],
    )
    if any(not np.all(np.isfinite(values)) for values in numeric):
        raise SnapshotError(f"{path}: non-finite physical field")
    if (
        np.any(columns["Masses"] <= 0)
        or np.any(columns["InternalEnergy"] <= 0)
        or np.any(columns["Density"] <= 0)
        or np.any(columns["SmoothingLength"] <= 0)
    ):
        raise SnapshotError(f"{path}: non-positive physical field")
    if np.any(columns["Coordinates"][:, :2] < 0) or np.any(
        columns["Coordinates"][:, :2] >= BOX_SIZE
    ):
        raise SnapshotError(f"{path}: planar coordinate outside periodic box")
    if not np.array_equal(columns["Coordinates"][:, 2], np.zeros(PARTICLE_COUNT)):
        raise SnapshotError(f"{path}: z coordinates must remain zero")
    return time, columns


def load_fixture(path: Path) -> dict[str, np.ndarray]:
    """Load and validate the exact hosted 128x128 float32 fixture."""
    time, columns = _sorted_gas(path)
    if time != 0.0:
        raise SnapshotError(f"{path}: fixture time must be zero")
    ids = columns["ParticleIDs"].astype(np.uint64)
    expected_coordinates = np.column_stack(
        (
            (ids % 128 + 0.5) / 128.0,
            (ids // 128 + 0.5) / 128.0,
            np.zeros(PARTICLE_COUNT),
        )
    )
    grid_x = ids % 128
    grid_y = ids // 128
    inside = (grid_x >= 32) & (grid_x < 96) & (grid_y >= 32) & (grid_y < 96)
    expected_mass = np.where(inside, 2.0**-19, 2.0**-21)
    expected_u = np.where(inside, 0.25, 1.0)
    expected_density = np.where(
        inside, 0.031_253_021_210_432_05, 0.007_813_255_302_608_013
    )
    expected_velocity = np.zeros((PARTICLE_COUNT, 3))
    expected_velocity[:, :2] = BOOST
    for label, actual, expected in (
        ("coordinates", columns["Coordinates"], expected_coordinates),
        ("velocities", columns["Velocities"], expected_velocity),
        ("masses", columns["Masses"], expected_mass),
        ("internal energy", columns["InternalEnergy"], expected_u),
        ("stored density", columns["Density"], expected_density),
    ):
        if not np.array_equal(actual, expected):
            raise SnapshotError(f"{path}: {label} differ from the pinned fixture")
    return columns


def periodic_residual(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Return signed minimum-image residuals in the unit periodic box."""
    return np.remainder(actual - expected + 0.5, BOX_SIZE) - 0.5


def analyze_snapshot(
    snapshot_path: Path,
    fixture: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return exact-translation, state, geometry, and conservation diagnostics."""
    time, columns = _sorted_gas(snapshot_path)
    if not np.array_equal(columns["Masses"], fixture["Masses"]):
        raise SnapshotError(f"{snapshot_path}: per-ID masses changed")
    expected_positions = np.remainder(
        fixture["Coordinates"][:, :2] + fixture["Velocities"][:, :2] * time,
        BOX_SIZE,
    )
    position_residual = periodic_residual(
        columns["Coordinates"][:, :2], expected_positions
    )
    velocity_residual = columns["Velocities"] - fixture["Velocities"]
    internal_energy_residual = (
        columns["InternalEnergy"] - fixture["InternalEnergy"]
    )
    reference = columns if baseline is None else baseline
    density_relative = columns["Density"] / reference["Density"] - 1.0
    smoothing_relative = (
        columns["SmoothingLength"] / reference["SmoothingLength"] - 1.0
    )
    expected_density = np.where(
        fixture["Masses"] == 2.0**-19,
        INITIALIZED_DENSITY_HIGH,
        INITIALIZED_DENSITY_LOW,
    )
    density_absolute_relative = columns["Density"] / expected_density - 1.0
    smoothing_absolute_relative = (
        columns["SmoothingLength"] / INITIALIZED_SMOOTHING_LENGTH - 1.0
    )
    pressure = (
        (GAMMA - 1.0) * columns["Density"] * columns["InternalEnergy"]
    )
    effective_neighbors = (
        math.pi
        * columns["SmoothingLength"] ** 2
        * columns["Density"]
        / columns["Masses"]
    )
    masses = columns["Masses"]
    velocities = columns["Velocities"]
    momentum = np.sum(masses[:, None] * velocities, axis=0)
    energy = np.sum(
        masses
        * (
            columns["InternalEnergy"]
            + 0.5 * np.sum(velocities * velocities, axis=1)
        )
    )
    metrics: dict[str, Any] = {
        "snapshot": str(snapshot_path),
        "time": time,
        "particle_count": PARTICLE_COUNT,
        "expected_wrapped_shift": np.remainder(BOOST * time, BOX_SIZE).tolist(),
        "position_component_maximum": np.max(np.abs(position_residual), axis=0).tolist(),
        "position_vector_rms": float(
            np.sqrt(np.mean(np.sum(position_residual**2, axis=1)))
        ),
        "velocity_component_maximum": np.max(
            np.abs(velocity_residual), axis=0
        ).tolist(),
        "velocity_vector_rms": float(
            np.sqrt(np.mean(np.sum(velocity_residual**2, axis=1)))
        ),
        "internal_energy_maximum": float(np.max(np.abs(internal_energy_residual))),
        "density_relative_maximum": float(np.max(np.abs(density_relative))),
        "density_absolute_relative_maximum": float(
            np.max(np.abs(density_absolute_relative))
        ),
        "smoothing_length_relative_maximum": float(
            np.max(np.abs(smoothing_relative))
        ),
        "smoothing_length_absolute_relative_maximum": float(
            np.max(np.abs(smoothing_absolute_relative))
        ),
        "pressure_relative_span": float(
            (np.max(pressure) - np.min(pressure)) / np.mean(pressure)
        ),
        "effective_neighbor_range": [
            float(np.min(effective_neighbors)),
            float(np.max(effective_neighbors)),
        ],
        "mass": float(np.sum(masses)),
        "momentum": momentum.tolist(),
        "total_energy": float(energy),
        "density_range": [
            float(np.min(columns["Density"])),
            float(np.max(columns["Density"])),
        ],
        "smoothing_length_range": [
            float(np.min(columns["SmoothingLength"])),
            float(np.max(columns["SmoothingLength"])),
        ],
    }
    return metrics, columns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(__file__).with_name("square") / "square_ics.hdf5",
    )
    args = parser.parse_args()
    fixture = load_fixture(args.fixture)
    metrics, _ = analyze_snapshot(args.snapshot, fixture)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
