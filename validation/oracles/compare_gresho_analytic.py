#!/usr/bin/env python3
"""Validate and compare a Gresho-vortex HDF5 snapshot with the exact equilibrium."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

import numpy as np

try:
    import h5py
except ModuleNotFoundError:  # The analytic helpers do not require HDF5.
    h5py = None  # type: ignore[assignment]


GAMMA = 1.4
BOX_SIZE = 1.0
CENTER = (0.5, 0.5)
OFFICIAL_PARTICLE_COUNT = 4_092
PRESSURE_RANGE = 4.0 * math.log(2.0) - 2.0
METRIC_NAMES = frozenset(
    {
        "density_l1",
        "pressure_l1",
        "pressure_normalized_l1",
        "specific_internal_energy_normalized_l1",
        "radial_velocity_rms",
        "tangential_velocity_l1",
        "velocity_vector_l1",
    }
)


class SnapshotError(ValueError):
    """The input is not a valid public-profile Gresho snapshot."""


def analytic_fields(radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return exact tangential velocity and pressure at nonnegative radii."""
    radius = np.asarray(radius, dtype=np.float64)
    if radius.ndim != 1 or not np.all(np.isfinite(radius)) or np.any(radius < 0.0):
        raise ValueError("radius must be a finite, nonnegative one-dimensional array")

    tangential = np.zeros_like(radius)
    pressure = np.full_like(radius, 3.0 + 4.0 * math.log(2.0))
    inner = radius < 0.2
    middle = (radius >= 0.2) & (radius < 0.4)
    tangential[inner] = 5.0 * radius[inner]
    pressure[inner] = 5.0 + 12.5 * radius[inner] ** 2
    tangential[middle] = 2.0 - 5.0 * radius[middle]
    middle_radius = radius[middle]
    pressure[middle] = (
        9.0
        + 12.5 * middle_radius**2
        - 20.0 * middle_radius
        + 4.0 * np.log(5.0 * middle_radius)
    )
    return tangential, pressure


def _read_array(group: h5py.Group, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if name not in group:
        raise SnapshotError(f"missing PartType0/{name}")
    values = np.asarray(group[name])
    if values.shape != shape:
        raise SnapshotError(
            f"PartType0/{name} has shape {values.shape}, expected {shape}"
        )
    return values


def read_snapshot(
    path: Path,
    *,
    gamma: float = GAMMA,
    dimensional_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Read a checked, particle-ID-sorted public-profile snapshot."""
    if h5py is None:
        raise SnapshotError(
            "h5py is required to read snapshots; run with `uv run --with h5py`"
        )
    if not math.isfinite(gamma) or gamma <= 1.0:
        raise SnapshotError("gamma must be finite and greater than one")
    if not math.isfinite(dimensional_tolerance) or dimensional_tolerance < 0.0:
        raise SnapshotError("dimensional tolerance must be finite and nonnegative")

    with h5py.File(path, "r") as handle:
        if "Header" not in handle or "PartType0" not in handle:
            raise SnapshotError("snapshot must contain Header and PartType0")
        header = handle["Header"].attrs
        box_size = float(header.get("BoxSize", math.nan))
        time = float(header.get("Time", math.nan))
        if box_size != BOX_SIZE:
            raise SnapshotError(f"Header/BoxSize is {box_size}, expected {BOX_SIZE}")
        if not math.isfinite(time) or time < 0.0:
            raise SnapshotError("Header/Time must be finite and nonnegative")

        group = handle["PartType0"]
        if "ParticleIDs" not in group:
            raise SnapshotError("missing PartType0/ParticleIDs")
        ids = np.asarray(group["ParticleIDs"])
        if ids.ndim != 1 or ids.size == 0:
            raise SnapshotError("ParticleIDs must be a nonempty one-dimensional array")
        count = ids.size
        coordinates = _read_array(group, "Coordinates", (count, 3)).astype(np.float64)
        velocities = _read_array(group, "Velocities", (count, 3)).astype(np.float64)
        masses = _read_array(group, "Masses", (count,)).astype(np.float64)
        density = _read_array(group, "Density", (count,)).astype(np.float64)
        internal_energy = _read_array(group, "InternalEnergy", (count,)).astype(
            np.float64
        )
        smoothing_length = _read_array(group, "SmoothingLength", (count,)).astype(
            np.float64
        )

    if len(np.unique(ids)) != count:
        raise SnapshotError("ParticleIDs are not unique")
    columns = (
        coordinates,
        velocities,
        masses,
        density,
        internal_energy,
        smoothing_length,
    )
    if not all(np.all(np.isfinite(column)) for column in columns):
        raise SnapshotError("particle fields contain non-finite values")
    if np.any(coordinates < 0.0) or np.any(coordinates >= BOX_SIZE):
        raise SnapshotError("coordinates lie outside the periodic unit box")
    if (
        np.max(np.abs(coordinates[:, 2])) > dimensional_tolerance
        or np.max(np.abs(velocities[:, 2])) > dimensional_tolerance
    ):
        raise SnapshotError("snapshot is not strictly two-dimensional")
    if (
        np.any(masses <= 0.0)
        or np.any(density <= 0.0)
        or np.any(internal_energy <= 0.0)
        or np.any(smoothing_length <= 0.0)
    ):
        raise SnapshotError("mass, density, internal energy, and Hsml must be positive")

    order = np.argsort(ids, kind="stable")
    ids = ids[order].astype(np.uint64)
    coordinates = coordinates[order]
    velocities = velocities[order]
    masses = masses[order]
    density = density[order]
    internal_energy = internal_energy[order]
    smoothing_length = smoothing_length[order]

    dx = (coordinates[:, 0] - CENTER[0] + 0.5) % BOX_SIZE - 0.5
    dy = (coordinates[:, 1] - CENTER[1] + 0.5) % BOX_SIZE - 0.5
    radius = np.hypot(dx, dy)
    radial_velocity = np.zeros(count, dtype=np.float64)
    tangential_velocity = np.zeros(count, dtype=np.float64)
    nonzero = radius > 0.0
    radial_velocity[nonzero] = (
        velocities[nonzero, 0] * dx[nonzero] + velocities[nonzero, 1] * dy[nonzero]
    ) / radius[nonzero]
    tangential_velocity[nonzero] = (
        -velocities[nonzero, 0] * dy[nonzero] + velocities[nonzero, 1] * dx[nonzero]
    ) / radius[nonzero]
    analytic_tangential, analytic_pressure = analytic_fields(radius)
    analytic_velocity = np.zeros((count, 2), dtype=np.float64)
    analytic_velocity[nonzero, 0] = (
        -analytic_tangential[nonzero] * dy[nonzero] / radius[nonzero]
    )
    analytic_velocity[nonzero, 1] = (
        analytic_tangential[nonzero] * dx[nonzero] / radius[nonzero]
    )

    pressure = (gamma - 1.0) * density * internal_energy
    return {
        "time": time,
        "gamma": gamma,
        "ids": ids,
        "coordinates": coordinates,
        "velocities": velocities,
        "masses": masses,
        "density": density,
        "internal_energy": internal_energy,
        "smoothing_length": smoothing_length,
        "radius": radius,
        "radial_velocity": radial_velocity,
        "tangential_velocity": tangential_velocity,
        "pressure": pressure,
        "analytic_density": np.ones(count, dtype=np.float64),
        "analytic_pressure": analytic_pressure,
        "analytic_internal_energy": analytic_pressure / (gamma - 1.0),
        "analytic_tangential_velocity": analytic_tangential,
        "analytic_velocity": analytic_velocity,
    }


def comparison_metrics(state: dict[str, Any]) -> dict[str, float]:
    """Compute mass-weighted particlewise errors against the exact field."""
    mass = np.asarray(state["masses"], dtype=np.float64)
    weights = mass / math.fsum(float(value) for value in mass)

    def weighted_mean(values: np.ndarray) -> float:
        return float(np.sum(weights * values))

    pressure_scale = PRESSURE_RANGE
    internal_scale = pressure_scale / (float(state["gamma"]) - 1.0)
    vector_error = np.linalg.norm(
        np.asarray(state["velocities"])[:, :2] - state["analytic_velocity"], axis=1
    )
    return {
        "density_l1": weighted_mean(
            np.abs(state["density"] - state["analytic_density"])
        ),
        "pressure_l1": weighted_mean(
            np.abs(state["pressure"] - state["analytic_pressure"])
        ),
        "pressure_normalized_l1": weighted_mean(
            np.abs(state["pressure"] - state["analytic_pressure"])
        )
        / pressure_scale,
        "specific_internal_energy_normalized_l1": weighted_mean(
            np.abs(state["internal_energy"] - state["analytic_internal_energy"])
        )
        / internal_scale,
        "radial_velocity_rms": math.sqrt(
            weighted_mean(np.asarray(state["radial_velocity"]) ** 2)
        ),
        "tangential_velocity_l1": weighted_mean(
            np.abs(state["tangential_velocity"] - state["analytic_tangential_velocity"])
        ),
        "velocity_vector_l1": weighted_mean(vector_error),
    }


def global_diagnostics(state: dict[str, Any]) -> dict[str, Any]:
    """Return reproducible conservation diagnostics without asserting a tolerance."""
    masses = np.asarray(state["masses"], dtype=np.float64)
    velocities = np.asarray(state["velocities"], dtype=np.float64)
    kinetic = 0.5 * np.sum(velocities**2, axis=1)
    total_mass = math.fsum(float(value) for value in masses)
    momentum = [
        math.fsum(float(value) for value in masses * velocities[:, component])
        for component in range(3)
    ]
    total_energy = math.fsum(
        float(value)
        for value in masses * (np.asarray(state["internal_energy"]) + kinetic)
    )
    angular_momentum_z = math.fsum(
        float(value)
        for value in masses
        * (
            (np.asarray(state["coordinates"])[:, 0] - CENTER[0]) * velocities[:, 1]
            - (np.asarray(state["coordinates"])[:, 1] - CENTER[1]) * velocities[:, 0]
        )
    )
    return {
        "mass": total_mass,
        "momentum": momentum,
        "energy": total_energy,
        "angular_momentum_z": angular_momentum_z,
    }


def reduced_rows(state: dict[str, Any], bins: int = 64) -> list[dict[str, Any]]:
    """Return deterministic mass-weighted radial-bin rows."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    radius = np.asarray(state["radius"])
    maximum_radius = math.sqrt(0.5)
    indices = np.minimum((radius / maximum_radius * bins).astype(int), bins - 1)
    fields = (
        "density",
        "pressure",
        "radial_velocity",
        "tangential_velocity",
        "analytic_density",
        "analytic_pressure",
        "analytic_tangential_velocity",
    )
    rows: list[dict[str, Any]] = []
    masses = np.asarray(state["masses"])
    for index in range(bins):
        selected = indices == index
        count = int(np.count_nonzero(selected))
        if count == 0:
            continue
        selected_mass = masses[selected]
        mass_sum = math.fsum(float(value) for value in selected_mass)
        row: dict[str, Any] = {
            "bin": index,
            "radius_min": index * maximum_radius / bins,
            "radius_max": (index + 1) * maximum_radius / bins,
            "particle_count": count,
            "mass": mass_sum,
        }
        for field in fields:
            row[field] = (
                math.fsum(
                    float(value)
                    for value in selected_mass * np.asarray(state[field])[selected]
                )
                / mass_sum
            )
        row["radius"] = (
            math.fsum(float(value) for value in selected_mass * radius[selected])
            / mass_sum
        )
        rows.append(row)
    return rows


def write_reduction(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write deterministic CSV or gzip-compressed CSV."""
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write an empty reduction")
    fields = list(rows[0])

    def write_rows(text_stream: TextIO) -> None:
        writer = csv.DictWriter(text_stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    if path.suffix == ".gz":
        with (
            path.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as binary,
            io.TextIOWrapper(binary, encoding="utf-8", newline="") as text,
        ):
            write_rows(text)
    else:
        with path.open("w", encoding="utf-8", newline="") as text:
            write_rows(text)


def parse_limits(values: Iterable[str]) -> dict[str, float]:
    limits: dict[str, float] = {}
    for value in values:
        try:
            name, raw_limit = value.split("=", 1)
            limit = float(raw_limit)
        except ValueError as error:
            raise ValueError(
                f"invalid metric limit {value!r}; expected NAME=LIMIT"
            ) from error
        if name not in METRIC_NAMES:
            raise ValueError(f"unknown metric {name!r}")
        if name in limits:
            raise ValueError(f"duplicate metric limit {name!r}")
        if not math.isfinite(limit) or limit < 0.0:
            raise ValueError(
                f"metric limit for {name!r} must be finite and nonnegative"
            )
        limits[name] = limit
    return limits


def enforce_limits(metrics: dict[str, float], limits: dict[str, float]) -> None:
    failures = [
        f"{name}={metrics[name]:.17g} exceeds {limit:.17g}"
        for name, limit in limits.items()
        if metrics[name] > limit
    ]
    if failures:
        raise SnapshotError("; ".join(failures))


def check_official_initial_condition(state: dict[str, Any]) -> None:
    """Enforce byte-fixture semantic identity without requiring its row order."""
    expected_ids = np.arange(OFFICIAL_PARTICLE_COUNT, dtype=np.uint64)
    if float(state["time"]) != 0.0:
        raise SnapshotError("official initial condition must have header time zero")
    if not np.array_equal(state["ids"], expected_ids):
        raise SnapshotError("official initial condition IDs must be exactly 0..4091")
    if not np.array_equal(
        np.asarray(state["density"]), np.ones(OFFICIAL_PARTICLE_COUNT)
    ):
        raise SnapshotError("official initial condition density must be exactly one")
    metrics = comparison_metrics(state)
    enforce_limits(
        metrics,
        {
            "density_l1": 0.0,
            "pressure_l1": 1.0e-6,
            "pressure_normalized_l1": 1.0e-6,
            "specific_internal_energy_normalized_l1": 1.0e-6,
            "radial_velocity_rms": 1.0e-7,
            "tangential_velocity_l1": 1.0e-7,
            "velocity_vector_l1": 1.0e-7,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--dimensional-tolerance", type=float, default=0.0)
    parser.add_argument("--require-time", type=float)
    parser.add_argument("--official-initial-condition", action="store_true")
    parser.add_argument(
        "--maximum", action="append", default=[], metavar="METRIC=LIMIT"
    )
    parser.add_argument("--output-reduced", type=Path)
    parser.add_argument("--bins", type=int, default=64)
    args = parser.parse_args()

    try:
        state = read_snapshot(
            args.snapshot,
            gamma=args.gamma,
            dimensional_tolerance=args.dimensional_tolerance,
        )
        if args.require_time is not None and float(state["time"]) != args.require_time:
            raise SnapshotError(
                f"Header/Time is {state['time']}, expected {args.require_time}"
            )
        if args.official_initial_condition:
            check_official_initial_condition(state)
        measured = comparison_metrics(state)
        enforce_limits(measured, parse_limits(args.maximum))
        if args.output_reduced is not None:
            write_reduction(args.output_reduced, reduced_rows(state, args.bins))
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "snapshot": str(args.snapshot),
                "time": state["time"],
                "particle_count": len(state["ids"]),
                "metrics": measured,
                "diagnostics": global_diagnostics(state),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
