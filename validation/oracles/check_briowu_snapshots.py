#!/usr/bin/env python3
"""Validate Brio-Wu snapshot state and report ID-aligned global diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np


EXPECTED_TIMES = (0.0, 0.1, 0.2)
EXPECTED_PARTICLES = 50_176
DOMAIN = np.array([4.0, 0.25, 0.25], dtype=np.float64)


def _array(group: h5py.Group, name: str, shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(group[name])
    if values.shape != shape:
        raise ValueError(f"{name} has shape {values.shape}, expected {shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    return values


def read_snapshot(path: Path, expected_time: float) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        time = float(handle["Header"].attrs["Time"])
        if time != expected_time:
            raise ValueError(
                f"{path}: Header/Time={time:.17g}, expected {expected_time:.17g}"
            )
        gas = handle["PartType0"]
        count = len(gas["ParticleIDs"])
        if count != EXPECTED_PARTICLES:
            raise ValueError(
                f"{path}: particle count {count}, expected {EXPECTED_PARTICLES}"
            )
        coordinates = _array(gas, "Coordinates", (count, 3)).astype(
            np.float64, copy=False
        )
        velocities = _array(gas, "Velocities", (count, 3)).astype(
            np.float64, copy=False
        )
        density = _array(gas, "Density", (count,)).astype(np.float64, copy=False)
        internal = _array(gas, "InternalEnergy", (count,)).astype(
            np.float64, copy=False
        )
        magnetic = _array(gas, "MagneticField", (count, 3)).astype(
            np.float64, copy=False
        )
        smoothing = _array(gas, "SmoothingLength", (count,)).astype(
            np.float64, copy=False
        )
        masses = _array(gas, "Masses", (count,)).astype(np.float64, copy=False)
        particle_ids = _array(gas, "ParticleIDs", (count,)).astype(
            np.uint64, copy=False
        )
        optional_fields = {}
        for name in (
            "DivBcleaningFunctionPhi",
            "DivergenceOfMagneticField",
        ):
            if name in gas:
                optional_fields[name] = _array(gas, name, (count,)).astype(
                    np.float64, copy=False
                )

    order = np.argsort(particle_ids, kind="stable")
    sorted_ids = particle_ids[order]
    if np.any(sorted_ids[1:] == sorted_ids[:-1]):
        raise ValueError(f"{path}: duplicate ParticleIDs")
    if not np.array_equal(sorted_ids, np.arange(count, dtype=np.uint64)):
        raise ValueError(f"{path}: ParticleIDs are not exactly 0..{count - 1}")
    if np.any(coordinates < 0.0) or np.any(coordinates >= DOMAIN):
        raise ValueError(f"{path}: coordinate outside rectangular periodic domain")
    if np.any(density <= 0.0):
        raise ValueError(f"{path}: non-positive density")
    if np.any(internal < 0.0):
        raise ValueError(f"{path}: negative specific internal energy")
    if np.any(smoothing <= 0.0) or np.any(masses <= 0.0):
        raise ValueError(f"{path}: non-positive smoothing length or mass")
    for name, values in (
        ("coordinate_z", coordinates[:, 2]),
        ("velocity_z", velocities[:, 2]),
        ("magnetic_z", magnetic[:, 2]),
    ):
        if np.any(values != 0.0):
            raise ValueError(f"{path}: {name} is not exactly zero")

    sorted_masses = masses[order]
    sorted_velocities = velocities[order]
    sorted_density = density[order]
    sorted_internal = internal[order]
    sorted_magnetic = magnetic[order]
    mass = math.fsum(float(value) for value in sorted_masses)
    momentum = [
        math.fsum(
            float(particle_mass * velocity)
            for particle_mass, velocity in zip(
                sorted_masses, sorted_velocities[:, component]
            )
        )
        for component in range(3)
    ]
    kinetic = math.fsum(
        float(0.5 * particle_mass * np.dot(velocity, velocity))
        for particle_mass, velocity in zip(sorted_masses, sorted_velocities)
    )
    thermal = math.fsum(
        float(particle_mass * specific)
        for particle_mass, specific in zip(sorted_masses, sorted_internal)
    )
    magnetic_energy = math.fsum(
        float(
            0.5
            * (particle_mass / rho)
            * np.dot(magnetic_field, magnetic_field)
        )
        for particle_mass, rho, magnetic_field in zip(
            sorted_masses, sorted_density, sorted_magnetic
        )
    )
    return {
        "path": str(path),
        "time": time,
        "particle_count": count,
        "particle_ids": sorted_ids,
        "masses": sorted_masses,
        "mass": mass,
        "momentum": momentum,
        "kinetic_energy": kinetic,
        "thermal_energy": thermal,
        "magnetic_energy": magnetic_energy,
        "diagnostic_total_energy": kinetic + thermal + magnetic_energy,
        "density_range": [float(np.min(density)), float(np.max(density))],
        "specific_internal_energy_range": [
            float(np.min(internal)),
            float(np.max(internal)),
        ],
        "optional_fields": sorted(optional_fields),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshots", type=Path, nargs=3)
    parser.add_argument("--maximum-relative-mass-change", type=float)
    parser.add_argument("--maximum-absolute-momentum", type=float)
    parser.add_argument("--maximum-relative-diagnostic-energy-change", type=float)
    args = parser.parse_args()
    for name, value in (
        ("maximum relative mass change", args.maximum_relative_mass_change),
        ("maximum absolute momentum", args.maximum_absolute_momentum),
        (
            "maximum relative diagnostic energy change",
            args.maximum_relative_diagnostic_energy_change,
        ),
    ):
        if value is not None and (not math.isfinite(value) or value < 0.0):
            raise ValueError(f"{name} must be nonnegative and finite")

    states = [
        read_snapshot(path, expected_time)
        for path, expected_time in zip(args.snapshots, EXPECTED_TIMES)
    ]
    first_ids = states[0].pop("particle_ids")
    first_masses = states[0].pop("masses")
    for state in states[1:]:
        if not np.array_equal(state.pop("particle_ids"), first_ids):
            raise ValueError("particle IDs changed between snapshots")
        if not np.array_equal(state.pop("masses"), first_masses):
            raise ValueError("per-ID masses changed between snapshots")

    initial_mass = states[0]["mass"]
    initial_energy = states[0]["diagnostic_total_energy"]
    failures = []
    for state in states:
        state["relative_mass_change"] = state["mass"] / initial_mass - 1.0
        state["relative_diagnostic_total_energy_change"] = (
            state["diagnostic_total_energy"] / initial_energy - 1.0
        )
        if (
            args.maximum_relative_mass_change is not None
            and abs(state["relative_mass_change"])
            > args.maximum_relative_mass_change
        ):
            failures.append(
                f"t={state['time']}: relative mass change "
                f"{state['relative_mass_change']:.8g} exceeds "
                f"{args.maximum_relative_mass_change:.8g}"
            )
        maximum_momentum = max(abs(value) for value in state["momentum"])
        if (
            args.maximum_absolute_momentum is not None
            and maximum_momentum > args.maximum_absolute_momentum
        ):
            failures.append(
                f"t={state['time']}: absolute momentum {maximum_momentum:.8g} "
                f"exceeds {args.maximum_absolute_momentum:.8g}"
            )
        if (
            args.maximum_relative_diagnostic_energy_change is not None
            and abs(state["relative_diagnostic_total_energy_change"])
            > args.maximum_relative_diagnostic_energy_change
        ):
            failures.append(
                f"t={state['time']}: relative diagnostic energy change "
                f"{state['relative_diagnostic_total_energy_change']:.8g} exceeds "
                f"{args.maximum_relative_diagnostic_energy_change:.8g}"
            )

    print(json.dumps({"snapshots": states}, indent=2, sort_keys=True))
    for failure in failures:
        print(f"FAIL: {failure}")
    return int(bool(failures))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"error: {error}") from error
