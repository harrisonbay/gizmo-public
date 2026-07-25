#!/usr/bin/env python3
"""Check a GIZMO sound-wave snapshot against the linear analytic solution."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

from generate_soundwave_ic import AMPLITUDE, GAMMA


def load_state(
    snapshot: Path,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(snapshot, "r") as handle:
        time = float(handle["Header"].attrs["Time"])
        gas = handle["PartType0"]
        x = np.asarray(gas["Coordinates"])[:, 0]
        velocity = np.asarray(gas["Velocities"])[:, 0]
        density = np.asarray(gas["Density"])
        internal_energy = np.asarray(gas["InternalEnergy"])
        masses = np.asarray(gas["Masses"])
    return time, x, velocity, density, internal_energy, masses


def analyze(snapshot: Path, reference: Path, amplitude: float) -> dict[str, float | str]:
    time, x, velocity, density, internal_energy, masses = load_state(snapshot)
    _, _, reference_velocity, _, reference_internal_energy, reference_masses = load_state(
        reference
    )
    expected = 1.0 + amplitude * np.sin(2.0 * math.pi * (x - time))
    error = density - expected
    relative_l1 = float(np.mean(np.abs(error)) / amplitude)
    expected_velocity = amplitude * np.sin(2.0 * math.pi * (x - time))
    relative_l1_velocity = float(
        np.mean(np.abs(velocity - expected_velocity)) / amplitude
    )
    pressure = (GAMMA - 1.0) * density * internal_energy
    expected_pressure = 1.0 / GAMMA + amplitude * np.sin(
        2.0 * math.pi * (x - time)
    )
    relative_l1_pressure = float(
        np.mean(np.abs(pressure - expected_pressure)) / amplitude
    )

    centered = density - np.mean(density)
    sin_basis = np.sin(2.0 * math.pi * (x - time))
    cos_basis = np.cos(2.0 * math.pi * (x - time))
    sine = 2.0 * float(np.mean(centered * sin_basis))
    cosine = 2.0 * float(np.mean(centered * cos_basis))
    fitted_amplitude = math.hypot(sine, cosine)
    amplitude_ratio = fitted_amplitude / amplitude
    phase_error = abs(math.atan2(cosine, sine))

    mass = float(np.sum(masses))
    reference_mass = float(np.sum(reference_masses))
    momentum = float(np.sum(masses * velocity))
    reference_momentum = float(np.sum(reference_masses * reference_velocity))
    energy = float(np.sum(masses * (internal_energy + 0.5 * velocity * velocity)))
    reference_energy = float(
        np.sum(
            reference_masses
            * (reference_internal_energy + 0.5 * reference_velocity * reference_velocity)
        )
    )

    return {
        "snapshot": str(snapshot),
        "reference": str(reference),
        "time": time,
        "relative_l1_density_error": relative_l1,
        "relative_l1_velocity_error": relative_l1_velocity,
        "relative_l1_pressure_error": relative_l1_pressure,
        "amplitude_ratio": amplitude_ratio,
        "phase_error_radians": phase_error,
        "relative_mass_drift": abs(mass - reference_mass) / reference_mass,
        "scaled_momentum_drift": abs(momentum - reference_momentum)
        / (reference_mass * amplitude),
        "relative_energy_drift": abs(energy - reference_energy) / reference_energy,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--amplitude", type=float, default=AMPLITUDE)
    parser.add_argument("--max-relative-l1", type=float, default=0.06)
    parser.add_argument("--max-relative-l1-velocity", type=float, default=0.06)
    parser.add_argument("--max-relative-l1-pressure", type=float, default=0.06)
    parser.add_argument("--min-amplitude-ratio", type=float, default=0.99)
    parser.add_argument("--max-amplitude-ratio", type=float, default=1.01)
    parser.add_argument("--max-phase-error", type=float, default=0.01)
    parser.add_argument("--max-relative-mass-drift", type=float, default=1.0e-12)
    parser.add_argument("--max-scaled-momentum-drift", type=float, default=1.0e-10)
    parser.add_argument("--max-relative-energy-drift", type=float, default=1.0e-8)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    metrics = analyze(args.snapshot, args.reference, args.amplitude)
    encoded = json.dumps(metrics, indent=2, sort_keys=True)
    print(encoded)
    if args.json_output is not None:
        args.json_output.write_text(f"{encoded}\n", encoding="utf-8")
    passed = (
        metrics["relative_l1_density_error"] <= args.max_relative_l1
        and metrics["relative_l1_velocity_error"] <= args.max_relative_l1_velocity
        and metrics["relative_l1_pressure_error"] <= args.max_relative_l1_pressure
        and args.min_amplitude_ratio <= metrics["amplitude_ratio"] <= args.max_amplitude_ratio
        and metrics["phase_error_radians"] <= args.max_phase_error
        and metrics["relative_mass_drift"] <= args.max_relative_mass_drift
        and metrics["scaled_momentum_drift"] <= args.max_scaled_momentum_drift
        and metrics["relative_energy_drift"] <= args.max_relative_energy_drift
    )
    if not passed:
        raise SystemExit("sound-wave oracle thresholds failed")


if __name__ == "__main__":
    main()
