#!/usr/bin/env python3
"""Measure Fourier phase, amplitude, and polarization of the public MHD wave."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np


GAMMA = 5.0 / 3.0
WAVENUMBER = 2.0 * math.pi
FAST_SPEED = 2.0
BASE = {
    "density": 1.0,
    "velocity_x": 0.0,
    "velocity_y": 0.0,
    "velocity_z": 0.0,
    "specific_internal_energy": 0.9,
    "pressure": 0.6,
    "magnetic_x": 1.0,
    "magnetic_y": math.sqrt(2.0),
    "magnetic_z": 0.5,
}
EIGENVECTOR = {
    "density": 0.447213595644e-6,
    "velocity_x": -0.894427191000e-6,
    "velocity_y": 0.421637021356e-6,
    "velocity_z": 0.149071198500e-6,
    "specific_internal_energy": 0.268324803201e-6,
    "pressure": 0.447213595644e-6,
    "magnetic_x": 0.0,
    "magnetic_y": 0.843274042711e-6,
    "magnetic_z": 0.298142396999e-6,
}


def fourier(values: np.ndarray, x: np.ndarray, mean: float) -> tuple[float, float]:
    perturbation = values - mean
    sine = 2.0 * float(np.mean(perturbation * np.sin(WAVENUMBER * x)))
    cosine = 2.0 * float(np.mean(perturbation * np.cos(WAVENUMBER * x)))
    return sine, cosine


def measure(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        time = float(handle["Header"].attrs["Time"])
        gas = handle["PartType0"]
        x = np.asarray(gas["Coordinates"], dtype=np.float64)[:, 0]
        velocity = np.asarray(gas["Velocities"], dtype=np.float64)
        density = np.asarray(gas["Density"], dtype=np.float64)
        internal_energy = np.asarray(gas["InternalEnergy"], dtype=np.float64)
        magnetic = np.asarray(gas["MagneticField"], dtype=np.float64)
    fields = {
        "density": density,
        "velocity_x": velocity[:, 0],
        "velocity_y": velocity[:, 1],
        "velocity_z": velocity[:, 2],
        "specific_internal_energy": internal_energy,
        "pressure": (GAMMA - 1.0) * density * internal_energy,
        "magnetic_x": magnetic[:, 0],
        "magnetic_y": magnetic[:, 1],
        "magnetic_z": magnetic[:, 2],
    }
    # The density crest has negative longitudinal velocity, so this fixture is
    # the left-going fast mode: sin(k * (x + c_fast * t)).
    expected_phase = WAVENUMBER * FAST_SPEED * time
    metrics: dict[str, object] = {
        "time": time,
        "expected_phase_radians": math.atan2(
            math.sin(expected_phase), math.cos(expected_phase)
        ),
        "fields": {},
    }
    for name, values in fields.items():
        expected_values = BASE[name] + EIGENVECTOR[name] * np.sin(
            WAVENUMBER * (x + FAST_SPEED * time)
        )
        absolute_error = np.abs(values - expected_values)
        sine, cosine = fourier(values, x, BASE[name])
        amplitude = math.hypot(sine, cosine)
        expected_amplitude = abs(EIGENVECTOR[name])
        phase = math.atan2(cosine, sine)
        if EIGENVECTOR[name] < 0.0:
            phase = math.atan2(-cosine, -sine)
        phase_error = math.atan2(
            math.sin(phase - expected_phase), math.cos(phase - expected_phase)
        )
        metrics["fields"][name] = {
            "sine": sine,
            "cosine": cosine,
            "amplitude": amplitude,
            "expected_amplitude": expected_amplitude,
            "relative_amplitude": (
                amplitude / expected_amplitude if expected_amplitude else None
            ),
            "phase_error_radians": phase_error if expected_amplitude else None,
            "maximum_absolute_perturbation": float(
                np.max(np.abs(values - BASE[name]))
            ),
            "analytic_l1": float(np.mean(absolute_error)),
            "analytic_rms": float(np.sqrt(np.mean(absolute_error * absolute_error))),
            "analytic_max": float(np.max(absolute_error)),
        }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="enforce the pinned 2048-particle analytic L1 and phase gates",
    )
    args = parser.parse_args()
    measured = measure(args.snapshot)
    print(json.dumps(measured, indent=2, sort_keys=True))
    if args.check:
        for name, field in measured["fields"].items():
            l1_limit = 1.0e-8 if name == "magnetic_x" else 3.0e-8
            if field["analytic_l1"] > l1_limit:
                raise SystemExit(
                    f"{name} analytic L1 {field['analytic_l1']:.6g} "
                    f"exceeds {l1_limit:.6g}"
                )
            phase_error = field["phase_error_radians"]
            if phase_error is not None and abs(phase_error) > 0.05:
                raise SystemExit(
                    f"{name} phase error {phase_error:.6g} exceeds 0.05 radians"
                )


if __name__ == "__main__":
    main()
