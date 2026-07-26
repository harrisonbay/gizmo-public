#!/usr/bin/env python3
"""Measure the McNally et al. Kelvin--Helmholtz benchmark diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py
except ModuleNotFoundError:
    h5py = None  # type: ignore[assignment]


BOX_SIZE = 1.0
GAMMA = 5.0 / 3.0
PRESSURE = 2.5
SMOOTHING_LENGTH = 0.025
PERTURBATION_AMPLITUDE = 0.01
OFFICIAL_PARTICLE_COUNT = 66_868
REFERENCE_TIME_MIN = 0.0
REFERENCE_TIME_MAX = 1.5


class SnapshotError(ValueError):
    """The input is not a valid two-dimensional KH snapshot."""


class ReferenceError(ValueError):
    """A published reference table is malformed or used outside its interval."""


def _as_vector(name: str, values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite one-dimensional array")
    return result


def mode_amplitude(
    x: np.ndarray,
    y: np.ndarray,
    velocity_y: np.ndarray,
    quadrature_volume: np.ndarray,
) -> float:
    """Evaluate McNally et al. equations 14--17 with point volumes."""
    x = _as_vector("x", x)
    y = _as_vector("y", y)
    velocity_y = _as_vector("velocity_y", velocity_y)
    volume = _as_vector("quadrature_volume", quadrature_volume)
    if not (x.shape == y.shape == velocity_y.shape == volume.shape):
        raise ValueError("mode-amplitude arrays must have identical shapes")
    if np.any((x < 0.0) | (x >= BOX_SIZE) | (y < 0.0) | (y >= BOX_SIZE)):
        raise ValueError("x and y must lie in the periodic unit box [0,1)")
    if np.any(volume <= 0.0):
        raise ValueError("quadrature volumes must be positive")

    interface_distance = np.where(
        y < 0.5, np.abs(y - 0.25), np.abs((1.0 - y) - 0.25)
    )
    envelope = np.exp(-4.0 * math.pi * interface_distance)
    denominator = math.fsum(float(value) for value in volume * envelope)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("mode-amplitude denominator must be finite and positive")

    phase = 4.0 * math.pi * x
    sine_sum = math.fsum(
        float(value)
        for value in velocity_y * volume * np.sin(phase) * envelope
    )
    cosine_sum = math.fsum(
        float(value)
        for value in velocity_y * volume * np.cos(phase) * envelope
    )
    return 2.0 * math.hypot(sine_sum / denominator, cosine_sum / denominator)


def maximum_vertical_kinetic_energy_density(
    density: np.ndarray, velocity_y: np.ndarray
) -> float:
    """Return max(0.5 rho v_y^2), the paper's noise-sensitive diagnostic."""
    density = _as_vector("density", density)
    velocity_y = _as_vector("velocity_y", velocity_y)
    if density.shape != velocity_y.shape:
        raise ValueError("density and velocity_y must have identical shapes")
    if np.any(density <= 0.0):
        raise ValueError("density must be positive")
    return float(np.max(0.5 * density * velocity_y**2))


def gizmo_phase_initial_fields(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the hosted IC's fields, a half-box y-translation of the paper."""
    x = _as_vector("x", x)
    y = _as_vector("y", y)
    if x.shape != y.shape:
        raise ValueError("x and y must have identical shapes")
    if np.any((x < 0.0) | (x >= BOX_SIZE) | (y < 0.0) | (y >= BOX_SIZE)):
        raise ValueError("x and y must lie in the periodic unit box [0,1)")

    density = np.empty_like(y)
    velocity_x = np.empty_like(y)
    regions = (
        y < 0.25,
        (y >= 0.25) & (y < 0.5),
        (y >= 0.5) & (y < 0.75),
        y >= 0.75,
    )
    exponentials = (
        np.exp((y - 0.25) / SMOOTHING_LENGTH),
        np.exp((-y + 0.25) / SMOOTHING_LENGTH),
        np.exp(-(0.75 - y) / SMOOTHING_LENGTH),
        np.exp(-(y - 0.75) / SMOOTHING_LENGTH),
    )
    # The hosted file has (rho_1,U_1)=(2,-0.5) outside and
    # (rho_2,U_2)=(1,+0.5) inside: the paper's state shifted by y=0.5.
    for index, (region, transition) in enumerate(
        zip(regions, exponentials, strict=True)
    ):
        if index in (0, 3):
            density[region] = 2.0 - 0.5 * transition[region]
            velocity_x[region] = -0.5 + 0.5 * transition[region]
        else:
            density[region] = 1.0 + 0.5 * transition[region]
            velocity_x[region] = 0.5 - 0.5 * transition[region]

    velocity_y = PERTURBATION_AMPLITUDE * np.sin(4.0 * math.pi * x)
    internal_energy = PRESSURE / ((GAMMA - 1.0) * density)
    return density, velocity_x, velocity_y, internal_energy


def _read_array(
    group: "h5py.Group", name: str, expected_shape: tuple[int, ...]
) -> np.ndarray:
    if name not in group:
        raise SnapshotError(f"missing PartType0/{name}")
    values = np.asarray(group[name])
    if values.shape != expected_shape:
        raise SnapshotError(
            f"PartType0/{name} has shape {values.shape}, expected {expected_shape}"
        )
    result = values.astype(np.float64)
    if not np.all(np.isfinite(result)):
        raise SnapshotError(f"PartType0/{name} contains non-finite values")
    return result


def read_snapshot(path: Path) -> dict[str, Any]:
    """Read and validate a GIZMO-format, strictly two-dimensional gas snapshot."""
    if h5py is None:
        raise SnapshotError(
            "h5py is required; run with `uv run --with h5py --with numpy`"
        )
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
        particle_ids = np.asarray(group["ParticleIDs"])
        if particle_ids.ndim != 1 or particle_ids.size == 0:
            raise SnapshotError("ParticleIDs must be a nonempty one-dimensional array")
        if not np.issubdtype(particle_ids.dtype, np.integer):
            raise SnapshotError("ParticleIDs must use an integer HDF5 datatype")
        count = particle_ids.size
        if len(np.unique(particle_ids)) != count:
            raise SnapshotError("ParticleIDs must be unique")
        coordinates = _read_array(group, "Coordinates", (count, 3))
        velocities = _read_array(group, "Velocities", (count, 3))
        masses = _read_array(group, "Masses", (count,))
        density = _read_array(group, "Density", (count,))
        internal_energy = _read_array(group, "InternalEnergy", (count,))
        smoothing_length = _read_array(group, "SmoothingLength", (count,))
        if np.any((coordinates[:, :2] < 0.0) | (coordinates[:, :2] >= BOX_SIZE)):
            raise SnapshotError("x and y coordinates must lie in [0,1)")
        if np.any(coordinates[:, 2] != 0.0) or np.any(velocities[:, 2] != 0.0):
            raise SnapshotError("snapshot must be strictly two-dimensional")
        if np.any(masses <= 0.0) or np.any(density <= 0.0):
            raise SnapshotError("masses and density must be positive")
        if np.any(internal_energy <= 0.0):
            raise SnapshotError("internal energy must be positive")
        if np.any(smoothing_length <= 0.0):
            raise SnapshotError("smoothing length must be positive")

    order = np.argsort(particle_ids)
    return {
        "time": time,
        "particle_ids": particle_ids[order],
        "coordinates": coordinates[order],
        "velocities": velocities[order],
        "masses": masses[order],
        "density": density[order],
        "internal_energy": internal_energy[order],
        "smoothing_length": smoothing_length[order],
    }


def snapshot_diagnostics(state: dict[str, Any]) -> dict[str, float | int]:
    """Calculate the two published diagnostics using volume m/rho."""
    coordinates = np.asarray(state["coordinates"])
    velocities = np.asarray(state["velocities"])
    masses = np.asarray(state["masses"])
    density = np.asarray(state["density"])
    volume = masses / density
    return {
        "particle_count": int(density.size),
        "quadrature_volume_sum": math.fsum(float(value) for value in volume),
        "mode_amplitude": mode_amplitude(
            coordinates[:, 0], coordinates[:, 1], velocities[:, 1], volume
        ),
        "maximum_vertical_kinetic_energy_density": (
            maximum_vertical_kinetic_energy_density(density, velocities[:, 1])
        ),
    }


def official_initial_condition_errors(state: dict[str, Any]) -> dict[str, float]:
    """Check schema and return max errors against the hosted GIZMO phase."""
    ids = np.asarray(state["particle_ids"])
    if ids.size != OFFICIAL_PARTICLE_COUNT:
        raise SnapshotError(
            f"official IC has {ids.size} particles, expected {OFFICIAL_PARTICLE_COUNT}"
        )
    if not np.array_equal(ids, np.arange(1, OFFICIAL_PARTICLE_COUNT + 1)):
        raise SnapshotError("official IC IDs must be the permutation 1..66868")
    if float(state["time"]) != 0.0:
        raise SnapshotError("official IC Header/Time must be zero")

    coordinates = np.asarray(state["coordinates"])
    velocities = np.asarray(state["velocities"])
    density = np.asarray(state["density"])
    internal_energy = np.asarray(state["internal_energy"])
    expected = gizmo_phase_initial_fields(coordinates[:, 0], coordinates[:, 1])
    pressure = (GAMMA - 1.0) * density * internal_energy
    return {
        "density_max_abs_error": float(np.max(np.abs(density - expected[0]))),
        "velocity_x_max_abs_error": float(
            np.max(np.abs(velocities[:, 0] - expected[1]))
        ),
        "velocity_y_max_abs_error": float(
            np.max(np.abs(velocities[:, 1] - expected[2]))
        ),
        "internal_energy_max_abs_error": float(
            np.max(np.abs(internal_energy - expected[3]))
        ),
        "pressure_max_abs_error": float(np.max(np.abs(pressure - PRESSURE))),
    }


def load_reference_table(path: Path, columns: int) -> np.ndarray:
    """Load one strict, finite, monotonically-timed public ASCII table."""
    rows: list[list[float]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != columns:
            raise ReferenceError(
                f"{path}:{line_number}: expected {columns} columns, got {len(fields)}"
            )
        try:
            row = [float(field) for field in fields]
        except ValueError as error:
            raise ReferenceError(f"{path}:{line_number}: invalid number") from error
        if not all(math.isfinite(value) for value in row):
            raise ReferenceError(f"{path}:{line_number}: non-finite value")
        rows.append(row)
    result = np.asarray(rows, dtype=np.float64)
    if result.shape != (76, columns):
        raise ReferenceError(
            f"{path}: expected 76 data rows, found {result.shape[0]}"
        )
    if not np.all(np.diff(result[:, 0]) > 0.0):
        raise ReferenceError(f"{path}: times must increase strictly")
    if not math.isclose(result[0, 0], REFERENCE_TIME_MIN, abs_tol=1e-14):
        raise ReferenceError(f"{path}: first time must be zero")
    if not math.isclose(result[-1, 0], REFERENCE_TIME_MAX, abs_tol=1e-14):
        raise ReferenceError(f"{path}: final time must be 1.5")
    if np.any(result[:, 1:] < 0.0):
        raise ReferenceError(f"{path}: diagnostic values must be nonnegative")
    return result


def load_public_references(directory: Path) -> dict[str, np.ndarray]:
    """Load both independently sampled public curves."""
    return {
        "mode": load_reference_table(directory / "khmode_rev0.txt", 3),
        "energy": load_reference_table(directory / "khener_rev1.txt", 2),
    }


def reference_at_time(
    time: float, references: dict[str, np.ndarray]
) -> dict[str, float]:
    """Interpolate inside [0,1.5], refusing all reference extrapolation."""
    if not math.isfinite(time):
        raise ReferenceError("snapshot time must be finite")
    if time < REFERENCE_TIME_MIN or time > REFERENCE_TIME_MAX:
        raise ReferenceError(
            f"time {time} is outside the published reference interval [0,1.5]"
        )
    mode = references["mode"]
    energy = references["energy"]
    return {
        "mode_amplitude": float(np.interp(time, mode[:, 0], mode[:, 1])),
        "mode_gci_absolute_uncertainty": float(
            np.interp(time, mode[:, 0], mode[:, 2])
        ),
        "maximum_vertical_kinetic_energy_density": float(
            np.interp(time, energy[:, 0], energy[:, 1])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument(
        "--reference-directory",
        type=Path,
        default=Path(__file__).resolve().parent / "kh_mcnally",
    )
    parser.add_argument("--official-initial-condition", action="store_true")
    parser.add_argument("--require-time", type=float)
    args = parser.parse_args()

    state = read_snapshot(args.snapshot)
    if args.require_time is not None and not math.isclose(
        float(state["time"]), args.require_time, rel_tol=0.0, abs_tol=1e-12
    ):
        raise SnapshotError(
            f"snapshot time is {state['time']}, expected {args.require_time}"
        )
    result: dict[str, Any] = {
        "snapshot": str(args.snapshot),
        "time": float(state["time"]),
        "diagnostics": snapshot_diagnostics(state),
    }
    if args.official_initial_condition:
        result["official_initial_condition_errors"] = (
            official_initial_condition_errors(state)
        )
    time = float(state["time"])
    result["published_reference_interval"] = [
        REFERENCE_TIME_MIN,
        REFERENCE_TIME_MAX,
    ]
    result["within_published_reference_interval"] = (
        REFERENCE_TIME_MIN <= time <= REFERENCE_TIME_MAX
    )
    if result["within_published_reference_interval"]:
        reference = reference_at_time(
            time, load_public_references(args.reference_directory)
        )
        result["published_reference"] = reference
        diagnostics = result["diagnostics"]
        result["difference_from_published_reference"] = {
            "mode_amplitude": (
                diagnostics["mode_amplitude"] - reference["mode_amplitude"]
            ),
            "maximum_vertical_kinetic_energy_density": (
                diagnostics["maximum_vertical_kinetic_energy_density"]
                - reference["maximum_vertical_kinetic_energy_density"]
            ),
        }
    else:
        result["published_reference"] = None
        result["difference_from_published_reference"] = None
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
