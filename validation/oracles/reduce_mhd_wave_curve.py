#!/usr/bin/env python3
"""Reduce an MHD-wave snapshot sequence to deterministic analytic L1 metrics."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import math
from pathlib import Path

import h5py
import numpy as np


GAMMA = 5.0 / 3.0
A = 1.0e-6 / math.sqrt(5.0)
BASE_AND_AMPLITUDE = {
    "density": (1.0, A),
    "velocity_x": (0.0, -2.0 * A),
    "velocity_y": (0.0, 2.0 * math.sqrt(2.0) * A / 3.0),
    "velocity_z": (0.0, A / 3.0),
    "pressure": (0.6, A),
    "magnetic_x": (1.0, 0.0),
    "magnetic_y": (math.sqrt(2.0), 4.0 * math.sqrt(2.0) * A / 3.0),
    "magnetic_z": (0.5, 2.0 * A / 3.0),
}
FIELD_NAMES = tuple(BASE_AND_AMPLITUDE)


def snapshot_metrics(path: Path) -> dict[str, float]:
    with h5py.File(path, "r") as handle:
        time = float(handle["Header"].attrs["Time"])
        gas = handle["PartType0"]
        x = np.asarray(gas["Coordinates"], dtype=np.float64)[:, 0]
        mass = np.asarray(gas["Masses"], dtype=np.float64)
        rho = np.asarray(gas["Density"], dtype=np.float64)
        velocity = np.asarray(gas["Velocities"], dtype=np.float64)
        internal_energy = np.asarray(gas["InternalEnergy"], dtype=np.float64)
        magnetic = np.asarray(gas["MagneticField"], dtype=np.float64)
    actual = {
        "density": rho,
        "velocity_x": velocity[:, 0],
        "velocity_y": velocity[:, 1],
        "velocity_z": velocity[:, 2],
        "pressure": (GAMMA - 1.0) * rho * internal_energy,
        "magnetic_x": magnetic[:, 0],
        "magnetic_y": magnetic[:, 1],
        "magnetic_z": magnetic[:, 2],
    }
    wave = np.sin(2.0 * math.pi * (x + 2.0 * time))
    row = {"time": time}
    for name, (base, amplitude) in BASE_AND_AMPLITUDE.items():
        error = actual[name] - (base + amplitude * wave)
        row[f"{name}_l1"] = float(np.mean(np.abs(error)))
        row[f"{name}_rms"] = float(np.sqrt(np.mean(error * error)))
        row[f"{name}_max"] = float(np.max(np.abs(error)))
    volume = mass / rho
    total_energy = mass * (
        internal_energy
        + 0.5 * np.sum(velocity * velocity, axis=1)
        + 0.5 * np.sum(magnetic * magnetic, axis=1) / rho
    )
    row["total_mass"] = float(np.sum(mass))
    row["momentum_x"] = float(np.sum(mass * velocity[:, 0]))
    row["momentum_y"] = float(np.sum(mass * velocity[:, 1]))
    row["momentum_z"] = float(np.sum(mass * velocity[:, 2]))
    row["total_energy"] = float(np.sum(total_energy))
    row["conserved_bx_volume"] = float(np.sum(volume * magnetic[:, 0]))
    row["conserved_by_volume"] = float(np.sum(volume * magnetic[:, 1]))
    row["conserved_bz_volume"] = float(np.sum(volume * magnetic[:, 2]))
    return row


def write_curve(paths: list[Path], output: Path) -> None:
    rows = [snapshot_metrics(path) for path in paths]
    rows.sort(key=lambda row: row["time"])
    if len({row["time"] for row in rows}) != len(rows):
        raise ValueError("snapshot times are not unique")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(
                    text, fieldnames=tuple(rows[0]), lineterminator="\n"
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {key: format(value, ".17g") for key, value in row.items()}
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_curve(args.snapshots, args.output)
    print(f"{args.output}: {len(args.snapshots)} snapshots")


if __name__ == "__main__":
    main()
