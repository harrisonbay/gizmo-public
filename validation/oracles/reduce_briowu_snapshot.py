#!/usr/bin/env python3
"""Reduce the public 2-D Brio-Wu sheet to deterministic one-dimensional profiles."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
from pathlib import Path

import h5py
import numpy as np


PROFILE_COLUMNS = (
    "rank_bin",
    "particle_count",
    "x_mean",
    "x_min",
    "x_max",
    "density_mean",
    "density_min",
    "density_max",
    "velocity_x_mean",
    "velocity_y_mean",
    "velocity_z_mean",
    "pressure_mean",
    "specific_internal_energy_mean",
    "magnetic_x_mean",
    "magnetic_y_mean",
    "magnetic_z_mean",
    "smoothing_length_mean",
    "mass_sum",
)


def _column(gas: h5py.Group, name: str) -> np.ndarray:
    return np.asarray(gas[name], dtype=np.float64)


def reduce(snapshot: Path, output: Path, columns: int, gamma: float) -> tuple[float, int]:
    if columns <= 0:
        raise ValueError("columns must be positive")
    if not np.isfinite(gamma) or gamma <= 1.0:
        raise ValueError("gamma must be finite and greater than one")
    with h5py.File(snapshot, "r") as handle:
        time = float(handle["Header"].attrs["Time"])
        gas = handle["PartType0"]
        coordinates = _column(gas, "Coordinates")
        velocities = _column(gas, "Velocities")
        density = _column(gas, "Density")
        internal_energy = _column(gas, "InternalEnergy")
        magnetic = _column(gas, "MagneticField")
        smoothing_length = _column(gas, "SmoothingLength")
        mass = _column(gas, "Masses")
        particle_ids = np.asarray(gas["ParticleIDs"], dtype=np.uint64)

    particle_count = len(density)
    if particle_count == 0 or particle_count % columns:
        raise ValueError(
            f"{particle_count} particles cannot be split into {columns} equal rank bins"
        )
    expected_shapes = {
        "Coordinates": (particle_count, 3),
        "Velocities": (particle_count, 3),
        "Density": (particle_count,),
        "InternalEnergy": (particle_count,),
        "MagneticField": (particle_count, 3),
        "SmoothingLength": (particle_count,),
        "Masses": (particle_count,),
        "ParticleIDs": (particle_count,),
    }
    arrays = {
        "Coordinates": coordinates,
        "Velocities": velocities,
        "Density": density,
        "InternalEnergy": internal_energy,
        "MagneticField": magnetic,
        "SmoothingLength": smoothing_length,
        "Masses": mass,
        "ParticleIDs": particle_ids,
    }
    for name, values in arrays.items():
        if values.shape != expected_shapes[name]:
            raise ValueError(
                f"{name} has shape {values.shape}, expected {expected_shapes[name]}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values")

    # np.lexsort uses its last key as the primary key. Particle ID is the final
    # tie-breaker, making the output independent of HDF5/MPI particle order.
    order = np.lexsort((particle_ids, coordinates[:, 1], coordinates[:, 0]))
    rows_per_column = particle_count // columns
    pressure = (gamma - 1.0) * density * internal_energy

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.writer(text, lineterminator="\n")
                writer.writerow(PROFILE_COLUMNS)
                for bin_index in range(columns):
                    indices = order[
                        bin_index * rows_per_column : (bin_index + 1) * rows_per_column
                    ]
                    values = (
                        bin_index,
                        len(indices),
                        coordinates[indices, 0].mean(),
                        coordinates[indices, 0].min(),
                        coordinates[indices, 0].max(),
                        density[indices].mean(),
                        density[indices].min(),
                        density[indices].max(),
                        velocities[indices, 0].mean(),
                        velocities[indices, 1].mean(),
                        velocities[indices, 2].mean(),
                        pressure[indices].mean(),
                        internal_energy[indices].mean(),
                        magnetic[indices, 0].mean(),
                        magnetic[indices, 1].mean(),
                        magnetic[indices, 2].mean(),
                        smoothing_length[indices].mean(),
                        mass[indices].sum(),
                    )
                    writer.writerow(
                        [str(values[0]), str(values[1])]
                        + [format(float(value), ".17g") for value in values[2:]]
                    )
    return time, particle_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--columns", type=int, default=896)
    parser.add_argument("--gamma", type=float, default=2.0)
    args = parser.parse_args()
    time, particle_count = reduce(args.snapshot, args.output, args.columns, args.gamma)
    print(
        f"{args.output}: time={time:.17g}, particles={particle_count}, "
        f"rank_bins={args.columns}"
    )


if __name__ == "__main__":
    main()
