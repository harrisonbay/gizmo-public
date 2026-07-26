#!/usr/bin/env python3
"""Write a deterministic, ID-sorted gas-and-grain table from a GIZMO snapshot."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
from pathlib import Path

import h5py
import numpy as np


FIELDS = (
    "particle_id",
    "particle_type",
    "x",
    "velocity_x",
    "mass",
    "smoothing_length",
    "density",
    "specific_internal_energy",
    "grain_size",
)


def _format(value: float) -> str:
    return format(float(value), ".17g")


def reduce(snapshot: Path, output: Path) -> float:
    rows: list[tuple[int, list[str]]] = []
    with h5py.File(snapshot, "r") as handle:
        time = float(handle["Header"].attrs["Time"])
        for particle_type in (0, 3):
            group = handle[f"PartType{particle_type}"]
            ids = np.asarray(group["ParticleIDs"], dtype=np.uint64)
            coordinates = np.asarray(group["Coordinates"], dtype=np.float64)
            velocities = np.asarray(group["Velocities"], dtype=np.float64)
            masses = np.asarray(group["Masses"], dtype=np.float64)
            smoothing_lengths = np.asarray(
                group["SmoothingLength"], dtype=np.float64
            )
            density = (
                np.asarray(group["Density"], dtype=np.float64)
                if particle_type == 0
                else None
            )
            internal_energy = (
                np.asarray(group["InternalEnergy"], dtype=np.float64)
                if particle_type == 0
                else None
            )
            grain_size = (
                np.asarray(group["GrainSize"], dtype=np.float64)
                if particle_type == 3
                else None
            )
            for index, particle_id in enumerate(ids):
                rows.append(
                    (
                        int(particle_id),
                        [
                            str(int(particle_id)),
                            str(particle_type),
                            _format(coordinates[index, 0]),
                            _format(velocities[index, 0]),
                            _format(masses[index]),
                            _format(smoothing_lengths[index]),
                            "" if density is None else _format(density[index]),
                            (
                                ""
                                if internal_energy is None
                                else _format(internal_energy[index])
                            ),
                            "" if grain_size is None else _format(grain_size[index]),
                        ],
                    )
                )
    rows.sort(key=lambda row: row[0])

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.writer(text, lineterminator="\n")
                writer.writerow(FIELDS)
                writer.writerows(row for _, row in rows)
    return time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    time = reduce(args.snapshot, args.output)
    print(f"{args.output}: time={time:.17g}")


if __name__ == "__main__":
    main()
