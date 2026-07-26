#!/usr/bin/env python3
"""Write a deterministic, ID-sorted semantic table from an MHD snapshot."""

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
    "x",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "density",
    "specific_internal_energy",
    "pressure",
    "smoothing_length",
    "mass",
    "magnetic_x",
    "magnetic_y",
    "magnetic_z",
)


def reduce(snapshot: Path, output: Path) -> float:
    with h5py.File(snapshot, "r") as handle:
        time = float(handle["Header"].attrs["Time"])
        gas = handle["PartType0"]
        particle_ids = np.asarray(gas["ParticleIDs"], dtype=np.uint64)
        order = np.argsort(particle_ids, kind="stable")
        coordinates = np.asarray(gas["Coordinates"], dtype=np.float64)
        velocities = np.asarray(gas["Velocities"], dtype=np.float64)
        magnetic = np.asarray(gas["MagneticField"], dtype=np.float64)
        density = np.asarray(gas["Density"], dtype=np.float64)
        internal_energy = np.asarray(gas["InternalEnergy"], dtype=np.float64)
        columns = (
            particle_ids,
            coordinates[:, 0],
            velocities[:, 0],
            velocities[:, 1],
            velocities[:, 2],
            density,
            internal_energy,
            (2.0 / 3.0) * density * internal_energy,
            np.asarray(gas["SmoothingLength"], dtype=np.float64),
            np.asarray(gas["Masses"], dtype=np.float64),
            magnetic[:, 0],
            magnetic[:, 1],
            magnetic[:, 2],
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.writer(text, lineterminator="\n")
                writer.writerow(FIELDS)
                for index in order:
                    writer.writerow(
                        [str(int(columns[0][index]))]
                        + [format(float(column[index]), ".17g") for column in columns[1:]]
                    )
    return time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    time = reduce(args.snapshot, args.output)
    print(f"{args.output}: time={time:.17g}")


if __name__ == "__main__":
    main()
