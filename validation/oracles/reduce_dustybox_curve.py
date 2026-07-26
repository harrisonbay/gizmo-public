#!/usr/bin/env python3
"""Reduce a corrected-C dusty-box snapshot sequence to a compact time curve."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import math
from pathlib import Path

import h5py
import numpy as np

from validation.oracles.compare_dustybox_analytic import analytic_velocities


FIELDS = (
    "snapshot",
    "time",
    "gas_mean_velocity_x",
    "gas_spatial_stddev",
    "grain_mean_velocity_x",
    "grain_spatial_stddev",
    "total_momentum_x",
    "gas_analytic_velocity_x",
    "grain_analytic_velocity_x",
    "gas_absolute_rms",
    "grain_absolute_rms",
)


def _format(value: float) -> str:
    return format(float(value), ".17g")


def reduce(snapshot_directory: Path, output: Path) -> None:
    snapshots = sorted(snapshot_directory.glob("snapshot_*.hdf5"))
    if not snapshots:
        raise ValueError(f"{snapshot_directory}: no snapshots")
    rows: list[list[str]] = []
    for expected_number, snapshot in enumerate(snapshots):
        expected_name = f"snapshot_{expected_number:03}.hdf5"
        if snapshot.name != expected_name:
            raise ValueError(
                f"expected {expected_name}, found {snapshot.name}"
            )
        with h5py.File(snapshot, "r") as handle:
            time = float(handle["Header"].attrs["Time"])
            gas_velocity = np.asarray(
                handle["PartType0"]["Velocities"], dtype=np.float64
            )[:, 0]
            grain_velocity = np.asarray(
                handle["PartType3"]["Velocities"], dtype=np.float64
            )[:, 0]
            gas_mass = np.asarray(
                handle["PartType0"]["Masses"], dtype=np.float64
            )
            grain_mass = np.asarray(
                handle["PartType3"]["Masses"], dtype=np.float64
            )
        expected_gas, expected_grain = analytic_velocities(time)
        gas_mean = math.fsum(map(float, gas_velocity)) / len(gas_velocity)
        grain_mean = math.fsum(map(float, grain_velocity)) / len(grain_velocity)
        gas_stddev = math.sqrt(
            math.fsum((float(value) - gas_mean) ** 2 for value in gas_velocity)
            / len(gas_velocity)
        )
        grain_stddev = math.sqrt(
            math.fsum(
                (float(value) - grain_mean) ** 2 for value in grain_velocity
            )
            / len(grain_velocity)
        )
        gas_rms = math.sqrt(
            math.fsum(
                (float(value) - expected_gas) ** 2 for value in gas_velocity
            )
            / len(gas_velocity)
        )
        grain_rms = math.sqrt(
            math.fsum(
                (float(value) - expected_grain) ** 2
                for value in grain_velocity
            )
            / len(grain_velocity)
        )
        momentum = math.fsum(
            [
                *(
                    float(mass) * float(velocity)
                    for mass, velocity in zip(
                        gas_mass, gas_velocity, strict=True
                    )
                ),
                *(
                    float(mass) * float(velocity)
                    for mass, velocity in zip(
                        grain_mass, grain_velocity, strict=True
                    )
                ),
            ]
        )
        rows.append(
            [
                str(expected_number),
                _format(time),
                _format(gas_mean),
                _format(gas_stddev),
                _format(grain_mean),
                _format(grain_stddev),
                _format(momentum),
                _format(expected_gas),
                _format(expected_grain),
                _format(gas_rms),
                _format(grain_rms),
            ]
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.writer(text, lineterminator="\n")
                writer.writerow(FIELDS)
                writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    reduce(args.snapshot_directory, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
