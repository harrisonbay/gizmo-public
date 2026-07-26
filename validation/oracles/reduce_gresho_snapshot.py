#!/usr/bin/env python3
"""Write a deterministic, ID-sorted semantic table from a Gresho snapshot."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
from pathlib import Path

from .compare_gresho_analytic import read_snapshot

FIELDS = (
    "particle_id",
    "x",
    "y",
    "velocity_x",
    "velocity_y",
    "density",
    "specific_internal_energy",
    "smoothing_length",
    "mass",
)


def reduce(snapshot: Path, output: Path) -> float:
    state = read_snapshot(snapshot)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        output.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text,
    ):
        writer = csv.writer(text, lineterminator="\n")
        writer.writerow(FIELDS)
        for index, particle_id in enumerate(state["ids"]):
            values = (
                state["coordinates"][index, 0],
                state["coordinates"][index, 1],
                state["velocities"][index, 0],
                state["velocities"][index, 1],
                state["density"][index],
                state["internal_energy"][index],
                state["smoothing_length"][index],
                state["masses"][index],
            )
            writer.writerow(
                [str(int(particle_id))]
                + [format(float(value), ".17g") for value in values]
            )
    return float(state["time"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    time = reduce(args.snapshot, args.output)
    print(f"{args.output}: time={time:.17g}")


if __name__ == "__main__":
    main()
