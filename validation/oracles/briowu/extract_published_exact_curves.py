#!/usr/bin/env python3
"""Digitize the seven dotted Brio-Wu reference curves from the paper figure.

This is vector-figure digitization, not recovery of the author's source table.
The input SVG must be produced from ``briowu_TP3.pdf`` with:

    pdftocairo -svg briowu_TP3.pdf briowu.svg

The pinned hashes and axis calibrations are documented beside the generated CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


EXPECTED_SVG_SHA256 = (
    "64f6160efec9064557c1300d0fd3865bfbe6e8768f0b561fd53e774e8f557fc8"
)

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
POINT_RE = re.compile(rf"([ML])\s*({NUMBER})\s+({NUMBER})")
MATRIX_RE = re.compile(
    rf"matrix\(\s*({NUMBER})\s*,\s*({NUMBER})\s*,\s*({NUMBER})\s*,"
    rf"\s*({NUMBER})\s*,\s*({NUMBER})\s*,\s*({NUMBER})\s*\)"
)


@dataclass(frozen=True)
class LinearCalibration:
    svg_a: float
    value_a: float
    svg_b: float
    value_b: float

    def convert(self, svg_coordinate: float) -> float:
        fraction = (svg_coordinate - self.svg_a) / (self.svg_b - self.svg_a)
        return self.value_a + fraction * (self.value_b - self.value_a)


# Each column's x calibration uses the labelled major ticks x=1.6 and x=2.8.
# Calibrating the columns separately also accounts for their slightly different
# PDF placement offsets.
X_CALIBRATIONS = (
    LinearCalibration(62.808593373, 1.6, 214.238281025, 2.8),
    LinearCalibration(305.359374925, 1.6, 456.789062575, 2.8),
    LinearCalibration(547.910156425, 1.6, 699.339844121, 2.8),
    LinearCalibration(790.460937975, 1.6, 941.890624670, 2.8),
)

# Each y calibration uses two labelled major ticks from its own panel. Values
# are in the published problem's code units. The order matches the seven
# full-width dotted paths in PDF/SVG drawing order.
PANELS = (
    ("vx", 0, LinearCalibration(132.238281450, 0.0, 27.238280904, 0.6)),
    ("vy", 1, LinearCalibration(11.308593773, 0.0, 166.117187446, -1.5)),
    ("bx", 2, LinearCalibration(12.828124900, 0.84, 202.238281250, 0.68)),
    ("by", 3, LinearCalibration(5.781250475, 1.0, 197.445312975, -1.0)),
    ("rho", 0, LinearCalibration(423.609375350, 0.0, 255.902344104, 1.0)),
    ("u", 1, LinearCalibration(378.886718700, 1.0, 267.082031203, 2.0)),
    ("pressure", 2, LinearCalibration(391.085937349, 0.2, 228.460937250, 1.0)),
)

EXPECTED_ENDPOINTS = {
    "vx": (0.0, 0.0),
    "vy": (0.0, 0.0),
    "bx": (0.75, 0.75),
    "by": (1.0, -1.0),
    "rho": (1.0, 0.125),
    "u": (1.0, 0.8),
    "pressure": (1.0, 0.1),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transformed_points(path: ET.Element) -> list[tuple[float, float]]:
    commands = POINT_RE.findall(path.attrib.get("d", ""))
    if not commands:
        return []
    matrix_match = MATRIX_RE.fullmatch(path.attrib.get("transform", ""))
    if matrix_match is None:
        raise ValueError("dotted path has no supported affine transform")
    a, b, c, d, e, f = (float(value) for value in matrix_match.groups())
    result = []
    for _, raw_x, raw_y in commands:
        x, y = float(raw_x), float(raw_y)
        result.append((a * x + c * y + e, b * x + d * y + f))
    return result


def extract(svg: Path) -> list[tuple[str, int, float, float, float, float]]:
    actual_hash = sha256(svg)
    if actual_hash != EXPECTED_SVG_SHA256:
        raise ValueError(
            "unexpected SVG bytes: got "
            f"{actual_hash}, expected {EXPECTED_SVG_SHA256}; use the pinned "
            "figure and conversion described in README.md"
        )

    root = ET.parse(svg).getroot()
    dotted_paths: list[list[tuple[float, float]]] = []
    for element in root.iter():
        if not element.tag.endswith("path"):
            continue
        if element.attrib.get("stroke-dasharray") != "1 3":
            continue
        points = transformed_points(element)
        if points and max(x for x, _ in points) - min(x for x, _ in points) > 180:
            dotted_paths.append(points)

    if len(dotted_paths) != len(PANELS):
        raise ValueError(
            f"expected seven full-panel dotted paths, found {len(dotted_paths)}"
        )

    rows = []
    for (field, column, calibration), points in zip(
        PANELS, dotted_paths, strict=True
    ):
        previous_x = -math.inf
        for index, (svg_x, svg_y) in enumerate(points):
            x = X_CALIBRATIONS[column].convert(svg_x)
            value = calibration.convert(svg_y)
            if x + 1.0e-10 < previous_x:
                raise ValueError(f"{field} path is not monotone in x at point {index}")
            if not (math.isfinite(x) and math.isfinite(value)):
                raise ValueError(f"{field} contains a non-finite coordinate")
            rows.append((field, index, x, value, svg_x, svg_y))
            previous_x = x

        expected_left, expected_right = EXPECTED_ENDPOINTS[field]
        observed_left = rows[-len(points)][3]
        observed_right = rows[-1][3]
        observed_x_left = rows[-len(points)][2]
        observed_x_right = rows[-1][2]
        if abs(observed_x_left - 1.5) > 0.02 or abs(observed_x_right - 3.0) > 0.02:
            raise ValueError(
                f"{field} does not span the plotted x range: "
                f"[{observed_x_left}, {observed_x_right}]"
            )
        if abs(observed_left - expected_left) > 5.0e-4:
            raise ValueError(
                f"{field} left endpoint {observed_left} != {expected_left}"
            )
        if abs(observed_right - expected_right) > 5.0e-4:
            raise ValueError(
                f"{field} right endpoint {observed_right} != {expected_right}"
            )
    return rows


def write_csv(
    output: Path, rows: list[tuple[str, int, float, float, float, float]]
) -> None:
    with output.open("w", newline="", encoding="ascii") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("field", "point_index", "x", "value", "svg_x", "svg_y"))
        for field, index, x, value, svg_x, svg_y in rows:
            writer.writerow(
                (
                    field,
                    index,
                    f"{x:.12g}",
                    f"{value:.12g}",
                    f"{svg_x:.12g}",
                    f"{svg_y:.12g}",
                )
            )


def verify_csv(path: Path) -> None:
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    fields = {row["field"] for row in rows}
    expected_fields = {field for field, _, _ in PANELS}
    if fields != expected_fields:
        raise ValueError(f"CSV fields {fields} != {expected_fields}")
    for field in expected_fields:
        indices = [int(row["point_index"]) for row in rows if row["field"] == field]
        xs = [float(row["x"]) for row in rows if row["field"] == field]
        if indices != list(range(len(indices))):
            raise ValueError(f"{field} point indices are not contiguous")
        if any(right + 1.0e-10 < left for left, right in zip(xs, xs[1:])):
            raise ValueError(f"{field} CSV x coordinates are not monotone")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("svg", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify output already matches a fresh deterministic extraction",
    )
    args = parser.parse_args()

    rows = extract(args.svg)
    if args.check:
        expected = args.output.read_bytes()
        temporary = args.output.with_suffix(args.output.suffix + ".check")
        try:
            write_csv(temporary, rows)
            if temporary.read_bytes() != expected:
                raise ValueError(f"{args.output} does not match fresh extraction")
        finally:
            temporary.unlink(missing_ok=True)
    else:
        write_csv(args.output, rows)
    verify_csv(args.output)
    print(f"verified {len(PANELS)} curves and {len(rows)} points: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, ET.ParseError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
