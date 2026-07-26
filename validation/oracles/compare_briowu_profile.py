#!/usr/bin/env python3
"""Compare a reduced Brio-Wu profile with the digitized published reference."""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
import math
from pathlib import Path


PROFILE_COLUMNS = {
    "vx": "velocity_x_mean",
    "vy": "velocity_y_mean",
    "bx": "magnetic_x_mean",
    "by": "magnetic_y_mean",
    "rho": "density_mean",
    "u": "specific_internal_energy_mean",
    "pressure": "pressure_mean",
}


def read_reference(path: Path) -> dict[str, list[tuple[float, float]]]:
    curves = {field: [] for field in PROFILE_COLUMNS}
    with path.open(newline="", encoding="ascii") as stream:
        for row in csv.DictReader(stream):
            field = row["field"]
            if field not in curves:
                raise ValueError(f"unexpected reference field {field!r}")
            curves[field].append((float(row["x"]), float(row["value"])))
    for field, points in curves.items():
        if len(points) < 2:
            raise ValueError(f"reference field {field!r} has fewer than two points")
        if any(not (math.isfinite(x) and math.isfinite(value)) for x, value in points):
            raise ValueError(f"reference field {field!r} contains non-finite values")
        if any(right[0] < left[0] for left, right in zip(points, points[1:])):
            raise ValueError(f"reference field {field!r} is not monotone in x")
    return curves


def read_profile(path: Path) -> list[dict[str, float]]:
    stream = (
        gzip.open(path, mode="rt", newline="", encoding="utf-8")
        if path.suffix == ".gz"
        else path.open(mode="r", newline="", encoding="utf-8")
    )
    with stream:
        rows = [
            {
                "x": float(row["x_mean"]),
                **{field: float(row[column]) for field, column in PROFILE_COLUMNS.items()},
            }
            for row in csv.DictReader(stream)
        ]
    if not rows:
        raise ValueError("profile is empty")
    if any(
        not all(math.isfinite(value) for value in row.values())
        for row in rows
    ):
        raise ValueError("profile contains non-finite values")
    if any(right["x"] < left["x"] for left, right in zip(rows, rows[1:])):
        raise ValueError("profile is not monotone in x")
    return rows


def interpolate(points: list[tuple[float, float]], x: float) -> float:
    xs = [point[0] for point in points]
    right = bisect.bisect_right(xs, x)
    if right == 0:
        return points[0][1]
    if right == len(points):
        return points[-1][1]
    x0, value0 = points[right - 1]
    x1, value1 = points[right]
    if x1 == x0:
        return 0.5 * (value0 + value1)
    fraction = (x - x0) / (x1 - x0)
    return value0 + fraction * (value1 - value0)


def compare(
    profile: list[dict[str, float]],
    curves: dict[str, list[tuple[float, float]]],
) -> dict[str, object]:
    x_min = max(points[0][0] for points in curves.values())
    x_max = min(points[-1][0] for points in curves.values())
    selected = [row for row in profile if x_min <= row["x"] <= x_max]
    if not selected:
        raise ValueError(f"profile has no samples in reference domain [{x_min}, {x_max}]")

    fields: dict[str, dict[str, float | int]] = {}
    for field, points in curves.items():
        errors = [
            abs(row[field] - interpolate(points, row["x"]))
            for row in selected
        ]
        reference_values = [value for _, value in points]
        scale = max(reference_values) - min(reference_values)
        if scale == 0.0:
            scale = max(max(abs(value) for value in reference_values), 1.0)
        ordered = sorted(errors)
        percentile_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
        l1 = sum(errors) / len(errors)
        fields[field] = {
            "samples": len(errors),
            "l1": l1,
            "normalized_l1": l1 / scale,
            "p95_absolute_error": ordered[percentile_index],
            "maximum_absolute_error": ordered[-1],
            "reference_scale": scale,
        }
    return {
        "reference_x_domain": [x_min, x_max],
        "profile_samples": len(selected),
        "fields": fields,
        "warning": (
            "Figure-path interpolation is an approximate physics-profile metric; "
            "it is not an original-table or elementwise oracle."
        ),
    }


def parse_limits(values: list[str]) -> dict[str, float]:
    limits = {}
    for value in values:
        field, separator, raw_limit = value.partition("=")
        if not separator or field not in PROFILE_COLUMNS:
            raise ValueError(f"invalid limit {value!r}; expected FIELD=VALUE")
        limit = float(raw_limit)
        if not math.isfinite(limit) or limit < 0.0:
            raise ValueError(f"invalid nonnegative finite limit {value!r}")
        limits[field] = limit
    return limits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).parent / "briowu" / "published_exact_figure.csv",
    )
    parser.add_argument(
        "--maximum-normalized-l1",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
    )
    args = parser.parse_args()
    result = compare(read_profile(args.profile), read_reference(args.reference))
    print(json.dumps(result, indent=2, sort_keys=True))

    limits = parse_limits(args.maximum_normalized_l1)
    failures = []
    fields = result["fields"]
    assert isinstance(fields, dict)
    for field, limit in limits.items():
        observed = fields[field]["normalized_l1"]
        if observed > limit:
            failures.append(f"{field} normalized L1 {observed:.8g} exceeds {limit:.8g}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
