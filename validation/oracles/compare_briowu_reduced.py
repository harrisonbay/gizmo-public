#!/usr/bin/env python3
"""Compare two deterministic reduced Brio-Wu profile tables."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from typing import TextIO


PHYSICAL_FIELDS = {
    "vx": "velocity_x_mean",
    "vy": "velocity_y_mean",
    "bx": "magnetic_x_mean",
    "by": "magnetic_y_mean",
    "rho": "density_mean",
    "u": "specific_internal_energy_mean",
    "pressure": "pressure_mean",
}

DIAGNOSTIC_FIELDS = {
    "x": "x_mean",
    "hsml": "smoothing_length_mean",
    "mass": "mass_sum",
}


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", newline="", encoding="utf-8")
    return path.open(mode="r", newline="", encoding="utf-8")


def read_profile(path: Path) -> list[dict[str, float]]:
    with _open_text(path) as stream:
        reader = csv.DictReader(stream)
        expected = {
            "rank_bin",
            "particle_count",
            *PHYSICAL_FIELDS.values(),
            *DIAGNOSTIC_FIELDS.values(),
        }
        missing = expected.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = [
            {
                "rank_bin": int(row["rank_bin"]),
                "particle_count": int(row["particle_count"]),
                **{
                    field: float(row[column])
                    for field, column in {
                        **PHYSICAL_FIELDS,
                        **DIAGNOSTIC_FIELDS,
                    }.items()
                },
            }
            for row in reader
        ]
    if not rows:
        raise ValueError(f"{path}: profile is empty")
    for expected_bin, row in enumerate(rows):
        if row["rank_bin"] != expected_bin:
            raise ValueError(
                f"{path}: rank_bin {row['rank_bin']} at row {expected_bin}"
            )
        if row["particle_count"] <= 0:
            raise ValueError(f"{path}: non-positive particle count at bin {expected_bin}")
        for field, value in row.items():
            if field not in {"rank_bin", "particle_count"} and not math.isfinite(value):
                raise ValueError(
                    f"{path}: non-finite {field} at bin {expected_bin}"
                )
    return rows


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def compare(
    actual: list[dict[str, float]],
    reference: list[dict[str, float]],
) -> dict[str, object]:
    if len(actual) != len(reference):
        raise ValueError(
            f"profile row counts differ: actual={len(actual)}, reference={len(reference)}"
        )
    for index, (actual_row, reference_row) in enumerate(zip(actual, reference)):
        for field in ("rank_bin", "particle_count"):
            if actual_row[field] != reference_row[field]:
                raise ValueError(
                    f"{field} differs at row {index}: "
                    f"actual={actual_row[field]}, reference={reference_row[field]}"
                )

    fields: dict[str, dict[str, float | int]] = {}
    for field in (*PHYSICAL_FIELDS, *DIAGNOSTIC_FIELDS):
        actual_values = [row[field] for row in actual]
        reference_values = [row[field] for row in reference]
        absolute_errors = [
            abs(actual_value - reference_value)
            for actual_value, reference_value in zip(actual_values, reference_values)
        ]
        symmetric_relative_errors = [
            abs(actual_value - reference_value)
            / max(abs(actual_value), abs(reference_value), 1.0e-300)
            for actual_value, reference_value in zip(
                actual_values, reference_values
            )
        ]
        scale = max(reference_values) - min(reference_values)
        if scale == 0.0:
            scale = max(max(abs(value) for value in reference_values), 1.0)
        mean_absolute = math.fsum(absolute_errors) / len(absolute_errors)
        fields[field] = {
            "samples": len(absolute_errors),
            "mean_absolute_error": mean_absolute,
            "normalized_l1": mean_absolute / scale,
            "mean_symmetric_relative_error": (
                math.fsum(symmetric_relative_errors)
                / len(symmetric_relative_errors)
            ),
            "p95_absolute_error": _percentile(absolute_errors, 0.95),
            "maximum_absolute_error": max(absolute_errors),
            "reference_scale": scale,
        }
    return {"profile_rows": len(actual), "fields": fields}


def parse_limits(
    values: list[str],
    valid_fields: set[str],
    description: str,
) -> dict[str, float]:
    limits = {}
    for value in values:
        field, separator, raw_limit = value.partition("=")
        if not separator or field not in valid_fields:
            raise ValueError(
                f"invalid {description} {value!r}; expected FIELD=VALUE "
                f"for one of {sorted(valid_fields)}"
            )
        limit = float(raw_limit)
        if not math.isfinite(limit) or limit < 0.0:
            raise ValueError(f"invalid nonnegative finite limit {value!r}")
        if field in limits:
            raise ValueError(f"duplicate limit for {field!r}")
        limits[field] = limit
    return limits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("actual", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument(
        "--maximum-normalized-l1",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
    )
    parser.add_argument(
        "--maximum-mean-relative-error",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
    )
    args = parser.parse_args()

    result = compare(read_profile(args.actual), read_profile(args.reference))
    print(json.dumps(result, indent=2, sort_keys=True))
    normalized_limits = parse_limits(
        args.maximum_normalized_l1,
        set(PHYSICAL_FIELDS) | set(DIAGNOSTIC_FIELDS),
        "normalized-L1 limit",
    )
    relative_limits = parse_limits(
        args.maximum_mean_relative_error,
        set(PHYSICAL_FIELDS) | set(DIAGNOSTIC_FIELDS),
        "mean-relative-error limit",
    )
    failures = []
    fields = result["fields"]
    assert isinstance(fields, dict)
    for field, limit in normalized_limits.items():
        observed = fields[field]["normalized_l1"]
        if observed > limit:
            failures.append(
                f"{field} normalized L1 {observed:.8g} exceeds {limit:.8g}"
            )
    for field, limit in relative_limits.items():
        observed = fields[field]["mean_symmetric_relative_error"]
        if observed > limit:
            failures.append(
                f"{field} mean symmetric relative error "
                f"{observed:.8g} exceeds {limit:.8g}"
            )
    for failure in failures:
        print(f"FAIL: {failure}")
    return int(bool(failures))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
