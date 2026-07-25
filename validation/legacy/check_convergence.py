#!/usr/bin/env python3
"""Require decreasing sound-wave errors across three particle resolutions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs=3, type=Path)
    parser.add_argument("--minimum-order", type=float, default=0.8)
    args = parser.parse_args()

    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.metrics]
    fields = [
        "relative_l1_density_error",
        "relative_l1_velocity_error",
        "relative_l1_pressure_error",
    ]
    orders: dict[str, list[float]] = {}
    for field in fields:
        errors = [float(row[field]) for row in rows]
        if not errors[0] > errors[1] > errors[2]:
            raise SystemExit(f"{field} does not decrease monotonically: {errors}")
        orders[field] = [
            math.log(errors[0] / errors[1], 2.0),
            math.log(errors[1] / errors[2], 2.0),
        ]
        if min(orders[field]) < args.minimum_order:
            raise SystemExit(
                f"{field} convergence order below {args.minimum_order}: {orders[field]}"
            )
    print(json.dumps({"convergence_orders": orders}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
