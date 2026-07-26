#!/usr/bin/env python3
"""Gate the first sixteen McNally KH snapshots against published diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from validation.oracles.analyze_kh_mcnally import (
    OFFICIAL_PARTICLE_COUNT,
    ReferenceError,
    SnapshotError,
    load_public_references,
    reference_at_time,
)
from validation.oracles.reduce_kh_mcnally_trajectory import (
    EXPECTED_TIMES,
    FIELDS,
    reduce_trajectory,
)

# The corrected C trajectory differs from the converged Pencil mode curve by
# at most 13.95% relative (and 0.01769 absolute) through t=1.5. A direct 20%
# band around the published curve preserves resolution headroom without
# accepting a frozen mode. Corrected C is a separate implementation-equivalence
# check and therefore uses a much tighter absolute threshold.
MODE_REFERENCE_RELATIVE_LIMIT = 0.20
MODE_REFERENCE_ABSOLUTE_FLOOR = 2.5e-4
CORRECTED_C_MODE_ABSOLUTE_LIMIT = 5.0e-4
MINIMUM_TERMINAL_MODE_GROWTH = 8.0

# This extrema diagnostic is deliberately noise-sensitive: corrected C
# reaches 4.72 times the published value near t=0.9. The floor contains that
# spike and the factor contains the later nonlinear curve while still
# rejecting a gross vertical-energy blow-up.
MAXIMUM_VERTICAL_KE_ABSOLUTE_FLOOR = 0.025
MAXIMUM_VERTICAL_KE_REFERENCE_FACTOR = 3.0
CORRECTED_C_VERTICAL_KE_ABSOLUTE_FLOOR = 2.5e-5
CORRECTED_C_VERTICAL_KE_RELATIVE_LIMIT = 0.15

# Corrected C has zero mass drift, 2.8e-17 absolute component-momentum drift,
# and 3.37e-7 relative drift in the staggered snapshot energy diagnostic.
MAXIMUM_RELATIVE_MASS_DRIFT = 1.0e-12
MAXIMUM_ABSOLUTE_MOMENTUM_DRIFT = 1.0e-10
MAXIMUM_RELATIVE_DIAGNOSTIC_ENERGY_DRIFT = 1.0e-4
MINIMUM_EFFECTIVE_NEIGHBORS = 39.5
MAXIMUM_EFFECTIVE_NEIGHBORS = 40.5


def load_corrected_c_trajectory(path: Path) -> list[dict[str, float | int]]:
    """Load the pinned reduced corrected-C trajectory with a strict schema."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ReferenceError(
                f"{path}: corrected-C columns differ from the pinned schema"
            )
        source_rows = list(reader)
    if len(source_rows) != len(EXPECTED_TIMES):
        raise ReferenceError(
            f"{path}: expected {len(EXPECTED_TIMES)} corrected-C rows, "
            f"found {len(source_rows)}"
        )
    rows: list[dict[str, float | int]] = []
    for index, (source, expected_time) in enumerate(
        zip(source_rows, EXPECTED_TIMES, strict=True)
    ):
        try:
            row: dict[str, float | int] = {
                name: (
                    int(source[name])
                    if name == "particle_count"
                    else float(source[name])
                )
                for name in FIELDS
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ReferenceError(f"{path}: invalid row {index + 2}") from error
        if any(
            not math.isfinite(float(row[name]))
            for name in FIELDS
            if name != "particle_count"
        ):
            raise ReferenceError(f"{path}: row {index + 2} is non-finite")
        if not math.isclose(
            float(row["time"]), expected_time, rel_tol=0.0, abs_tol=2.0e-15
        ):
            raise ReferenceError(
                f"{path}: row {index + 2} time is outside the 16-snapshot schedule"
            )
        if row["particle_count"] != OFFICIAL_PARTICLE_COUNT:
            raise ReferenceError(
                f"{path}: row {index + 2} has the wrong particle count"
            )
        for name in (
            "mode_amplitude",
            "maximum_vertical_kinetic_energy_density",
            "mass",
            "diagnostic_total_energy",
            "density_minimum",
            "density_maximum",
            "pressure_minimum",
            "pressure_maximum",
            "smoothing_length_minimum",
            "smoothing_length_maximum",
            "smoothing_length_mean",
            "effective_neighbors_minimum",
            "effective_neighbors_maximum",
            "effective_neighbors_mean",
        ):
            if float(row[name]) <= 0.0:
                raise ReferenceError(f"{path}: row {index + 2} has non-positive {name}")
        rows.append(row)
    return rows


def enforce_trajectory_rows(
    rows: list[dict[str, float | int]],
    references: dict[str, Any],
    corrected_c: list[dict[str, float | int]],
) -> dict[str, Any]:
    """Enforce published-curve, anti-noop, state, and conservation gates."""
    if len(rows) != len(EXPECTED_TIMES) or len(corrected_c) != len(EXPECTED_TIMES):
        raise SnapshotError("KH gate requires exactly sixteen rows through t=1.5")
    reports: list[dict[str, Any]] = []
    initial = rows[0]
    initial_mass = float(initial["mass"])
    initial_energy = float(initial["diagnostic_total_energy"])
    initial_momentum = (
        float(initial["momentum_x"]),
        float(initial["momentum_y"]),
    )
    failures: list[str] = []
    for index, (row, c_row, expected_time) in enumerate(
        zip(rows, corrected_c, EXPECTED_TIMES, strict=True)
    ):
        time = float(row["time"])
        if not math.isclose(time, expected_time, rel_tol=0.0, abs_tol=2.0e-15):
            failures.append(f"row {index}: time {time} != {expected_time}")
            continue
        if int(row["particle_count"]) != OFFICIAL_PARTICLE_COUNT:
            failures.append(
                f"t={time}: particle count {row['particle_count']} != "
                f"{OFFICIAL_PARTICLE_COUNT}"
            )
        for name in FIELDS:
            if name != "particle_count" and not math.isfinite(float(row[name])):
                failures.append(f"t={time}: {name} is non-finite")
        for name in (
            "mode_amplitude",
            "maximum_vertical_kinetic_energy_density",
            "mass",
            "diagnostic_total_energy",
            "density_minimum",
            "density_maximum",
            "pressure_minimum",
            "pressure_maximum",
            "smoothing_length_minimum",
            "smoothing_length_maximum",
            "smoothing_length_mean",
            "effective_neighbors_minimum",
            "effective_neighbors_maximum",
            "effective_neighbors_mean",
        ):
            if float(row[name]) <= 0.0:
                failures.append(f"t={time}: {name} is non-positive")
        effective_neighbor_range = (
            float(row["effective_neighbors_minimum"]),
            float(row["effective_neighbors_maximum"]),
        )
        if (
            effective_neighbor_range[0] < MINIMUM_EFFECTIVE_NEIGHBORS
            or effective_neighbor_range[1] > MAXIMUM_EFFECTIVE_NEIGHBORS
        ):
            failures.append(
                f"t={time}: effective-neighbor range "
                f"{effective_neighbor_range} is outside "
                f"[{MINIMUM_EFFECTIVE_NEIGHBORS}, {MAXIMUM_EFFECTIVE_NEIGHBORS}]"
            )

        published = reference_at_time(time, references)
        mode = float(row["mode_amplitude"])
        published_mode = published["mode_amplitude"]
        published_limit = max(
            MODE_REFERENCE_ABSOLUTE_FLOOR,
            MODE_REFERENCE_RELATIVE_LIMIT * published_mode,
        )
        published_difference = abs(mode - published_mode)
        if published_difference > published_limit:
            failures.append(
                f"t={time}: mode error {published_difference:.8g} exceeds "
                f"published-curve limit {published_limit:.8g}"
            )
        corrected_mode = float(c_row["mode_amplitude"])
        corrected_difference = abs(mode - corrected_mode)
        if corrected_difference > CORRECTED_C_MODE_ABSOLUTE_LIMIT:
            failures.append(
                f"t={time}: mode difference from corrected C "
                f"{corrected_difference:.8g} exceeds "
                f"{CORRECTED_C_MODE_ABSOLUTE_LIMIT:.8g}"
            )

        vertical_ke = float(row["maximum_vertical_kinetic_energy_density"])
        vertical_ke_limit = max(
            MAXIMUM_VERTICAL_KE_ABSOLUTE_FLOOR,
            MAXIMUM_VERTICAL_KE_REFERENCE_FACTOR
            * published["maximum_vertical_kinetic_energy_density"],
        )
        if vertical_ke > vertical_ke_limit:
            failures.append(
                f"t={time}: maximum vertical kinetic-energy density "
                f"{vertical_ke:.8g} exceeds broad limit {vertical_ke_limit:.8g}"
            )
        corrected_vertical_ke = float(
            c_row["maximum_vertical_kinetic_energy_density"]
        )
        corrected_vertical_ke_limit = max(
            CORRECTED_C_VERTICAL_KE_ABSOLUTE_FLOOR,
            CORRECTED_C_VERTICAL_KE_RELATIVE_LIMIT * corrected_vertical_ke,
        )
        corrected_vertical_ke_difference = abs(
            vertical_ke - corrected_vertical_ke
        )
        if corrected_vertical_ke_difference > corrected_vertical_ke_limit:
            failures.append(
                f"t={time}: vertical kinetic-energy density difference from "
                f"corrected C {corrected_vertical_ke_difference:.8g} exceeds "
                f"{corrected_vertical_ke_limit:.8g}"
            )

        relative_mass_drift = abs(float(row["mass"]) / initial_mass - 1.0)
        momentum_drift = max(
            abs(float(row["momentum_x"]) - initial_momentum[0]),
            abs(float(row["momentum_y"]) - initial_momentum[1]),
        )
        relative_energy_drift = abs(
            float(row["diagnostic_total_energy"]) / initial_energy - 1.0
        )
        if relative_mass_drift > MAXIMUM_RELATIVE_MASS_DRIFT:
            failures.append(
                f"t={time}: relative mass drift {relative_mass_drift:.8g} exceeds "
                f"{MAXIMUM_RELATIVE_MASS_DRIFT:.8g}"
            )
        if momentum_drift > MAXIMUM_ABSOLUTE_MOMENTUM_DRIFT:
            failures.append(
                f"t={time}: momentum drift {momentum_drift:.8g} exceeds "
                f"{MAXIMUM_ABSOLUTE_MOMENTUM_DRIFT:.8g}"
            )
        if relative_energy_drift > MAXIMUM_RELATIVE_DIAGNOSTIC_ENERGY_DRIFT:
            failures.append(
                f"t={time}: relative diagnostic-energy drift "
                f"{relative_energy_drift:.8g} exceeds "
                f"{MAXIMUM_RELATIVE_DIAGNOSTIC_ENERGY_DRIFT:.8g}"
            )
        reports.append(
            {
                "time": time,
                "mode_amplitude": mode,
                "published_mode_amplitude": published_mode,
                "published_mode_absolute_error": published_difference,
                "corrected_c_mode_amplitude": corrected_mode,
                "corrected_c_mode_absolute_difference": corrected_difference,
                "maximum_vertical_kinetic_energy_density": vertical_ke,
                "maximum_vertical_kinetic_energy_density_limit": vertical_ke_limit,
                "relative_mass_drift": relative_mass_drift,
                "maximum_absolute_component_momentum_drift": momentum_drift,
                "relative_diagnostic_energy_drift": relative_energy_drift,
                "effective_neighbor_range": effective_neighbor_range,
            }
        )

    growth = float(rows[-1]["mode_amplitude"]) / float(rows[0]["mode_amplitude"])
    if not math.isfinite(growth) or growth < MINIMUM_TERMINAL_MODE_GROWTH:
        failures.append(
            f"terminal mode growth {growth:.8g} is below anti-noop floor "
            f"{MINIMUM_TERMINAL_MODE_GROWTH:.8g}"
        )
    if failures:
        raise SnapshotError("; ".join(failures))
    return {"terminal_mode_growth": growth, "snapshots": reports}


def check_trajectory(
    output_directory: Path,
    reference_directory: Path,
    corrected_c_path: Path,
) -> dict[str, Any]:
    rows = reduce_trajectory(output_directory)
    references = load_public_references(reference_directory)
    corrected_c = load_corrected_c_trajectory(corrected_c_path)
    return enforce_trajectory_rows(rows, references, corrected_c)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--reference-directory",
        type=Path,
        default=Path(__file__).with_name("kh_mcnally"),
    )
    parser.add_argument("--corrected-c-trajectory", type=Path)
    args = parser.parse_args()
    corrected_c = args.corrected_c_trajectory or (
        args.reference_directory / "corrected-c-trajectory.csv"
    )
    try:
        report = check_trajectory(
            args.output_directory, args.reference_directory, corrected_c
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
