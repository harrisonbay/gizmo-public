from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import unittest
from pathlib import Path

from validation.oracles.check_square_trajectory import (
    EXPECTED_TIMES,
    trajectory_failures,
)

ORACLE_DIR = Path(__file__).resolve().parents[1] / "oracles" / "square"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ideal_trajectory() -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for index, time in enumerate(EXPECTED_TIMES):
        reports.append(
            {
                "time": time,
                "expected_wrapped_shift": [0.5 if index % 2 else 0.0, 0.0],
                "position_component_maximum": [0.0, 0.0],
                "velocity_component_maximum": [0.0, 0.0, 0.0],
                "internal_energy_maximum": 0.0,
                "density_relative_maximum": 0.0,
                "density_absolute_relative_maximum": 0.0,
                "smoothing_length_relative_maximum": 0.0,
                "smoothing_length_absolute_relative_maximum": 0.0,
                "pressure_relative_span": 0.0,
                "effective_neighbor_range": [12.0, 12.0],
                "mass": 1.0,
                "momentum": [1.0, -1.0, 0.0],
                "total_energy": 1.0,
            }
        )
    return reports


class SquareOracleTests(unittest.TestCase):
    def test_public_and_corrected_c_artifacts_are_checksum_pinned(self) -> None:
        assets = json.loads((ORACLE_DIR / "assets.json").read_text())
        for asset in assets["assets"]:
            path = ORACLE_DIR / asset["name"]
            if path.exists():
                self.assertEqual(path.stat().st_size, asset["size_bytes"])
                self.assertEqual(sha256(path), asset["sha256"])

        manifest = json.loads(
            (ORACLE_DIR / "corrected-c-manifest.json").read_text()
        )
        self.assertIsNone(manifest["source_commit"])
        self.assertIsNone(manifest["compiler"])
        self.assertEqual(manifest["mpi_ranks"], 4)
        for key, name in (
            ("analyzer", "analyze_square.py"),
            ("trajectory_checker", "check_square_trajectory.py"),
            ("reducer", "reduce_square_trajectory.py"),
        ):
            self.assertEqual(
                sha256(ORACLE_DIR.parent / name), manifest["sha256"][key]
            )
        for key, name in (
            ("fixture", "square_ics.hdf5"),
            ("config", "corrected-c-config.sh"),
            ("parameters", "corrected-c.params"),
            ("parameters_usedvalues", "parameters-usedvalues"),
            ("reduced_trajectory", "corrected-c-trajectory.csv"),
        ):
            path = ORACLE_DIR / name
            if path.exists():
                self.assertEqual(sha256(path), manifest["sha256"][key])

    def test_corrected_c_reduced_trajectory_satisfies_declared_gates(self) -> None:
        with (ORACLE_DIR / "corrected-c-trajectory.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), len(EXPECTED_TIMES))
        self.assertEqual(
            [float(row["time"]) for row in rows], list(EXPECTED_TIMES)
        )
        self.assertLessEqual(
            max(float(row["position_x_max"]) for row in rows), 1.0e-9
        )
        self.assertLessEqual(
            max(float(row["position_y_max"]) for row in rows), 1.0e-9
        )
        self.assertEqual(max(float(row["velocity_max"]) for row in rows), 0.0)
        self.assertLessEqual(
            max(float(row["pressure_relative_span"]) for row in rows), 2.0e-9
        )
        self.assertGreaterEqual(
            min(float(row["effective_neighbors_min"]) for row in rows), 11.999
        )
        self.assertLessEqual(
            max(float(row["effective_neighbors_max"]) for row in rows), 12.001
        )

    def test_exact_translation_passes(self) -> None:
        self.assertEqual(trajectory_failures(ideal_trajectory()), [])

    def test_gate_rejects_noop_diffusion_bad_geometry_and_nonconservation(self) -> None:
        frozen = ideal_trajectory()
        frozen[1]["position_component_maximum"] = [0.5, 0.0]
        self.assertTrue(
            any("position error" in failure for failure in trajectory_failures(frozen))
        )

        diffused = ideal_trajectory()
        diffused[8]["density_relative_maximum"] = 0.01
        diffused[8]["internal_energy_maximum"] = 0.01
        diffused[8]["pressure_relative_span"] = 0.1
        failures = trajectory_failures(diffused)
        self.assertTrue(any("density drift" in failure for failure in failures))
        self.assertTrue(any("internal-energy" in failure for failure in failures))
        self.assertTrue(any("pressure" in failure for failure in failures))

        scale_invariant_wrong_initialization = ideal_trajectory()
        for report in scale_invariant_wrong_initialization:
            report["density_absolute_relative_maximum"] = 0.1
            report["smoothing_length_absolute_relative_maximum"] = 1.0 - 1.0 / (
                1.1**0.5
            )
        failures = trajectory_failures(scale_invariant_wrong_initialization)
        self.assertTrue(any("initialized density" in failure for failure in failures))
        self.assertTrue(
            any("initialized smoothing length" in failure for failure in failures)
        )

        wrong_neighbors = ideal_trajectory()
        wrong_neighbors[4]["effective_neighbor_range"] = [10.0, 14.0]
        self.assertTrue(
            any(
                "effective-neighbor" in failure
                for failure in trajectory_failures(wrong_neighbors)
            )
        )

        nonconservative = ideal_trajectory()
        nonconservative[12]["mass"] = 1.001
        nonconservative[12]["momentum"] = [1.0, -0.9, 0.0]
        nonconservative[12]["total_energy"] = 1.01
        failures = trajectory_failures(nonconservative)
        self.assertTrue(any("mass drift" in failure for failure in failures))
        self.assertTrue(any("momentum drift" in failure for failure in failures))
        self.assertTrue(any("energy drift" in failure for failure in failures))

    def test_gate_rejects_bad_time_and_aliased_odd_phase(self) -> None:
        wrong_time = ideal_trajectory()
        wrong_time[3]["time"] = 1.6
        self.assertTrue(
            any("expected 1.5" in failure for failure in trajectory_failures(wrong_time))
        )

        non_finite_time = ideal_trajectory()
        non_finite_time[3]["time"] = math.nan
        self.assertTrue(
            any(
                "time contains a non-finite value" in failure
                for failure in trajectory_failures(non_finite_time)
            )
        )

        aliased = copy.deepcopy(ideal_trajectory())
        aliased[5]["expected_wrapped_shift"] = [0.0, 0.0]
        self.assertTrue(
            any(
                "lacks half-box" in failure
                for failure in trajectory_failures(aliased)
            )
        )

    def test_gate_requires_all_twenty_one_snapshots(self) -> None:
        self.assertEqual(
            trajectory_failures(ideal_trajectory()[:-1]),
            ["expected 21 reports, found 20"],
        )


if __name__ == "__main__":
    unittest.main()
