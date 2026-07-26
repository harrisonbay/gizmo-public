from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import unittest

from validation.oracles.compare_dustybox_analytic import (
    analytic_velocities,
    metrics,
    read_state,
)


ORACLE_DIR = Path(__file__).parents[1] / "oracles" / "dustybox"


class DustyboxOracleTests(unittest.TestCase):
    def test_public_assets_are_fully_pinned(self) -> None:
        manifest = json.loads((ORACLE_DIR / "assets.json").read_text())
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(
            {asset["name"] for asset in manifest["assets"]},
            {"dustybox_ics.hdf5", "public.params"},
        )
        for asset in manifest["assets"]:
            self.assertTrue(asset["url"].startswith("http://www.tapir.caltech.edu/"))
            local = ORACLE_DIR / asset["name"]
            if local.exists():
                self.assertEqual(local.stat().st_size, asset["size_bytes"])
                self.assertEqual(
                    hashlib.sha256(local.read_bytes()).hexdigest(),
                    asset["sha256"],
                )

    def test_corrected_c_manifest_and_tables_are_self_consistent(self) -> None:
        manifest = json.loads(
            (ORACLE_DIR / "evolution-manifest.json").read_text()
        )
        self.assertEqual(len(manifest["source_commit"]), 40)
        self.assertEqual(manifest["timeline"]["steps"], 32_768)
        self.assertEqual(manifest["timeline"]["shared_step_ticks"], 2**45)
        self.assertEqual(manifest["timeline"]["scheduled_outputs"], 251)
        self.assertTrue(
            manifest["observations"]["terminal_tick_was_snapped_exactly_to_time_max"]
        )
        for record in manifest["inputs"].values():
            self.assertEqual(
                hashlib.sha256((ORACLE_DIR / record["path"]).read_bytes()).hexdigest(),
                record["sha256"],
            )

        initial_momentum = manifest["snapshots"][0]["analytic_metrics"][
            "total_momentum_x"
        ]
        for snapshot in manifest["snapshots"]:
            table = ORACLE_DIR / snapshot["table"]
            self.assertEqual(
                hashlib.sha256(table.read_bytes()).hexdigest(),
                snapshot["table_sha256"],
            )
            with gzip.open(table, "rt", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 128)
            self.assertEqual(
                [int(row["particle_id"]) for row in rows], list(range(1, 129))
            )
            self.assertEqual(
                [int(row["particle_type"]) for row in rows].count(0), 64
            )
            self.assertEqual(
                [int(row["particle_type"]) for row in rows].count(3), 64
            )
            for row in rows:
                self.assertTrue(
                    all(
                        math.isfinite(float(row[field]))
                        for field in (
                            "x",
                            "velocity_x",
                            "mass",
                            "smoothing_length",
                        )
                    )
                )
                if int(row["particle_type"]) == 0:
                    self.assertTrue(row["density"])
                    self.assertTrue(row["specific_internal_energy"])
                    self.assertFalse(row["grain_size"])
                else:
                    self.assertFalse(row["density"])
                    self.assertFalse(row["specific_internal_energy"])
                    self.assertTrue(row["grain_size"])
            measured = metrics(rows, snapshot["time"])
            self.assertLess(
                abs(measured["total_momentum_x"] - initial_momentum), 1.4e-14
            )
            self.assertEqual(set(measured), set(snapshot["analytic_metrics"]))
            for key, value in measured.items():
                self.assertAlmostEqual(
                    value,
                    snapshot["analytic_metrics"][key],
                    delta=1.0e-14,
                )

    def test_epstein_solution_has_the_documented_initial_and_terminal_limits(
        self,
    ) -> None:
        self.assertEqual(analytic_velocities(0.0), (0.0, 1.0))
        gas, grain = analytic_velocities(1000.0)
        self.assertEqual((gas, grain), (0.5, 0.5))
        with self.assertRaises(ValueError):
            analytic_velocities(-1.0)

    def test_corrected_c_tracks_the_analytic_solution(self) -> None:
        manifest = json.loads(
            (ORACLE_DIR / "evolution-manifest.json").read_text()
        )
        curve_record = manifest["evolution_curve"]
        curve_path = ORACLE_DIR / curve_record["table"]
        self.assertEqual(
            hashlib.sha256(curve_path.read_bytes()).hexdigest(),
            curve_record["table_sha256"],
        )
        with gzip.open(
            curve_path, "rt", encoding="utf-8", newline=""
        ) as handle:
            curve = list(csv.DictReader(handle))
        self.assertEqual(len(curve), curve_record["snapshot_count"])
        self.assertEqual(
            [int(row["snapshot"]) for row in curve], list(range(251))
        )
        self.assertEqual(float(curve[0]["time"]), 0.0)
        self.assertEqual(float(curve[-1]["time"]), 2.5)
        self.assertTrue(
            all(
                float(left["time"]) < float(right["time"])
                for left, right in zip(curve, curve[1:])
            )
        )
        for row in curve:
            time = float(row["time"])
            expected_gas, expected_grain = analytic_velocities(time)
            self.assertAlmostEqual(
                float(row["gas_analytic_velocity_x"]),
                expected_gas,
                delta=1.0e-15,
            )
            self.assertAlmostEqual(
                float(row["grain_analytic_velocity_x"]),
                expected_grain,
                delta=1.0e-15,
            )
            for species, expected in (
                ("gas", expected_gas),
                ("grain", expected_grain),
            ):
                reconstructed_rms = math.hypot(
                    float(row[f"{species}_mean_velocity_x"]) - expected,
                    float(row[f"{species}_spatial_stddev"]),
                )
                self.assertAlmostEqual(
                    float(row[f"{species}_absolute_rms"]),
                    reconstructed_rms,
                    delta=1.0e-14,
                )
        maximum_row = max(
            curve,
            key=lambda row: max(
                float(row["gas_absolute_rms"]),
                float(row["grain_absolute_rms"]),
            ),
        )
        measured_maximum = max(
            float(maximum_row["gas_absolute_rms"]),
            float(maximum_row["grain_absolute_rms"]),
        )
        self.assertAlmostEqual(
            measured_maximum,
            curve_record["maximum_analytic_velocity_rms"],
            delta=1.0e-14,
        )
        self.assertEqual(
            int(maximum_row["snapshot"]),
            curve_record["maximum_analytic_velocity_rms_snapshot"],
        )
        maximum_momentum_error = max(
            abs(float(row["total_momentum_x"]) - 1.0) for row in curve
        )
        self.assertAlmostEqual(
            maximum_momentum_error,
            curve_record["maximum_total_momentum_error"],
            delta=1.0e-16,
        )
        self.assertLess(measured_maximum, 6.0e-5)


if __name__ == "__main__":
    unittest.main()
