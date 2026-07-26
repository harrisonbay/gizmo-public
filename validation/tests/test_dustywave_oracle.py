from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import unittest

from validation.oracles.compare_dustywave_reference import (
    metrics,
    read_reference,
    read_state,
)


ORACLE_DIR = Path(__file__).parents[1] / "oracles" / "dustywave"


class DustywaveOracleTests(unittest.TestCase):
    def test_public_assets_are_fully_pinned(self) -> None:
        manifest = json.loads((ORACLE_DIR / "assets.json").read_text())
        self.assertEqual(
            {asset["name"] for asset in manifest["assets"]},
            {"dustywave_ics.hdf5", "dustwave_exact.txt", "public.params"},
        )
        for asset in manifest["assets"]:
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
        self.assertEqual(manifest["timeline"]["steps"], 32_768)
        self.assertEqual(manifest["timeline"]["shared_step_ticks"], 2**45)
        self.assertTrue(
            manifest["observations"]["all_softened_profile_particles_shared_one_timebin"]
        )
        self.assertTrue(manifest["observations"]["literal_public_profile_stalls"])
        for record in manifest["inputs"].values():
            self.assertEqual(
                hashlib.sha256((ORACLE_DIR / record["path"]).read_bytes()).hexdigest(),
                record["sha256"],
            )
        initial_momentum = manifest["snapshots"][0]["total_momentum_x"]
        for snapshot_index, snapshot in enumerate(manifest["snapshots"]):
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
            measured_momentum = sum(
                float(row["mass"]) * float(row["velocity_x"]) for row in rows
            )
            self.assertAlmostEqual(
                measured_momentum, snapshot["total_momentum_x"], delta=1.0e-24
            )
            momentum_tolerance = 2.0e-14 if snapshot_index == 3 else 1.0e-17
            self.assertLess(
                abs(measured_momentum - initial_momentum), momentum_tolerance
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

    def test_public_reference_metrics_match_the_manifest(self) -> None:
        reference_path = ORACLE_DIR / "dustwave_exact.txt"
        if not reference_path.exists():
            self.skipTest("ignored public reference asset has not been fetched")
        manifest = json.loads(
            (ORACLE_DIR / "evolution-manifest.json").read_text()
        )
        measured = metrics(
            read_state(ORACLE_DIR / "evolution_t1.2.csv.gz"),
            read_reference(reference_path),
        )
        for phase in ("absolute_rms", "peak_normalized_rms"):
            for species in ("grain_velocity_x", "gas_velocity_x"):
                self.assertAlmostEqual(
                    measured[f"{species}_{phase}"],
                    manifest["public_reference"][phase][species],
                    delta=1.0e-14,
                )


if __name__ == "__main__":
    unittest.main()
