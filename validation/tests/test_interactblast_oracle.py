from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import unittest

from validation.oracles.compare_interactblast_reference import (
    metrics,
    read_reference,
    read_state,
)


ORACLE_DIR = Path(__file__).parents[1] / "oracles" / "interactblast"


class InteractblastOracleTests(unittest.TestCase):
    def test_public_assets_are_fully_pinned(self) -> None:
        manifest = json.loads(
            (ORACLE_DIR / "assets.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(
            {asset["name"] for asset in manifest["assets"]},
            {"interactblast_ics.hdf5", "interactblast_exact.txt", "public.params"},
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
            (ORACLE_DIR / "evolution-manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(manifest["source_commit"]), 40)
        self.assertEqual(manifest["profile"]["particle_count"], 512)
        self.assertEqual(manifest["timeline"]["steps"], 262_144)
        self.assertEqual(manifest["timeline"]["shared_step_ticks"], 2**42)
        self.assertEqual(manifest["timeline"]["scheduled_outputs"], 11)
        self.assertTrue(manifest["observations"]["wall_crossings_observed"])

        for record in manifest["inputs"].values():
            self.assertEqual(
                hashlib.sha256((ORACLE_DIR / record["path"]).read_bytes()).hexdigest(),
                record["sha256"],
            )

        for snapshot in manifest["snapshots"]:
            table = ORACLE_DIR / snapshot["table"]
            self.assertEqual(
                hashlib.sha256(table.read_bytes()).hexdigest(),
                snapshot["table_sha256"],
            )
            with gzip.open(table, "rt", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 512)
            self.assertEqual(
                [int(row["particle_id"]) for row in rows],
                list(range(1, 513)),
            )
            for row in rows:
                values = [
                    float(row[field])
                    for field in (
                        "x",
                        "velocity_x",
                        "density",
                        "specific_internal_energy",
                        "smoothing_length",
                        "mass",
                    )
                ]
                self.assertTrue(all(math.isfinite(value) for value in values))
                self.assertGreaterEqual(values[0], 0.0)
                self.assertLessEqual(values[0], 1.0)
                self.assertTrue(all(value > 0.0 for value in values[2:]))

    def test_public_reference_metrics_match_the_manifest(self) -> None:
        reference_path = ORACLE_DIR / "interactblast_exact.txt"
        if not reference_path.exists():
            self.skipTest("ignored public reference asset has not been fetched")
        manifest = json.loads(
            (ORACLE_DIR / "evolution-manifest.json").read_text(encoding="utf-8")
        )
        measured = metrics(
            read_state(ORACLE_DIR / "evolution_tfinal.csv.gz"),
            read_reference(reference_path),
        )
        expected = manifest["public_reference"]["volume_weighted_l1"]
        self.assertEqual(set(measured), set(expected))
        for field, value in measured.items():
            self.assertAlmostEqual(value, expected[field], delta=1.0e-12)


if __name__ == "__main__":
    unittest.main()
