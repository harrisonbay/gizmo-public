from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import unittest


ORACLE_DIR = Path(__file__).parents[1] / "oracles" / "soundwave"


class SoundwaveEvolutionOracleTests(unittest.TestCase):
    def test_manifest_and_semantic_tables_are_self_consistent(self) -> None:
        manifest = json.loads(
            (ORACLE_DIR / "evolution-manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(manifest["source_commit"]), 40)
        self.assertEqual(manifest["timeline"]["steps"], 65_536)
        self.assertEqual(manifest["timeline"]["shared_step_ticks"], 8192)

        for snapshot in manifest["snapshots"]:
            table = ORACLE_DIR / snapshot["table"]
            digest = hashlib.sha256(table.read_bytes()).hexdigest()
            self.assertEqual(digest, snapshot["table_sha256"])
            with gzip.open(table, "rt", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2048)
            self.assertEqual(
                [int(row["particle_id"]) for row in rows],
                list(range(2048)),
            )
            for row in rows:
                values = {
                    field: float(row[field])
                    for field in (
                        "x",
                        "velocity_x",
                        "density",
                        "specific_internal_energy",
                        "smoothing_length",
                        "mass",
                    )
                }
                self.assertTrue(all(math.isfinite(value) for value in values.values()))
                self.assertGreaterEqual(values["x"], 0.0)
                self.assertLess(values["x"], 1.0)
                for field in (
                    "density",
                    "specific_internal_energy",
                    "smoothing_length",
                    "mass",
                ):
                    self.assertGreater(values[field], 0.0)


if __name__ == "__main__":
    unittest.main()
