from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import unittest


ORACLE_DIR = Path(__file__).parents[1] / "oracles" / "shocktube"


class ShocktubeOracleTests(unittest.TestCase):
    def test_public_assets_are_fully_pinned(self) -> None:
        manifest = json.loads(
            (ORACLE_DIR / "assets.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(
            {asset["name"] for asset in manifest["assets"]},
            {
                "shocktube_ics_emass.hdf5",
                "shocktube_ics_diffmass.hdf5",
                "shocktube_exact.txt",
            },
        )
        for asset in manifest["assets"]:
            self.assertTrue(asset["url"].startswith("http://www.tapir.caltech.edu/"))
            self.assertEqual(len(asset["sha256"]), 64)
            self.assertGreater(asset["size_bytes"], 0)
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
        self.assertEqual(manifest["timeline"]["steps"], 8_192)
        self.assertEqual(
            manifest["timeline"]["shared_step_ticks"],
            140_737_488_355_328,
        )
        self.assertEqual(manifest["timeline"]["scheduled_outputs"], 11)
        self.assertEqual(
            set(manifest["public_reference"]["volume_weighted_l1"]),
            {"density", "pressure", "entropy", "velocity_x"},
        )
        self.assertTrue(
            all(
                0.0 < value < 0.01
                for value in manifest["public_reference"][
                    "volume_weighted_l1"
                ].values()
            )
        )

        for input_name in (
            "config",
            "parameters",
            "first_drift_parameters",
            "first_drift_output_times",
            "resolved_parameters",
            "post_kick_observation_patch",
        ):
            record = manifest["inputs"][input_name]
            digest = hashlib.sha256(
                (ORACLE_DIR / record["path"]).read_bytes()
            ).hexdigest()
            self.assertEqual(digest, record["sha256"])

        phases = [snapshot["phase"] for snapshot in manifest["snapshots"]]
        self.assertEqual(
            phases,
            [
                "initial drift after first half-kick",
                "first endpoint drift before force and second kick",
                "first completed endpoint after force and second kick",
                "scheduled terminal drift before endpoint force and second kick",
            ],
        )
        for snapshot in manifest["snapshots"]:
            table = ORACLE_DIR / snapshot["table"]
            self.assertEqual(
                hashlib.sha256(table.read_bytes()).hexdigest(),
                snapshot["table_sha256"],
            )
            with gzip.open(table, "rt", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 320)
            self.assertEqual(
                [int(row["particle_id"]) for row in rows],
                list(range(320)),
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
                self.assertLess(values[0], 80.0)
                self.assertTrue(all(value > 0.0 for value in values[2:]))

    def test_corrected_c_diffmass_manifest_and_tables_are_self_consistent(
        self,
    ) -> None:
        manifest = json.loads(
            (ORACLE_DIR / "diffmass-evolution-manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(manifest["source_commit"]), 40)
        self.assertEqual(manifest["corrected_c_finding"], "GZ-0010")
        self.assertEqual(manifest["timeline"]["steps"], 8_192)
        self.assertEqual(manifest["timeline"]["scheduled_outputs"], 11)

        for input_name in ("config", "parameters", "resolved_parameters"):
            record = manifest["inputs"][input_name]
            self.assertEqual(
                hashlib.sha256(
                    (ORACLE_DIR / record["path"]).read_bytes()
                ).hexdigest(),
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
                list(range(512)),
            )


if __name__ == "__main__":
    unittest.main()
