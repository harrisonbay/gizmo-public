from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from validation.oracles.compare_briowu_profile import (
    compare as compare_published,
    read_profile as read_published_profile,
    read_reference as read_published_reference,
)
from validation.oracles.compare_briowu_reduced import (
    compare as compare_reduced,
    parse_limits,
    read_profile as read_reduced_profile,
)


ORACLE_DIR = Path(__file__).parents[1] / "oracles" / "briowu"
ORACLES_DIR = ORACLE_DIR.parent


class BriowuOracleTests(unittest.TestCase):
    def test_manifest_profiles_and_tools_are_self_consistent(self) -> None:
        manifest = json.loads(
            (ORACLE_DIR / "evolution-manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["execution"]["terminal_step"], 2048)
        self.assertEqual(manifest["execution"]["output_times"], [0.0, 0.1, 0.2])
        self.assertEqual(manifest["execution"]["particle_count"], 50_176)

        reducer = manifest["semantic_artifacts"]["reducer"]
        reducer_path = (ORACLE_DIR / reducer["path"]).resolve()
        self.assertEqual(
            hashlib.sha256(reducer_path.read_bytes()).hexdigest(),
            reducer["sha256"],
        )
        for record in manifest["semantic_artifacts"]["profiles"]:
            path = ORACLE_DIR / record["path"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                record["sha256"],
            )
            rows = read_reduced_profile(path)
            self.assertEqual(len(rows), reducer["rank_bins"])
            self.assertTrue(
                all(
                    row["particle_count"] == reducer["particles_per_bin"]
                    for row in rows
                )
            )

        published = manifest["published_figure_reference"]
        for field in ("table", "extractor", "comparator"):
            path = ORACLE_DIR / published[field]
            if field == "comparator":
                path = ORACLES_DIR / Path(published[field]).name
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                published[f"{field}_sha256"],
            )

    def test_reduced_self_comparison_is_exact(self) -> None:
        profile = read_reduced_profile(ORACLE_DIR / "evolution_t0.2.csv.gz")
        result = compare_reduced(profile, profile)
        for metrics in result["fields"].values():
            self.assertEqual(metrics["mean_absolute_error"], 0.0)
            self.assertEqual(metrics["normalized_l1"], 0.0)
            self.assertEqual(metrics["maximum_absolute_error"], 0.0)

    def test_reduced_comparator_detects_perturbations_and_bad_limits(self) -> None:
        reference = read_reduced_profile(ORACLE_DIR / "evolution_t0.2.csv.gz")
        actual = [dict(row) for row in reference]
        actual[0]["rho"] += 1.0
        result = compare_reduced(actual, reference)
        self.assertGreater(result["fields"]["rho"]["normalized_l1"], 0.0)
        with self.assertRaisesRegex(ValueError, "duplicate limit"):
            parse_limits(["rho=0.1", "rho=0.2"], {"rho"}, "test limit")
        with self.assertRaisesRegex(ValueError, "expected FIELD=VALUE"):
            parse_limits(["unknown=0.1"], {"rho"}, "test limit")

    def test_corrected_c_published_profile_metrics_match_manifest(self) -> None:
        manifest = json.loads(
            (ORACLE_DIR / "evolution-manifest.json").read_text(encoding="utf-8")
        )
        published = manifest["published_figure_reference"]
        result = compare_published(
            read_published_profile(ORACLE_DIR / "evolution_t0.2.csv.gz"),
            read_published_reference(ORACLE_DIR / published["table"]),
        )
        for field, expected in published[
            "corrected_c_terminal_normalized_l1"
        ].items():
            self.assertAlmostEqual(
                result["fields"][field]["normalized_l1"],
                expected,
                delta=1.0e-15,
            )


if __name__ == "__main__":
    unittest.main()
