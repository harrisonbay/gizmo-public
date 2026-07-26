import hashlib
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ORACLES = ROOT / "validation" / "oracles"
BRIOWU = ORACLES / "briowu"
SPEC = importlib.util.spec_from_file_location(
    "compare_briowu_profile", ORACLES / "compare_briowu_profile.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load Brio-Wu profile comparator")
COMPARATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARATOR)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BrioWuOracleRegression(unittest.TestCase):
    def test_corrected_c_terminal_profile_passes_published_figure_gate(self):
        manifest = json.loads((BRIOWU / "evolution-manifest.json").read_text())
        published = manifest["published_figure_reference"]
        result = COMPARATOR.compare(
            COMPARATOR.read_profile(BRIOWU / "evolution_t0.2.csv.gz"),
            COMPARATOR.read_reference(BRIOWU / "published_exact_figure.csv"),
        )
        self.assertEqual(result["profile_samples"], 334)
        self.assertEqual(set(result["fields"]), set(COMPARATOR.PROFILE_COLUMNS))
        gate = published["maximum_normalized_l1_gate"]
        for field, expected in published["corrected_c_terminal_normalized_l1"].items():
            observed = result["fields"][field]["normalized_l1"]
            self.assertAlmostEqual(observed, expected, places=15)
            self.assertLess(observed, gate)

    def test_manifest_pins_every_derived_published_figure_artifact(self):
        manifest = json.loads((BRIOWU / "evolution-manifest.json").read_text())
        published = manifest["published_figure_reference"]
        self.assertEqual(
            sha256(BRIOWU / published["extractor"]),
            published["extractor_sha256"],
        )
        self.assertEqual(
            sha256(BRIOWU / published["table"]),
            published["table_sha256"],
        )
        comparator = (BRIOWU / published["comparator"]).resolve()
        self.assertEqual(sha256(comparator), published["comparator_sha256"])

    def test_t0_phase_is_explicitly_not_the_unevolved_ic(self):
        manifest = json.loads((BRIOWU / "evolution-manifest.json").read_text())
        phase = manifest["execution"]["output_phase"]["snapshot_000"]
        self.assertIn("first half-kick", phase)
        self.assertIn("zero elapsed drift", phase)
        self.assertTrue(
            any(
                "post-first-half-kick" in limitation
                for limitation in manifest["limitations"]
            )
        )


if __name__ == "__main__":
    unittest.main()
