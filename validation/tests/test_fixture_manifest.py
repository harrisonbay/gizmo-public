from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from validation.fixture_manifest import load_and_validate, validate_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "validation" / "fixtures" / "manifest.json"


class FixtureManifestTests(unittest.TestCase):
    def test_repository_fixtures_match_manifest(self) -> None:
        self.assertEqual(load_and_validate(MANIFEST_PATH), [])

    def test_modified_fixture_fails_digest_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture_root = Path(directory)
            fixture = fixture_root / "case.txt"
            fixture.write_bytes(b"actual")
            data = {
                "schema_version": 1,
                "fixtures": [
                    {
                        "path": "case.txt",
                        "sha256": hashlib.sha256(b"expected").hexdigest(),
                        "size_bytes": len(b"actual"),
                        "finding_ids": ["GZ-0001"],
                        "description": "A deliberately mismatched test fixture.",
                    }
                ],
            }
            errors = validate_manifest(data, fixture_root)
            self.assertIn("fixtures[0].sha256 does not match case.txt", errors)

    def test_fixture_path_cannot_escape_manifest_directory(self) -> None:
        data = {
            "schema_version": 1,
            "fixtures": [
                {
                    "path": "../outside",
                    "sha256": "0" * 64,
                    "size_bytes": 0,
                    "finding_ids": [],
                    "description": "Unsafe path.",
                }
            ],
        }
        errors = validate_manifest(data, MANIFEST_PATH.parent)
        self.assertIn(
            "fixtures[0].path must be a safe path relative to the manifest", errors
        )

    def test_duplicate_fixture_paths_are_rejected(self) -> None:
        with MANIFEST_PATH.open(encoding="utf-8") as handle:
            data = json.load(handle)
        data["fixtures"].append(dict(data["fixtures"][0]))
        errors = validate_manifest(data, MANIFEST_PATH.parent)
        self.assertIn("duplicate fixture path: unknown-ion.json", errors)


if __name__ == "__main__":
    unittest.main()
