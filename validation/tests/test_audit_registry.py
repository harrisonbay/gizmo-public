from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from validation.audit_registry import (
    REQUIRED_FINDING_FIELDS,
    classify_oracle_type,
    validate_registry,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "audit" / "findings.json"


def registry() -> dict:
    with REGISTRY_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


class OracleClassificationTests(unittest.TestCase):
    def test_oracle_types_have_expected_broad_classes(self) -> None:
        expected = {
            "concurrency_stress": "dynamic",
            "differential": "property",
            "invariant": "property",
            "metamorphic": "property",
            "reference": "exact",
            "regression": "exact",
            "sanitizer": "dynamic",
            "static_analysis": "static",
        }
        for oracle_type, classification in expected.items():
            with self.subTest(oracle_type=oracle_type):
                self.assertEqual(classify_oracle_type(oracle_type), classification)

    def test_unknown_oracle_type_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown oracle type"):
            classify_oracle_type("looks_about_right")


class RegistryIntegrityTests(unittest.TestCase):
    def test_living_registry_is_valid(self) -> None:
        self.assertEqual(validate_registry(registry(), REPO_ROOT), [])

    def test_schema_documents_are_valid_json(self) -> None:
        for path in (
            REPO_ROOT / "audit" / "finding.schema.json",
            REPO_ROOT / "validation" / "fixture-manifest.schema.json",
        ):
            with self.subTest(path=path):
                with path.open(encoding="utf-8") as handle:
                    self.assertIsInstance(json.load(handle), dict)

    def test_finding_ids_are_unique(self) -> None:
        data = registry()
        data["findings"].append(copy.deepcopy(data["findings"][0]))
        errors = validate_registry(data, REPO_ROOT)
        self.assertIn("duplicate finding id: GZ-0001", errors)

    def test_every_required_field_is_enforced(self) -> None:
        baseline = registry()
        for field in REQUIRED_FINDING_FIELDS:
            with self.subTest(field=field):
                data = copy.deepcopy(baseline)
                del data["findings"][0][field]
                errors = validate_registry(data, REPO_ROOT)
                self.assertTrue(
                    any("missing required fields" in error and field in error for error in errors),
                    errors,
                )

    def test_source_links_must_pin_revision_and_lines(self) -> None:
        data = registry()
        data["findings"][0]["sources"][0]["url"] = (
            "https://github.com/pfhopkins/gizmo-public/blob/main/gravity/gravtree.c"
        )
        errors = validate_registry(data, REPO_ROOT)
        self.assertTrue(any("must pin the registry SHA" in error for error in errors))

    def test_declared_oracle_class_must_match_type(self) -> None:
        data = registry()
        data["findings"][0]["oracle"]["classification"] = "exact"
        errors = validate_registry(data, REPO_ROOT)
        self.assertTrue(any("classification must be 'property'" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
