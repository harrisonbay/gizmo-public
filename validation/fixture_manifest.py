"""Validate fixture metadata and verify fixture SHA-256 digests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

FINDING_ID = re.compile(r"^GZ-\d{4}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_FIELDS = {"path", "sha256", "size_bytes", "finding_ids", "description"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_fixture_path(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts


def validate_manifest(data: Any, fixture_root: Path) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["manifest root must be an object"]
    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    fixtures = data.get("fixtures")
    if not isinstance(fixtures, list):
        return errors + ["fixtures must be an array"]

    fixture_root = fixture_root.resolve()
    seen_paths: set[str] = set()
    for index, fixture in enumerate(fixtures):
        label = f"fixtures[{index}]"
        if not isinstance(fixture, dict):
            errors.append(f"{label} must be an object")
            continue
        missing = REQUIRED_FIELDS - fixture.keys()
        if missing:
            errors.append(f"{label} missing required fields: {', '.join(sorted(missing))}")
        path_value = fixture.get("path")
        if not _safe_fixture_path(path_value):
            errors.append(f"{label}.path must be a safe path relative to the manifest")
            continue
        if path_value in seen_paths:
            errors.append(f"duplicate fixture path: {path_value}")
        seen_paths.add(path_value)

        fixture_path = (fixture_root / path_value).resolve()
        try:
            fixture_path.relative_to(fixture_root)
        except ValueError:
            errors.append(f"{label}.path escapes the fixture directory")
            continue
        if not fixture_path.is_file():
            errors.append(f"{label}.path does not exist: {path_value}")
            continue

        expected_digest = fixture.get("sha256")
        if not isinstance(expected_digest, str) or not SHA256.fullmatch(expected_digest):
            errors.append(f"{label}.sha256 must be 64 lowercase hexadecimal characters")
        elif sha256_file(fixture_path) != expected_digest:
            errors.append(f"{label}.sha256 does not match {path_value}")

        expected_size = fixture.get("size_bytes")
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
        ):
            errors.append(f"{label}.size_bytes must be a non-negative integer")
        elif fixture_path.stat().st_size != expected_size:
            errors.append(f"{label}.size_bytes does not match {path_value}")

        finding_ids = fixture.get("finding_ids")
        if (
            not isinstance(finding_ids, list)
            or any(
                not isinstance(finding_id, str)
                or not FINDING_ID.fullmatch(finding_id)
                for finding_id in finding_ids
            )
        ):
            errors.append(f"{label}.finding_ids must contain only GZ-NNNN ids")
        elif len(finding_ids) != len(set(finding_ids)):
            errors.append(f"{label}.finding_ids contains duplicates")
        if not isinstance(fixture.get("description"), str) or not fixture[
            "description"
        ].strip():
            errors.append(f"{label}.description must be a non-empty string")
    return errors


def load_and_validate(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return validate_manifest(data, path.parent)


def main() -> int:
    default = Path(__file__).resolve().parent / "fixtures" / "manifest.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=default)
    args = parser.parse_args()
    errors = load_and_validate(args.manifest)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"verified {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
