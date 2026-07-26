"""Validate the machine-readable correctness finding registry."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

FINDING_ID = re.compile(r"^GZ-\d{4}$")
SHA = re.compile(r"^[0-9a-f]{40}$")
STATUSES = {"confirmed", "fixed", "under_investigation", "rejected"}
CONFIDENCES = {"low", "medium", "high"}
ORACLE_CLASSIFICATIONS = {
    "concurrency_stress": "dynamic",
    "differential": "property",
    "invariant": "property",
    "metamorphic": "property",
    "reference": "exact",
    "regression": "exact",
    "sanitizer": "dynamic",
    "static_analysis": "static",
}
REQUIRED_FINDING_FIELDS = {
    "id",
    "title",
    "status",
    "confidence",
    "affected_flags",
    "summary",
    "impact",
    "reproduction",
    "falsification",
    "oracle",
    "sources",
    "dossier",
}


def classify_oracle_type(oracle_type: str) -> str:
    """Return the broad oracle class for a supported concrete oracle type."""
    try:
        return ORACLE_CLASSIFICATIONS[oracle_type]
    except KeyError as error:
        supported = ", ".join(sorted(ORACLE_CLASSIFICATIONS))
        raise ValueError(
            f"unknown oracle type {oracle_type!r}; expected one of: {supported}"
        ) from error


def _safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_registry(data: Any, repo_root: Path) -> list[str]:
    """Return every registry integrity error without stopping at the first."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["registry root must be an object"]
    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    revision = data.get("pinned_revision")
    if not isinstance(revision, dict):
        errors.append("pinned_revision must be an object")
        repository = ""
        pinned_sha = ""
    else:
        repository = revision.get("repository", "")
        pinned_sha = revision.get("sha", "")
        if not _nonempty_string(repository) or not repository.startswith("https://"):
            errors.append("pinned_revision.repository must be an HTTPS URL")
        if not isinstance(pinned_sha, str) or not SHA.fullmatch(pinned_sha):
            errors.append("pinned_revision.sha must be a lowercase 40-character SHA")

    findings = data.get("findings")
    if not isinstance(findings, list):
        return errors + ["findings must be an array"]

    seen_ids: set[str] = set()
    for index, finding in enumerate(findings):
        label = f"findings[{index}]"
        if not isinstance(finding, dict):
            errors.append(f"{label} must be an object")
            continue
        missing = REQUIRED_FINDING_FIELDS - finding.keys()
        if missing:
            errors.append(f"{label} missing required fields: {', '.join(sorted(missing))}")

        finding_id = finding.get("id")
        if not isinstance(finding_id, str) or not FINDING_ID.fullmatch(finding_id):
            errors.append(f"{label}.id must match GZ-NNNN")
        elif finding_id in seen_ids:
            errors.append(f"duplicate finding id: {finding_id}")
        else:
            seen_ids.add(finding_id)

        for field in (
            "title",
            "summary",
            "impact",
            "reproduction",
            "falsification",
        ):
            if not _nonempty_string(finding.get(field)):
                errors.append(f"{label}.{field} must be a non-empty string")
        if finding.get("status") not in STATUSES:
            errors.append(f"{label}.status is not supported")
        if finding.get("confidence") not in CONFIDENCES:
            errors.append(f"{label}.confidence is not supported")

        flags = finding.get("affected_flags")
        if (
            not isinstance(flags, list)
            or not flags
            or any(not _nonempty_string(flag) for flag in flags)
        ):
            errors.append(f"{label}.affected_flags must contain non-empty strings")
        elif len(flags) != len(set(flags)):
            errors.append(f"{label}.affected_flags contains duplicates")

        oracle = finding.get("oracle")
        if not isinstance(oracle, dict):
            errors.append(f"{label}.oracle must be an object")
        else:
            oracle_type = oracle.get("type")
            try:
                expected_class = classify_oracle_type(oracle_type)
            except ValueError as error:
                errors.append(f"{label}.oracle.type: {error}")
            else:
                if oracle.get("classification") != expected_class:
                    errors.append(
                        f"{label}.oracle.classification must be {expected_class!r}"
                    )
            if not _nonempty_string(oracle.get("rationale")):
                errors.append(f"{label}.oracle.rationale must be a non-empty string")

        dossier = finding.get("dossier")
        if not _safe_relative_path(dossier):
            errors.append(f"{label}.dossier must be a safe relative path")
        else:
            expected_dossier = f"audit/findings/{finding_id}.md"
            if dossier != expected_dossier:
                errors.append(f"{label}.dossier must be {expected_dossier}")
            elif not (repo_root / dossier).is_file():
                errors.append(f"{label}.dossier does not exist: {dossier}")

        sources = finding.get("sources")
        if not isinstance(sources, list) or not sources:
            errors.append(f"{label}.sources must be a non-empty array")
            continue
        for source_index, source in enumerate(sources):
            source_label = f"{label}.sources[{source_index}]"
            if not isinstance(source, dict):
                errors.append(f"{source_label} must be an object")
                continue
            path = source.get("path")
            start = source.get("line_start")
            end = source.get("line_end")
            url = source.get("url")
            if not _safe_relative_path(path):
                errors.append(f"{source_label}.path must be a safe relative path")
                continue
            if not (repo_root / path).is_file():
                errors.append(f"{source_label}.path does not exist: {path}")
            if not isinstance(start, int) or isinstance(start, bool) or start < 1:
                errors.append(f"{source_label}.line_start must be a positive integer")
            if not isinstance(end, int) or isinstance(end, bool) or end < 1:
                errors.append(f"{source_label}.line_end must be a positive integer")
            if isinstance(start, int) and isinstance(end, int) and end < start:
                errors.append(f"{source_label} line range is reversed")
            if (
                _nonempty_string(repository)
                and isinstance(pinned_sha, str)
                and SHA.fullmatch(pinned_sha)
                and isinstance(start, int)
                and isinstance(end, int)
            ):
                fragment = f"#L{start}" if start == end else f"#L{start}-L{end}"
                expected_url = (
                    f"{repository.rstrip('/')}/blob/{pinned_sha}/{path}{fragment}"
                )
                if url != expected_url:
                    errors.append(
                        f"{source_label}.url must pin the registry SHA and exact lines"
                    )
    return errors


def load_and_validate(path: Path, repo_root: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return validate_registry(json.load(handle), repo_root)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "registry",
        nargs="?",
        type=Path,
        default=repo_root / "audit" / "findings.json",
    )
    args = parser.parse_args()
    errors = load_and_validate(args.registry, repo_root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"validated {args.registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
