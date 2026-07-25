#!/usr/bin/env python3
"""Fetch and verify explicitly pinned public validation assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(problem: str) -> None:
    problem_dir = ROOT / problem
    manifest_path = problem_dir / "assets.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{manifest_path}: unsupported schema version")

    for asset in manifest["assets"]:
        destination = problem_dir / asset["name"]
        if (
            destination.is_file()
            and destination.stat().st_size == asset["size_bytes"]
            and sha256(destination) == asset["sha256"]
        ):
            print(f"verified {destination}")
            continue

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{asset['name']}.", dir=problem_dir
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with urllib.request.urlopen(asset["url"], timeout=60) as response:
                with temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            if temporary.stat().st_size != asset["size_bytes"]:
                raise ValueError(f"{asset['name']}: byte-size mismatch")
            actual_digest = sha256(temporary)
            if actual_digest != asset["sha256"]:
                raise ValueError(
                    f"{asset['name']}: SHA-256 mismatch: {actual_digest}"
                )
            temporary.replace(destination)
            print(f"fetched and verified {destination}")
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    choices = sorted(path.name for path in ROOT.iterdir() if path.is_dir())
    parser.add_argument("problem", choices=choices)
    args = parser.parse_args()
    fetch(args.problem)


if __name__ == "__main__":
    main()
