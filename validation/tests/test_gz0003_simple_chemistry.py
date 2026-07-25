from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def c_function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    opening_brace = source.index("{", start)
    depth = 0
    for index in range(opening_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening_brace + 1 : index]
    raise AssertionError(f"unterminated function: {signature}")


class UnknownIonRegressionTests(unittest.TestCase):
    def test_lookup_has_an_explicit_invalid_ion_path(self) -> None:
        source = (REPO_ROOT / "cooling" / "simple_chemistry.c").read_text(
            encoding="utf-8"
        )
        body = c_function_body(source, "int ion_name_to_index(char *ion_name)")
        self.assertIn("terminate(", body)
        self.assertIn("Unknown ion", body)


if __name__ == "__main__":
    unittest.main()
