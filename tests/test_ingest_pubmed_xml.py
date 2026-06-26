"""Unit tests for sentence splitting and deterministic ID generation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Allow importing scripts as modules in test runs from repository root.
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from ingest_pubmed_xml import (  # noqa: E402
    make_child_id,
    make_parent_id,
    make_qdrant_point_id,
    split_sentences,
)


class TestSentenceSplitting(unittest.TestCase):
    def test_split_sentences_breaks_on_terminal_punctuation(self) -> None:
        paragraph = (
            "Acute kidney injury is common in critical care. "
            "Biomarker-guided assessment improved early risk stratification."
        )

        result = split_sentences(paragraph)

        self.assertEqual(
            result,
            [
                "Acute kidney injury is common in critical care.",
                "Biomarker-guided assessment improved early risk stratification.",
            ],
        )

    def test_split_sentences_filters_short_fragments(self) -> None:
        paragraph = (
            "Short one. "
            "This longer sentence should remain in the output because it has enough characters. "
            "Tiny."
        )

        result = split_sentences(paragraph)

        self.assertEqual(
            result,
            [
                "This longer sentence should remain in the output because it has enough characters.",
            ],
        )


class TestDeterministicIds(unittest.TestCase):
    def test_make_parent_id_is_deterministic(self) -> None:
        self.assertEqual(make_parent_id("31452104", 2), "31452104:p2")
        self.assertEqual(make_parent_id("31452104", 2), "31452104:p2")

    def test_make_child_id_is_deterministic(self) -> None:
        self.assertEqual(make_child_id("31452104:p2", 1), "31452104:p2:c1")
        self.assertEqual(make_child_id("31452104:p2", 1), "31452104:p2:c1")

    def test_make_qdrant_point_id_is_deterministic(self) -> None:
        first = make_qdrant_point_id("31452104:p2", 1)
        second = make_qdrant_point_id("31452104:p2", 1)
        different = make_qdrant_point_id("31452104:p2", 2)

        self.assertEqual(first, second)
        self.assertNotEqual(first, different)


if __name__ == "__main__":
    unittest.main()
