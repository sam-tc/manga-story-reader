
"""Check the panel-aware BYTE test submission.

Validates structure and confirms that reordering did not
add, delete, or alter recognized text.
"""

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ORIGINAL = ROOT / "submissions/test_predictions.jsonl"
CANDIDATE = ROOT / "submissions/test_predictions_panel.jsonl"
MANIFEST = ROOT / "dataset/sequences.json"


def read_jsonl(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def validate():
    original = read_jsonl(ORIGINAL)
    candidate = read_jsonl(CANDIDATE)

    with MANIFEST.open(encoding="utf-8") as file:
        sequences = json.load(file)

    official_ids = {
        item["sequence_id"]
        for item in sequences
        if item["split"] == "test"
    }

    old_ids = [
        item["sequence_id"]
        for item in original
    ]

    new_ids = [
        item["sequence_id"]
        for item in candidate
    ]

    assert len(original) == 15
    assert len(candidate) == 15
    assert len(set(new_ids)) == 15
    assert new_ids == old_ids, "Sequence order changed"
    assert set(new_ids) == official_ids, (
        "Official test sequence IDs do not match"
    )

    changed_pages = 0
    total_entries = 0
    empty_pages = 0
    narration_entries = 0
    punctuation_entries = 0

    for old_sequence, new_sequence in zip(
        original, candidate
    ):
        assert set(new_sequence) == {
            "sequence_id", "pages"
        }

        old_pages = old_sequence["pages"]
        new_pages = new_sequence["pages"]

        assert len(old_pages) == 3
        assert len(new_pages) == 3

        for old_page, new_page in zip(
            old_pages, new_pages
        ):
            if old_page != new_page:
                changed_pages += 1

            if not new_page:
                empty_pages += 1

            for entry in new_page:
                assert isinstance(entry, dict)
                assert set(entry) == {
                    "speaker", "text"
                }

                speaker = entry["speaker"]
                text = entry["text"]

                assert isinstance(speaker, str)
                assert isinstance(text, str)
                assert speaker.strip()
                assert text.strip()

                total_entries += 1

                if speaker == "NARRATION":
                    narration_entries += 1

                if re.fullmatch(
                    r"[.?!…！？]+", text.strip()
                ):
                    punctuation_entries += 1

            old_texts = Counter(
                entry["text"]
                for entry in old_page
            )

            new_texts = Counter(
                entry["text"]
                for entry in new_page
            )

            assert old_texts == new_texts, (
                "Text was added, lost, or modified"
            )

    print("\nBYTE SUBMISSION VALIDATION")
    print("-" * 42)
    print("JSONL structure: PASS")
    print("Official test sequence IDs: PASS")
    print("Sequence order: PASS")
    print("Three pages per sequence: PASS")
    print("Speaker/text field validation: PASS")
    print("Recognized text preservation: PASS")
    print()
    print("Sequences:", len(candidate))
    print("Pages:", len(candidate) * 3)
    print("Dialogue/text entries:", total_entries)
    print("Changed pages:", changed_pages)
    print("Pages with zero entries:", empty_pages)
    print("NARRATION entries:", narration_entries)
    print(
        "Punctuation-only entries:",
        punctuation_entries
    )
    print()
    print("Original submission preserved:", ORIGINAL)
    print("Panel-aware candidate:", CANDIDATE)
    print("\nAll structural checks passed.")


if __name__ == "__main__":
    validate()
