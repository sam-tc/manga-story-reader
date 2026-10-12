import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "experiments" / "validation"


def load_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return {
            row["sequence_id"]: row["pages"]
            for line in file
            if line.strip()
            for row in [json.loads(line)]
        }


def narration_entries(pages):
    return [
        item
        for page in pages
        for item in page
        if item["speaker"] == "NARRATION"
    ]


def main():
    references = load_jsonl(
        VAL / "validation_references.jsonl"
    )

    predictions = load_jsonl(
        VAL / "visual_narration_predictions.jsonl"
    )

    wrong_on_non_narration = []

    print("Sequence | Actual narration | Predicted narration")
    print("-" * 65)

    for sequence_id, gold_pages in references.items():
        gold = narration_entries(gold_pages)
        predicted = narration_entries(
            predictions[sequence_id]
        )

        print(
            f"{sequence_id} | "
            f"{len(gold)} | {len(predicted)}"
        )

        if not gold:
            for item in predicted:
                wrong_on_non_narration.append(
                    (sequence_id, item["text"])
                )

    print("\nDefinitely incorrect narration labels:")
    print(len(wrong_on_non_narration))

    print("\nExamples:")
    for sequence_id, text in wrong_on_non_narration[:12]:
        print(f"{sequence_id}: {text}")


if __name__ == "__main__":
    main()
