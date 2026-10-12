
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "experiments" / "validation"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


references = load_jsonl(
    VAL / "validation_references.jsonl"
)

single = load_json(
    VAL / "grouped_scores.json"
)["sequences"]

alternating = load_json(
    VAL / "alternate_2_speakers_scores.json"
)["sequences"]

results = []
categories = defaultdict(list)

for sequence in references:
    sequence_id = sequence["sequence_id"]

    entries = [
        entry
        for page in sequence["pages"]
        for entry in page
    ]

    speakers = {
        entry["speaker"]
        for entry in entries
        if entry["speaker"] not in (
            "NARRATION", "UNKNOWN"
        )
    }

    speaker_count = len(speakers)

    narration_count = sum(
        entry["speaker"] == "NARRATION"
        for entry in entries
    )

    old_f1 = (
        single[sequence_id]["balanced_joint_f1"]
        or 0.0
    )

    new_f1 = (
        alternating[sequence_id]["balanced_joint_f1"]
        or 0.0
    )

    row = {
        "id": sequence_id,
        "speakers": speaker_count,
        "dialogues": len(entries),
        "narration": narration_count,
        "single_f1": old_f1,
        "alternate_f1": new_f1,
        "change": new_f1 - old_f1,
    }

    results.append(row)

    category = (
        "1 speaker" if speaker_count == 1
        else "2 speakers" if speaker_count == 2
        else "3+ speakers" if speaker_count >= 3
        else "0 identifiable speakers"
    )

    categories[category].append(row)


print("SPEAKER DISTRIBUTION")
print("-" * 55)

for category, rows in sorted(categories.items()):
    average = sum(
        row["alternate_f1"] for row in rows
    ) / len(rows)

    print(
        f"{category}: {len(rows)} sequences, "
        f"average F1 = {average:.4f}"
    )


print("\nWORST SEQUENCES")
print("-" * 55)

for row in sorted(
    results,
    key=lambda r: r["alternate_f1"]
)[:6]:
    print(
        f"{row['id']} | "
        f"speakers={row['speakers']} | "
        f"lines={row['dialogues']} | "
        f"narration={row['narration']} | "
        f"F1={row['alternate_f1']:.4f}"
    )


print("\nBIGGEST DECLINES FROM ALTERNATING")
print("-" * 55)

for row in sorted(
    results,
    key=lambda r: r["change"]
)[:5]:
    print(
        f"{row['id']} | "
        f"change={row['change']:+.4f} | "
        f"speakers={row['speakers']}"
    )
