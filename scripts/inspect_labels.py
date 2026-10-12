import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
LABELS_PATH = ROOT / "dataset" / "development" / "labels.jsonl"

with LABELS_PATH.open("r", encoding="utf-8") as file:
    sequences = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

speaker_counts = Counter()
total_pages = 0
total_text_entries = 0
empty_pages = 0

for sequence in sequences:
    assert "sequence_id" in sequence
    assert len(sequence["pages"]) == 3

    for page in sequence["pages"]:
        total_pages += 1

        if not page:
            empty_pages += 1

        for entry in page:
            assert "speaker" in entry
            assert "text" in entry

            speaker_counts[entry["speaker"]] += 1
            total_text_entries += 1

print("Development sequences:", len(sequences))
print("Total pages:", total_pages)
print("Total text entries:", total_text_entries)
print("Empty pages:", empty_pages)

print("\nSpeaker label counts:")
for speaker, count in speaker_counts.most_common():
    print(f"{speaker}: {count}")

print("\nFirst sequence example:")
first = sequences[0]
print("Sequence ID:", first["sequence_id"])

for page_number, page in enumerate(first["pages"], start=1):
    print(f"\nPage {page_number}")

    for entry in page[:3]:
        print(f"  {entry['speaker']}: {entry['text']}")
