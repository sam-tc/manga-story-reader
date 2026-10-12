
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

manifest_path = ROOT / "dataset" / "sequences.json"
output_dir = ROOT / "experiments" / "splits"
output_dir.mkdir(parents=True, exist_ok=True)

with manifest_path.open("r", encoding="utf-8") as file:
    sequences = json.load(file)

development = [
    sequence
    for sequence in sequences
    if sequence["split"] == "development"
]

assert len(development) == 80

sequence_ids = [
    sequence["sequence_id"]
    for sequence in development
]

assert len(set(sequence_ids)) == 80

# Fixed seed makes the split reproducible.
rng = random.Random(42)
rng.shuffle(sequence_ids)

validation_ids = sorted(sequence_ids[:16])
training_ids = sorted(sequence_ids[16:])

assert len(training_ids) == 64
assert len(validation_ids) == 16

assert not (
    set(training_ids) & set(validation_ids)
)

assert (
    set(training_ids) | set(validation_ids)
) == set(sequence_ids)

for filename, ids in [
    ("train_ids.json", training_ids),
    ("val_ids.json", validation_ids),
]:
    path = output_dir / filename

    path.write_text(
        json.dumps(ids, indent=2),
        encoding="utf-8"
    )

print("Training sequences:", len(training_ids))
print("Validation sequences:", len(validation_ids))
print("Training pages:", len(training_ids) * 3)
print("Validation pages:", len(validation_ids) * 3)
print("Overlapping sequences:", len(
    set(training_ids) & set(validation_ids)
))
print("Split files saved to:", output_dir)
