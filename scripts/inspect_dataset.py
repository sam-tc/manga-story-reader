import json
from pathlib import Path

manifest_path = Path("dataset/sequences.json")

with manifest_path.open("r", encoding="utf-8") as file:
    sequences = json.load(file)

development_count = 0
test_count = 0

for sequence in sequences:
    if sequence["split"] == "development":
        development_count += 1
    elif sequence["split"] == "test":
        test_count += 1

print("Total sequences:", len(sequences))
print("Development sequences:", development_count)
print("Test sequences:", test_count)

print("\nFirst sequence:")
print(json.dumps(sequences[0], indent=2))

image_count = 0
missing_images = []

for sequence in sequences:
    if len(sequence["images"]) != 3:
        raise ValueError(f"{sequence['sequence_id']} must contain exactly 3 pages.")

    for relative_path in sequence["images"]:
        full_path = Path("dataset") / relative_path
        image_count += 1

        if not full_path.is_file():
            missing_images.append(str(full_path))

print("\nImage files listed:", image_count)
print("Missing image files:", len(missing_images))

if missing_images:
    for missing_path in missing_images:
        print(missing_path)
    raise SystemExit("Dataset is incomplete. Restore the missing images.")

print("Every sequence lists 3 pages, and all listed image files exist.")

labels = []

with Path("dataset/development/labels.jsonl").open("r", encoding="utf-8") as file:
    for line in file:
        if line.strip():
            labels.append(json.loads(line))

development_ids = set()
for sequence in sequences:
    if sequence["split"] == "development":
        development_ids.add(sequence["sequence_id"])

label_ids = set()
for label in labels:
    sequence_id = label["sequence_id"]

    if sequence_id in label_ids:
        raise ValueError(f"Duplicate label sequence: {sequence_id}")

    label_ids.add(sequence_id)

    if len(label["pages"]) != 3:
        raise ValueError(f"{sequence_id} must have labels for exactly 3 pages.")

if label_ids != development_ids:
    raise ValueError("Development sequence IDs and label IDs do not match.")

print("\nLabelled sequences:", len(labels))
print("Label IDs match all development sequence IDs.")
print("Every label record contains 3 page entries.")

print("\nFirst two provided story lines from the first labelled page:")
print(json.dumps(labels[0]["pages"][0][:2], indent=2))