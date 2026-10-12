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