
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
CACHE = ROOT / "experiments" / "ocr_cache"
OUTPUT = ROOT / "experiments" / "validation"
OUTPUT.mkdir(parents=True, exist_ok=True)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def reading_order(detection, page_height):
    """
    Approximate reading order using horizontal bands.
    Within each band, prefer right-to-left positions.
    This is only an initial baseline.
    """
    x0, y0, x1, y1 = detection["bbox"]

    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2

    band_height = max(page_height / 12, 1)
    band = int(center_y / band_height)

    return (band, -center_x, center_y)


def main():
    validation_ids = set(read_json(
        ROOT / "experiments/splits/val_ids.json"
    ))

    manifest = read_json(DATASET / "sequences.json")

    validation_sequences = [
        sequence for sequence in manifest
        if sequence["split"] == "development"
        and sequence["sequence_id"] in validation_ids
    ]

    assert len(validation_sequences) == 16

    predictions = []
    total_entries = 0

    for sequence in validation_sequences:
        sequence_id = sequence["sequence_id"]
        pages = []

        for page_number in range(1, 4):
            cache_path = (
                CACHE / "development" / sequence_id
                / f"{page_number:02d}.json"
            )

            if not cache_path.exists():
                raise FileNotFoundError(
                    f"Missing OCR cache: {cache_path}"
                )

            data = read_json(cache_path)

            detections = sorted(
                data["detections"],
                key=lambda item: reading_order(
                    item, data["height"]
                ),
            )

            page_predictions = []

            for detection in detections:
                text = detection["text"].strip()

                if not text:
                    continue

                page_predictions.append({
                    "speaker": "character1",
                    "text": text,
                })

            total_entries += len(page_predictions)
            pages.append(page_predictions)

        predictions.append({
            "sequence_id": sequence_id,
            "pages": pages,
        })

    predictions_path = OUTPUT / "baseline_predictions.jsonl"

    with predictions_path.open(
        "w", encoding="utf-8"
    ) as file:
        for prediction in predictions:
            file.write(
                json.dumps(prediction, ensure_ascii=False)
                + "\n"
            )

    # Prepare validation references for scoring only.
    # These labels NEVER enter our prediction pipeline.
    references_path = OUTPUT / "validation_references.jsonl"

    reference_count = 0

    with (
        (DATASET / "development/labels.jsonl").open(
            "r", encoding="utf-8"
        ) as source,
        references_path.open(
            "w", encoding="utf-8"
        ) as target,
    ):
        for line in source:
            if not line.strip():
                continue

            item = json.loads(line)

            if item["sequence_id"] in validation_ids:
                target.write(
                    json.dumps(item, ensure_ascii=False)
                    + "\n"
                )
                reference_count += 1

    assert reference_count == 16

    scores_path = OUTPUT / "baseline_scores.json"

    command = [
        sys.executable,
        str(DATASET / "score.py"),
        "--references",
        str(references_path),
        "--predictions",
        str(predictions_path),
        "--output",
        str(scores_path),
    ]

    print("Validation sequences:", len(predictions))
    print("Total predicted entries:", total_entries)
    print("Running official BYTE scorer...\n")

    subprocess.run(command, check=True)

    print("\nPredictions:", predictions_path)
    print("Score report:", scores_path)


if __name__ == "__main__":
    main()
