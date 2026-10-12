
import json
import subprocess
import sys
from pathlib import Path

import joblib

from evaluate_baseline import reading_order
from train_text_filter import features

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
CACHE = ROOT / "experiments" / "ocr_cache"
OUTPUT = ROOT / "experiments" / "validation"
OUTPUT.mkdir(parents=True, exist_ok=True)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    # Load the model trained on 64 training sequences.
    package = joblib.load(ROOT / "models/text_filter.joblib")

    model = package["model"]
    vectorizer = package["vectorizer"]
    threshold = package["threshold"]

    # Identify the probability column for class 1.
    positive_index = list(model.classes_).index(1)

    validation_ids = set(read_json(
        ROOT / "experiments/splits/val_ids.json"
    ))

    manifest = read_json(DATASET / "sequences.json")

    sequences = [
        item for item in manifest
        if item["split"] == "development"
        and item["sequence_id"] in validation_ids
    ]

    assert len(sequences) == 16

    predictions = []
    total_detections = 0
    kept_detections = 0

    for sequence in sequences:
        sequence_id = sequence["sequence_id"]
        pages = []

        for page_number in range(1, 4):
            cache_path = (
                CACHE / "development" / sequence_id
                / f"{page_number:02d}.json"
            )

            data = read_json(cache_path)

            detections = sorted(
                data["detections"],
                key=lambda item: reading_order(
                    item, data["height"]
                ),
            )

            total_detections += len(detections)

            if not detections:
                pages.append([])
                continue

            # Extract the same features used during training.
            feature_rows = [
                features(
                    item,
                    data["width"],
                    data["height"],
                )
                for item in detections
            ]

            X = vectorizer.transform(feature_rows)

            # Predict probability of being story text.
            probabilities = model.predict_proba(X)[
                :, positive_index
            ]

            page_predictions = []

            for detection, probability in zip(
                detections, probabilities
            ):
                if probability < threshold:
                    continue

                text = detection["text"].strip()

                if not text:
                    continue

                page_predictions.append({
                    "speaker": "character1",
                    "text": text,
                })

                kept_detections += 1

            pages.append(page_predictions)

        predictions.append({
            "sequence_id": sequence_id,
            "pages": pages,
        })

    # Check that every validation sequence is included.
    assert {
        row["sequence_id"] for row in predictions
    } == validation_ids

    predictions_path = OUTPUT / "filter_predictions.jsonl"

    with predictions_path.open("w", encoding="utf-8") as file:
        for row in predictions:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    # References are used only by the scorer.
    references_path = OUTPUT / "validation_references.jsonl"

    if not references_path.exists():
        raise FileNotFoundError(
            "Run evaluate_baseline.py first to "
            "prepare validation references."
        )

    scores_path = OUTPUT / "filter_scores.json"

    print("Validation sequences:", len(predictions))
    print("Total OCR detections:", total_detections)
    print("Kept by classifier:", kept_detections)
    print("Rejected:", total_detections - kept_detections)
    print("Threshold:", threshold)

    print("\nRunning official BYTE scorer...\n")

    subprocess.run([
        sys.executable,
        str(DATASET / "score.py"),
        "--references",
        str(references_path),
        "--predictions",
        str(predictions_path),
        "--output",
        str(scores_path),
    ], check=True)

    baseline = read_json(
        OUTPUT / "baseline_scores.json"
    )["macro"]

    current = read_json(scores_path)["macro"]

    print("\nCOMPARISON")
    print("-" * 65)

    for metric in [
        "text_order_score",
        "balanced_joint_f1",
        "matched_token_coverage",
        "joint_f1",
    ]:
        old = baseline[metric]
        new = current[metric]

        if old is None or new is None:
            print(f"{metric}: cannot compare")
            continue

        change = new - old

        print(
            f"{metric}: "
            f"baseline={old:.4f}  "
            f"trained={new:.4f}  "
            f"change={change:+.4f}"
        )

    print("\nSaved predictions:", predictions_path)
    print("Saved scores:", scores_path)


if __name__ == "__main__":
    main()
