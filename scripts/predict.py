import argparse
import json
from pathlib import Path

import joblib
from rapidocr import RapidOCR

from batch_ocr import extract_page
from evaluate_grouping import group_fragments
from train_text_filter import features

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
CACHE = ROOT / "experiments" / "ocr_cache"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def select_sequences(split):
    manifest = read_json(DATASET / "sequences.json")

    if split == "val":
        ids = set(read_json(
            ROOT / "experiments/splits/val_ids.json"
        ))

        selected = [
            sequence for sequence in manifest
            if sequence["split"] == "development"
            and sequence["sequence_id"] in ids
        ]

        assert len(selected) == 16
        assert {s["sequence_id"] for s in selected} == ids

        return selected

    # Preserve the official sample submission's ID order.
    sample = read_jsonl(
        DATASET / "sample_submission.jsonl"
    )

    sample_ids = [
        row["sequence_id"] for row in sample
    ]

    assert len(sample_ids) == 15
    assert len(set(sample_ids)) == 15

    by_id = {
        s["sequence_id"]: s
        for s in manifest
        if s["split"] == "test"
    }

    assert set(sample_ids) == set(by_id)

    return [by_id[sequence_id] for sequence_id in sample_ids]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["val", "test"],
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    output_path = args.output

    if output_path is None:
        if args.split == "test":
            output_path = (
                ROOT / "submissions/test_predictions.jsonl"
            )
        else:
            output_path = (
                ROOT / "experiments/validation/"
                "pipeline_predictions.jsonl"
            )

    protected_outputs = {
        (ROOT / "submissions/test_predictions.jsonl").resolve(),
        (ROOT / "submissions/test_predictions_panel.jsonl").resolve(),
    }

    if output_path.resolve() in protected_outputs:
        parser.error(
            "Refusing to overwrite a verified submission. "
            "Use --output with a different filename."
        )

    # Load our locally trained model.
    package = joblib.load(
        ROOT / "models/text_filter.joblib"
    )

    model = package["model"]
    vectorizer = package["vectorizer"]
    threshold = package["threshold"]
    positive_index = list(model.classes_).index(1)

    sequences = select_sequences(args.split)

    # Load OCR lazily, only when cache is unavailable.
    engine = None

    predictions = []

    for sequence in sequences:
        sequence_id = sequence["sequence_id"]
        pages = []

        # Speaker labels remain consistent across all 3 pages.
        turn_index = 0

        for page_number, relative_path in enumerate(
            sequence["images"], start=1
        ):
            cache_path = (
                CACHE
                / sequence["split"]
                / sequence_id
                / f"{page_number:02d}.json"
            )

            if cache_path.exists():
                page = read_json(cache_path)
                source = "cache"
            else:
                if engine is None:
                    engine = RapidOCR(
                        params={"Rec.lang_type": "en"}
                    )

                image_path = DATASET / relative_path

                if not image_path.exists():
                    raise FileNotFoundError(image_path)

                page = extract_page(engine, image_path)

                cache_path.parent.mkdir(
                    parents=True, exist_ok=True
                )

                cache_path.write_text(
                    json.dumps(page, indent=2),
                    encoding="utf-8",
                )

                source = "OCR"

            detections = page["detections"]

            if detections:
                feature_rows = [
                    features(
                        item,
                        page["width"],
                        page["height"],
                    )
                    for item in detections
                ]

                X = vectorizer.transform(feature_rows)

                probabilities = model.predict_proba(X)[
                    :, positive_index
                ]

                kept = [
                    detection
                    for detection, probability in zip(
                        detections, probabilities
                    )
                    if probability >= threshold
                    and detection["text"].strip()
                ]
            else:
                kept = []

            groups = group_fragments(
                kept, page["height"]
            )

            page_predictions = []

            for group in groups:
                text = " ".join(
                    item["text"].strip()
                    for item in group
                ).strip()

                if not text:
                    continue

                # Current best measured speaker baseline.
                speaker = f"character{turn_index % 2 + 1}"
                turn_index += 1

                page_predictions.append({
                    "speaker": speaker,
                    "text": text,
                })

            pages.append(page_predictions)

            print(
                f"{sequence_id} page {page_number}: "
                f"{len(detections)} detections, "
                f"{len(page_predictions)} groups "
                f"({source})"
            )

        assert len(pages) == 3

        predictions.append({
            "sequence_id": sequence_id,
            "pages": pages,
        })

    assert len(predictions) == len(sequences)
    assert len({
        p["sequence_id"] for p in predictions
    }) == len(sequences)

    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    with output_path.open("w", encoding="utf-8") as file:
        for prediction in predictions:
            file.write(
                json.dumps(
                    prediction,
                    ensure_ascii=False,
                ) + "\n"
            )

    print("\nInference completed.")
    print("Sequences:", len(predictions))
    print("Pages:", len(predictions) * 3)
    print("Output:", output_path)


if __name__ == "__main__":
    main()
