
import json
import subprocess
import sys
from pathlib import Path

import joblib
from train_text_filter import features

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
CACHE = ROOT / "experiments/ocr_cache/development"
OUTPUT = ROOT / "experiments/validation"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def can_follow(above, below):
    """Score whether two OCR fragments belong to consecutive lines."""
    ax0, ay0, ax1, ay1 = above["bbox"]
    bx0, by0, bx1, by1 = below["bbox"]

    ah = max(ay1 - ay0, 1)
    bh = max(by1 - by0, 1)
    aw = max(ax1 - ax0, 1)
    bw = max(bx1 - bx0, 1)

    gap = by0 - ay1

    # The next line must be below the previous one.
    if by0 <= ay0 + 0.45 * min(ah, bh):
        return None

    # Large vertical gaps suggest separate text regions.
    if not (-0.35 * min(ah, bh) <= gap <= 0.85 * max(ah, bh)):
        return None

    overlap_x = max(0, min(ax1, bx1) - max(ax0, bx0))

    if overlap_x / min(aw, bw) < 0.3:
        return None

    center_distance = abs(
        (ax0 + ax1) - (bx0 + bx1)
    ) / 2

    if center_distance > 0.5 * max(aw, bw):
        return None

    return (
        max(gap, 0) / max(ah, bh)
        + center_distance / max(aw, bw)
    )


def group_fragments(fragments, page_height):
    """Group neighboring lines using geometry only."""
    fragments = sorted(
        fragments,
        key=lambda d: (d["bbox"][1], -d["bbox"][0]),
    )

    groups = []

    for fragment in fragments:
        choices = []

        for index, group in enumerate(groups):
            score = can_follow(group[-1], fragment)

            if score is not None:
                choices.append((score, index))

        if choices:
            best_group = min(choices)[1]
            groups[best_group].append(fragment)
        else:
            groups.append([fragment])

    def group_order(group):
        right = max(d["bbox"][2] for d in group)
        top = min(d["bbox"][1] for d in group)
        bottom = max(d["bbox"][3] for d in group)

        center_y = (top + bottom) / 2
        band = int(center_y / max(page_height / 12, 1))

        return (band, -right, top)

    return sorted(groups, key=group_order)


def main():
    package = joblib.load(
        ROOT / "models/text_filter.joblib"
    )

    model = package["model"]
    vectorizer = package["vectorizer"]
    threshold = package["threshold"]

    positive_index = list(model.classes_).index(1)

    val_ids = set(load_json(
        ROOT / "experiments/splits/val_ids.json"
    ))

    sequences = [
        s for s in load_json(DATASET / "sequences.json")
        if s["split"] == "development"
        and s["sequence_id"] in val_ids
    ]

    assert len(sequences) == 16

    predictions = []
    fragments_kept = 0
    groups_created = 0
    combined_groups = 0

    for sequence in sequences:
        pages = []

        for page_number in range(1, 4):
            cache_path = (
                CACHE / sequence["sequence_id"]
                / f"{page_number:02d}.json"
            )

            page = load_json(cache_path)
            detections = page["detections"]

            if detections:
                X = vectorizer.transform([
                    features(
                        d,
                        page["width"],
                        page["height"],
                    )
                    for d in detections
                ])

                probabilities = model.predict_proba(X)[
                    :, positive_index
                ]

                kept = [
                    d for d, probability in zip(
                        detections, probabilities
                    )
                    if probability >= threshold
                    and d["text"].strip()
                ]
            else:
                kept = []

            groups = group_fragments(
                kept, page["height"]
            )

            fragments_kept += len(kept)
            groups_created += len(groups)
            combined_groups += sum(
                len(group) > 1 for group in groups
            )

            pages.append([
                {
                    "speaker": "character1",
                    "text": " ".join(
                        part["text"].strip()
                        for part in group
                    ),
                }
                for group in groups
            ])

        predictions.append({
            "sequence_id": sequence["sequence_id"],
            "pages": pages,
        })

    assert {
        row["sequence_id"] for row in predictions
    } == val_ids

    OUTPUT.mkdir(parents=True, exist_ok=True)

    predictions_path = OUTPUT / "grouped_predictions.jsonl"

    with predictions_path.open("w", encoding="utf-8") as file:
        for row in predictions:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    scores_path = OUTPUT / "grouped_scores.json"

    print("Fragments kept:", fragments_kept)
    print("Dialogue candidates:", groups_created)
    print("Combined groups:", combined_groups)
    print("\nRunning official BYTE scorer...\n")

    subprocess.run([
        sys.executable,
        str(DATASET / "score.py"),
        "--references",
        str(OUTPUT / "validation_references.jsonl"),
        "--predictions",
        str(predictions_path),
        "--output",
        str(scores_path),
    ], check=True)

    old = load_json(
        OUTPUT / "filter_scores.json"
    )["macro"]

    new = load_json(scores_path)["macro"]

    print("\nCOMPARISON")
    print("-" * 55)

    for metric in [
        "text_order_score",
        "balanced_joint_f1",
        "matched_token_coverage",
    ]:
        print(
            f"{metric}: "
            f"{old[metric]:.4f} -> "
            f"{new[metric]:.4f} "
            f"({new[metric] - old[metric]:+.4f})"
        )


if __name__ == "__main__":
    main()
