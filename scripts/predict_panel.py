
"""Panel-aware manga inference.

Run validation:
    python scripts/predict_panel.py --split val

The original BYTE submission is never overwritten.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib

from diagnose_speaker_links import (
    find_balloon,
    find_frame,
    union_box,
)
from evaluate_grouping import group_fragments
from evaluate_panel_reading_order import build_predictions
from predict import (
    CACHE,
    DATASET, 
    ROOT,
    read_json,
    read_jsonl,
    select_sequences,
)
from train_text_filter import features

from evaluate_visual_narration import (
    detect_caption_rectangles,
    group_is_in_caption,
)

from collections import Counter

VISION = ROOT / "experiments/vision_detections"
VALIDATION = ROOT / "experiments/validation"


def build_links(sequences, segmentation, baseline):
    """Match OCR text groups to detected balloons and panels."""

    package = joblib.load(
        ROOT / "models/text_filter.joblib"
    )

    model = package["model"]
    vectorizer = package["vectorizer"]
    threshold = float(package["threshold"])

    positive_index = list(model.classes_).index(1)

    links = {}

    for sequence in sequences:
        sid = sequence["sequence_id"]
        linked_pages = []

        for page_idx, relative_path in enumerate(
            sequence["images"]
        ):
            vision_page = segmentation[sid][page_idx]

            if vision_page["image"] != relative_path:
                raise RuntimeError(
                    f"Segmentation image mismatch: {sid}"
                )

            ocr = read_json(
                CACHE
                / sequence["split"]
                / sid
                / f"{page_idx + 1:02d}.json"
            )

            if (
                ocr["width"],
                ocr["height"],
            ) != (
                vision_page["width"],
                vision_page["height"],
            ):
                raise RuntimeError(
                    f"Image dimensions mismatch: {sid}"
                )

            detections = ocr["detections"]

            if detections:
                rows = [
                    features(
                        item,
                        ocr["width"],
                        ocr["height"],
                    )
                    for item in detections
                ]

                X = vectorizer.transform(rows)

                probabilities = model.predict_proba(X)[
                    :, positive_index
                ]

                kept = [
                    item
                    for item, probability in zip(
                        detections, probabilities
                    )
                    if probability >= threshold
                    and item["text"].strip()
                ]
            else:
                kept = []

            groups = group_fragments(
                kept,
                ocr["height"],
            )

            original = baseline[sid][page_idx]

            texts = [
                " ".join(
                    item["text"].strip()
                    for item in group
                ).strip()
                for group in groups
            ]

            if texts != [
                item["text"] for item in original
            ]:
                raise RuntimeError(
                    f"OCR reconstruction mismatch: {sid}"
                )

            regions = vision_page["detections"]

            # Locate potential narration rectangles on this page.
            image_path = DATASET / relative_path
            caption_boxes = detect_caption_rectangles(image_path)

            page_links = []

            for group, prediction in zip(
                groups, original
            ):
                bbox = union_box(group)

                balloon = find_balloon(
                    bbox, regions
                )

                target_bbox = (
                    regions[balloon]["bbox"]
                    if balloon is not None
                    else bbox
                )

                frame = find_frame(
                    target_bbox, regions
                )

                page_links.append({
                    "text": prediction["text"],
                    "baseline_speaker": prediction["speaker"],
                    "ocr_group_bbox": bbox,
                    "balloon_region_index": balloon,
                    "frame_region_index": frame,
                    "visual_caption_candidate": group_is_in_caption(
                        group, caption_boxes
                    ),
                })

            linked_pages.append(page_links)

        links[sid] = linked_pages

    return links



def apply_narration(predictions, links, threshold=0.50):
    """Assign narration only when visual and text evidence agree."""

    package = joblib.load(
        ROOT / "models/narration_detector.joblib"
    )

    model = package["model"]
    positive_index = list(model.classes_).index(1)

    updated = {}
    narration_count = 0

    for sid, pages in predictions.items():
        updated_pages = []

        for page_index, page in enumerate(pages):
            original_links = links[sid][page_index]

            original_counts = Counter(
                item["text"] for item in original_links
            )

            panel_counts = Counter(
                item["text"] for item in page
            )

            if original_counts != panel_counts:
                raise RuntimeError(
                    f"Text mismatch before narration: {sid}"
                )

            # Match by text only when it appears exactly once.
            # Repeated identical text would be ambiguous.
            visual_candidates = {
                item["text"]
                for item in original_links
                if item["visual_caption_candidate"]
                and original_counts[item["text"]] == 1
            }

            texts = [item["text"] for item in page]

            probabilities = (
                model.predict_proba(texts)[:, positive_index]
                if texts else []
            )

            new_page = []

            for item, probability in zip(
                page, probabilities
            ):
                is_narration = (
                    item["text"] in visual_candidates
                    and probability >= threshold
                )

                speaker = (
                    "NARRATION"
                    if is_narration
                    else item["speaker"]
                )

                if is_narration:
                    narration_count += 1

                new_page.append({
                    "speaker": speaker,
                    "text": item["text"],
                })

            updated_pages.append(new_page)

        updated[sid] = updated_pages

    return updated, narration_count


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

    parser.add_argument(
        "--narration",
        action="store_true",
        help="Enable experimental narration classification",
    )    

    args = parser.parse_args()

    if args.output is not None:
        output = args.output
    elif args.split == "val":
        output = VALIDATION / (
            "panel_top_narration_predictions.jsonl"
            if args.narration
            else "panel_top_live_predictions.jsonl"
        )
    else:
        output = ROOT / "submissions" / (
            "test_predictions_panel_narration.jsonl"
            if args.narration
            else "test_predictions_panel.jsonl"
        )

    if args.narration:
        protected_outputs = {
            (VALIDATION / "panel_top_live_predictions.jsonl").resolve(),
            (ROOT / "submissions/test_predictions_panel.jsonl").resolve(),
        }
        if output.resolve() in protected_outputs:
            parser.error(
                "Narration mode cannot overwrite verified predictions"
            )

    protected = (
        ROOT / "submissions/test_predictions.jsonl"
    ).resolve()

    if output.resolve() == protected:
        parser.error(
            "Refusing to overwrite the original submission"
        )

    selected_output = (
        ROOT / "submissions/test_predictions_panel.jsonl"
    ).resolve()

    if output.resolve() == selected_output and selected_output.exists():
        parser.error(
            "Refusing to overwrite the verified panel submission. "
            "Use --output with a different filename."
        )

    sequences = select_sequences(args.split)

    expected_ids = [
        s["sequence_id"] for s in sequences
    ]

    segmentation_path = (
        VISION / f"segmentation_{args.split}.json"
    )

    if not segmentation_path.is_file():
        raise FileNotFoundError(
            f"Missing segmentation: {segmentation_path}"
        )

    data = read_json(segmentation_path)

    if data.get("split") != args.split:
        raise RuntimeError("Wrong segmentation split")

    segmentation = {
        s["sequence_id"]: s["pages"]
        for s in data["sequences"]
    }

    if (
        set(segmentation) != set(expected_ids)
        or any(
            len(pages) != 3
            for pages in segmentation.values()
        )
    ):
        raise RuntimeError(
            "Segmentation sequences do not match"
        )

    # Generate original predictions in a temporary file.
    # This protects the existing BYTE submission.
    with tempfile.TemporaryDirectory(
        prefix="manga-panel-"
    ) as temp:
        baseline_path = (
            Path(temp) / "baseline.jsonl"
        )

        subprocess.run([
            sys.executable,
            str(ROOT / "scripts/predict.py"),
            "--split",
            args.split,
            "--output",
            str(baseline_path),
        ], check=True)

        rows = read_jsonl(baseline_path)

    if [
        row["sequence_id"] for row in rows
    ] != expected_ids:
        raise RuntimeError(
            "Sequence IDs or order do not match"
        )

    baseline = {
        row["sequence_id"]: row["pages"]
        for row in rows
    }

    if args.split == "val":
        saved_baseline = {
            row["sequence_id"]: row["pages"]
            for row in read_jsonl(
                VALIDATION / "pipeline_predictions.jsonl"
            )
        }

        if baseline != saved_baseline:
            raise RuntimeError(
                "Original validation baseline differs"
            )

        print("Original BYTE baseline: PASS")

    links = build_links(
        sequences,
        segmentation,
        baseline,
    )

    predictions, counts = build_predictions(
        baseline,
        links,
        segmentation,
        "panel_top",
    )

    # Verify the improved validation result before
    # writing any output.
    if args.split == "val":
        expected = {
            row["sequence_id"]: row["pages"]
            for row in read_jsonl(
                VALIDATION / "panel_top_predictions.jsonl"
            )
        }

        if predictions != expected:
            raise RuntimeError(
                "Panel-top predictions do not match "
                "the previously tested experiment"
            )

        print("Panel-top experiment: PASS")


    if args.narration:
        predictions, narration_count = apply_narration(
            predictions,
            links,
            threshold=0.50,
        )

        print(
            f"Narration entries assigned: {narration_count}"
        )        

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w", encoding="utf-8"
    ) as file:
        for sid in expected_ids:
            file.write(
                json.dumps({
                    "sequence_id": sid,
                    "pages": predictions[sid],
                }, ensure_ascii=False) + "\n"
            )

    print("\nPanel-aware inference completed.")
    print("Sequences:", len(predictions))
    print("Pages:", len(predictions) * 3)
    print("Changed pages:", counts["changed_pages"])
    print("Output:", output)

    if args.split == "val":
        sys.path.insert(
            0, str(ROOT / "dataset")
        )
        import score

        references = score.load_jsonl(
            VALIDATION / "validation_references.jsonl",
            reference=True,
        )

        metrics = score.evaluate(
            references, predictions
        )["macro"]

        print(
            "Text order:",
            round(metrics["text_order_score"], 4),
        )
        print(
            "Balanced joint F1:",
            round(metrics["balanced_joint_f1"], 4),
        )


if __name__ == "__main__":
    main()
