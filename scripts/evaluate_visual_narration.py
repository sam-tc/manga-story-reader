"""Experimental caption-box detector using image geometry; does not edit test output.

Run on the 16 held-out development sequences (whose OCR cache must exist).
The observed rounded-vs-rectangular distinction is a hypothesis, not a universal rule.
"""
import copy
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

from evaluate_grouping import group_fragments
from predict import CACHE, DATASET, ROOT, read_json, read_jsonl, select_sequences
from train_text_filter import features

VAL = ROOT / "experiments" / "validation"
sys.path.insert(0, str(DATASET))
import score  # noqa: E402


def detect_caption_rectangles(image_path):
    """Return likely solid rectangular text boxes from a manga page.

    Thresholds are rough geometry-based starting points and should not be
    interpreted as ground truth. Actual image layout may vary significantly.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(image_path)

    height, width = image.shape
    _, dark = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (40 <= w <= width * 0.45 and 55 <= h <= height * 0.55):
            continue

        # A rectangular contour should have four corners and nearly fill
        # its own bounding rectangle. The latter rejects many speech ovals.
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        if len(polygon) != 4:
            continue
        if cv2.contourArea(contour) / (w * h) < 0.92:
            continue

        # Many real caption boxes are white, whereas manga panel frames
        # may enclose large black/gray illustrations.
        pad_x = max(2, int(w * 0.07))
        pad_y = max(2, int(h * 0.07))
        interior = image[y + pad_y:y + h - pad_y,
                         x + pad_x:x + w - pad_x]
        if interior.size == 0 or float(np.mean(interior > 195)) < 0.55:
            continue

        candidates.append((int(x), int(y), int(x + w), int(y + h)))

    # The inner and outer edge of a border often produce duplicate contours.
    deduplicated = []
    for box in sorted(candidates, key=lambda b: -((b[2]-b[0])*(b[3]-b[1]))):
        cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
        if any(
            other[0] <= cx <= other[2]
            and other[1] <= cy <= other[3]
            and abs(other[0]-box[0]) < 10
            and abs(other[1]-box[1]) < 10
            for other in deduplicated
        ):
            continue
        deduplicated.append(box)

    # Avoid mistaking an enclosing panel border for a narration box.
    # A panel containing two distinct caption rectangles is likely a frame.
    filtered = []
    for box in deduplicated:
        inside = sum(
            other != box
            and box[0] <= (other[0] + other[2]) / 2 <= box[2]
            and box[1] <= (other[1] + other[3]) / 2 <= box[3]
            for other in deduplicated
        )
        if inside < 2:
            filtered.append(box)
    return filtered


def group_is_in_caption(group, boxes):
    x0 = min(item["bbox"][0] for item in group)
    y0 = min(item["bbox"][1] for item in group)
    x1 = max(item["bbox"][2] for item in group)
    y1 = max(item["bbox"][3] for item in group)
    x_center, y_center = (x0+x1)/2, (y0+y1)/2
    return any(
        left+2 <= x_center <= right-2
        and top+2 <= y_center <= bottom-2
        for left, top, right, bottom in boxes
    )


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def narration_coverage(references, predictions):
    """Diagnostic: amount of reference narration aligned and correctly labeled."""
    total_gold_tokens = 0
    aligned_narration = 0.0
    correctly_labeled = 0.0
    for sid, gold_pages in references.items():
        pred_pages = predictions[sid]
        for gold, predicted in zip(gold_pages, pred_pages):
            gold_tokens = score.tokens(gold)
            pred_tokens = score.tokens(predicted)
            total_gold_tokens += sum(label == "NARRATION" for _, label in gold_tokens)
            for (gt, gs), (pt, ps), weight in score.align_tokens(gold_tokens, pred_tokens):
                if gs == "NARRATION":
                    aligned_narration += weight
                    if ps == "NARRATION":
                        correctly_labeled += weight
    return {
        "reference_narration_tokens": total_gold_tokens,
        "matched_narration_token_weight": round(aligned_narration, 3),
        "correctly_labeled_narration_weight": round(correctly_labeled, 3),
    }


def main():
    package = joblib.load(ROOT / "models/text_filter.joblib")
    model = package["model"]
    vectorizer = package["vectorizer"]
    threshold = float(package["threshold"])
    positive_index = list(model.classes_).index(1)

    records = []
    selected = select_sequences("val")
    box_count = 0
    tagged_count = 0
    for sequence in selected:
        pages = []
        marked_pages = []
        sequence_id = sequence["sequence_id"]
        turn_index = 0

        for page_num, rel_path in enumerate(sequence["images"], 1):
            cache_path = (
                CACHE / "development" / sequence_id / f"{page_num:02d}.json"
            )
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"Missing OCR cache: {cache_path}. Run batch_ocr.py --split val first."
                )
            page = read_json(cache_path)
            detections = page["detections"]
            probabilities = (
                model.predict_proba(vectorizer.transform([
                    features(d, page["width"], page["height"])
                    for d in detections
                ]))[:, positive_index]
                if detections else []
            )
            kept = [
                d for d, p in zip(detections, probabilities)
                if p >= threshold and d["text"].strip()
            ]
            groups = group_fragments(kept, page["height"])
            boxes = detect_caption_rectangles(DATASET / rel_path)
            box_count += len(boxes)

            predicted_page = []
            marked_page = []
            for group in groups:
                text = " ".join(d["text"].strip() for d in group).strip()
                if not text:
                    continue
                speaker = f"character{turn_index % 2 + 1}"
                turn_index += 1
                predicted_page.append({"speaker": speaker, "text": text})
                in_caption = group_is_in_caption(group, boxes)
                tagged_count += int(in_caption)
                marked_page.append({
                    "speaker": "NARRATION" if in_caption else speaker,
                    "text": text,
                })
            pages.append(predicted_page)
            marked_pages.append(marked_page)

        records.append((sequence_id, pages, marked_pages))

    # A required safety check: if we cannot reproduce the original 16-sequence
    # baseline exactly, do not treat the visual results as comparable.
    archived = {
        row["sequence_id"]: row["pages"]
        for row in read_jsonl(VAL / "pipeline_predictions.jsonl")
    }
    regenerated = {sid: baseline for sid, baseline, _ in records}
    if regenerated != archived:
        raise RuntimeError("Baseline reconstruction FAILED; no scores saved.")
    print("Baseline reconstruction: PASS")

    references = score.load_jsonl(
        VAL / "validation_references.jsonl", reference=True
    )
    baseline_scores = score.evaluate(references, regenerated)["macro"]
    visual = {sid: marked for sid, _, marked in records}
    report = score.evaluate(references, visual)
    new_scores = report["macro"]
    assert abs(new_scores["text_order_score"] - baseline_scores["text_order_score"]) < 1e-12

    print(f"Rectangular caption candidates found: {box_count}")
    print(f"Text groups tagged as narration: {tagged_count}")
    print("\nMetric                    Before    After    Change")
    print("-" * 52)
    for name in ("text_order_score", "balanced_joint_f1", "joint_f1"):
        print(f"{name:25s} {baseline_scores[name]:.4f}    {new_scores[name]:.4f}    {new_scores[name]-baseline_scores[name]:+.4f}")
    print("\nNarration alignment (baseline):", narration_coverage(references, regenerated))
    print("Narration alignment (visual):  ", narration_coverage(references, visual))

    output = VAL / "visual_narration_predictions.jsonl"
    scores = VAL / "visual_narration_scores.json"
    write_jsonl(output, [
        {"sequence_id": sid, "pages": marked}
        for sid, _, marked in records
    ])
    scores.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nSaved:", output)
    print("Saved:", scores)
    print("Experimental only; the test submission is unchanged.")


if __name__ == "__main__":
    main()
