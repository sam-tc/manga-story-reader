"""Experimental BYTE manga narration classifier using OCR + pretrained YOLO box features.

Run inside the ORIGINAL project .venv (no torch needed; YOLO detections are cached):
    python scripts/train_yolo_narration.py

Uses only development TRAIN sequence annotations to train. Uses held-out
VALIDATION labels only for scoring. Does not modify scripts/predict.py or test output.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from evaluate_grouping import group_fragments
from predict import CACHE, DATASET, ROOT, read_json, select_sequences
from train_text_filter import features

VAL = ROOT / "experiments" / "validation"
VISION = ROOT / "experiments" / "vision_detections"
sys.path.insert(0, str(DATASET))
import score  # noqa: E402


def box_union(fragments):
    return [
        min(item["bbox"][0] for item in fragments),
        min(item["bbox"][1] for item in fragments),
        max(item["bbox"][2] for item in fragments),
        max(item["bbox"][3] for item in fragments),
    ]


def intersection_over_group(group_box, region_box):
    x0 = max(group_box[0], region_box[0])
    y0 = max(group_box[1], region_box[1])
    x1 = min(group_box[2], region_box[2])
    y1 = min(group_box[3], region_box[3])
    overlap = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = max(1.0, (group_box[2] - group_box[0]) *
               (group_box[3] - group_box[1]))
    return overlap / area


def visual_features(group, page, regions):
    text = " ".join(item["text"].strip() for item in group).strip()
    group_box = box_union(group)
    avg_conf = sum(float(item["confidence"]) for item in group) / len(group)
    result = features({"text": text, "bbox": group_box,
                       "confidence": avg_conf}, page["width"], page["height"])
    result["fragment_count"] = min(len(group), 12) / 12

    cx = (group_box[0] + group_box[2]) / 2
    cy = (group_box[1] + group_box[3]) / 2
    for name in ("narration", "bubble", "other", "text", "ui"):
        result[f"yolo:{name}:center_conf"] = 0.0
        result[f"yolo:{name}:overlap_conf"] = 0.0
        result[f"yolo:{name}:max_overlap"] = 0.0

    for region in regions:
        name = region["class"]
        if name not in ("narration", "bubble", "other", "text", "ui"):
            continue
        x0, y0, x1, y1 = region["bbox"]
        confidence = float(region["confidence"])
        fraction = intersection_over_group(group_box, region["bbox"])
        result[f"yolo:{name}:max_overlap"] = max(
            result[f"yolo:{name}:max_overlap"], fraction
        )
        result[f"yolo:{name}:overlap_conf"] = max(
            result[f"yolo:{name}:overlap_conf"], fraction * confidence
        )
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            result[f"yolo:{name}:center_conf"] = max(
                result[f"yolo:{name}:center_conf"], confidence
            )
    return result


def load_vision(split, selected):
    payload = read_json(VISION / f"{split}.json")
    assert payload["split"] == split
    result = {row["sequence_id"]: row["pages"] for row in payload["sequences"]}
    assert set(result) == {row["sequence_id"] for row in selected}
    for sequence in selected:
        pages = result[sequence["sequence_id"]]
        assert len(pages) == len(sequence["images"]) == 3
        for index, p in enumerate(pages):
            assert p["image"] == sequence["images"][index], (
                "Vision/image mismatch", sequence["sequence_id"], index)
    return result


def make_grouped_data(split, model_package):
    selected = select_sequences("val") if split == "val" else [
        row for row in read_json(DATASET / "sequences.json")
        if row["split"] == "development" and row["sequence_id"] in
        set(read_json(ROOT / "experiments/splits/train_ids.json"))
    ]
    assert len(selected) == (16 if split == "val" else 64)
    vision = load_vision(split, selected)
    text_model = model_package["model"]
    vectorizer = model_package["vectorizer"]
    idx = list(text_model.classes_).index(1)
    threshold = float(model_package["threshold"])
    output = {}

    for sequence in selected:
        sid = sequence["sequence_id"]
        pages = []
        for page_number in range(1, 4):
            page = read_json(CACHE / "development" / sid / f"{page_number:02d}.json")
            region_page = vision[sid][page_number - 1]
            assert int(region_page["width"]) == int(page["width"])
            assert int(region_page["height"]) == int(page["height"])
            detections = page["detections"]
            if detections:
                X = vectorizer.transform([
                    features(d, page["width"], page["height"]) for d in detections
                ])
                probs = text_model.predict_proba(X)[:, idx]
                kept = [d for d, probability in zip(detections, probs)
                        if probability >= threshold and d["text"].strip()]
            else:
                kept = []
            groups = group_fragments(kept, page["height"])
            pages.append([{
                "text": " ".join(d["text"].strip() for d in group).strip(),
                "features": visual_features(group, page, region_page["boxes"]),
            } for group in groups])
        output[sid] = pages
    return output


def weak_labels(gold_page, groups):
    """Use train annotation to label only groups reliably aligned to references.

    The scorer aligns tokens within each page; a unique synthetic speaker tag
    allows us to map matched tokens back to their OCR group. Ambiguous groups
    are skipped rather than assigned possibly false labels.
    """
    predicted = [{"speaker": f"group_{i}", "text": item["text"]}
                 for i, item in enumerate(groups)]
    matched = defaultdict(lambda: Counter())
    for gold, pred, weight in score.align_tokens(
        score.tokens(gold_page), score.tokens(predicted)
    ):
        if gold[1] is None or gold[1] == "UNKNOWN":
            continue
        label = "narration" if gold[1] == "NARRATION" else "other"
        matched[pred[1]][label] += weight
    labels = []
    for i, item in enumerate(groups):
        counts = matched[f"group_{i}"]
        total = counts.total()
        length = max(len(score.tokens([{"speaker": "x", "text": item["text"]}])), 1)
        if total < 2 or total / length < 0.50:
            labels.append(None)
            continue
        majority, weight = counts.most_common(1)[0]
        if weight / total < 0.85:
            labels.append(None)
        else:
            labels.append(1 if majority == "narration" else 0)
    return labels


def generate_candidate(rows, probabilities, threshold):
    output = {}
    selected_count = 0
    for sid, pages in rows.items():
        t = 0
        new_pages = []
        for page_number, page in enumerate(pages):
            pred_page = []
            for i, item in enumerate(page):
                narr = probabilities[sid][page_number][i] >= threshold
                if narr:
                    speaker = "NARRATION"
                    selected_count += 1
                else:
                    speaker = f"character{t % 2 + 1}"
                    t += 1  # narrator text is NOT counted as a character turn
                pred_page.append({"speaker": speaker, "text": item["text"]})
            new_pages.append(pred_page)
        output[sid] = new_pages
    return output, selected_count


def save_jsonl(path, records):
    with path.open("w", encoding="utf-8") as out:
        for sid, pages in records.items():
            out.write(json.dumps({"sequence_id": sid, "pages": pages},
                                 ensure_ascii=False) + "\n")


def main():
    text_model = joblib.load(ROOT / "models/text_filter.joblib")
    print("Reconstructing train/validation OCR groups using saved text filter...")
    train = make_grouped_data("train", text_model)
    val = make_grouped_data("val", text_model)
    baseline = score.load_jsonl(VAL / "pipeline_predictions.jsonl")
    empty_probs = {sid: [[0.0] * len(page) for page in pages]
                   for sid, pages in val.items()}
    recon, _ = generate_candidate(val, empty_probs, 0.50)
    assert recon == baseline, "Baseline reconstruction failed; stop and investigate"
    print("Original validation text/order/speaker assignments: PASS")

    gold_train = score.load_jsonl(DATASET / "development/labels.jsonl", reference=True)
    X, y = [], []
    skipped = 0
    for sid, pages in train.items():
        for page_index, groups in enumerate(pages):
            labels = weak_labels(gold_train[sid][page_index], groups)
            for item, label in zip(groups, labels):
                if label is None:
                    skipped += 1
                else:
                    X.append(item["features"])
                    y.append(label)
    print("Reliable training groups:", len(y))
    print("Narration training groups:", sum(y))
    print("Other training groups:", len(y) - sum(y))
    print("Ambiguous/unmatched skipped:", skipped)
    if sum(y) < 12 or len(y) - sum(y) < 20:
        raise RuntimeError("Too few confidently labeled training examples; inspect matching")

    vectorizer = DictVectorizer(sparse=True)
    Xt = vectorizer.fit_transform(X)
    model = LogisticRegression(class_weight="balanced", max_iter=2000,
                               random_state=42)
    model.fit(Xt, y)
    print("Trained narration classifier with", Xt.shape[1], "features")

    probabilities = {}
    pos = list(model.classes_).index(1)
    for sid, pages in val.items():
        probabilities[sid] = []
        for groups in pages:
            if groups:
                matrix = vectorizer.transform([g["features"] for g in groups])
                probabilities[sid].append(model.predict_proba(matrix)[:, pos].tolist())
            else:
                probabilities[sid].append([])

    gold_val = score.load_jsonl(VAL / "validation_references.jsonl", reference=True)
    original_score = score.evaluate(gold_val, baseline)["macro"]
    print("\nThreshold | Pred narration | False on no-narration sequences | Balanced joint F1 | Delta")
    print("-" * 88)
    results = []
    best = None
    for threshold in (0.35, 0.50, 0.65, 0.80, 0.90, 0.95):
        predictions, count = generate_candidate(val, probabilities, threshold)
        report = score.evaluate(gold_val, predictions)["macro"]
        assert abs(report["text_order_score"] - original_score["text_order_score"]) < 1e-12
        false = sum(
            item["speaker"] == "NARRATION"
            for sid, pages in predictions.items()
            if not any(row["speaker"] == "NARRATION"
                       for page in gold_val[sid] for row in page)
            for page in pages for item in page
        )
        delta = report["balanced_joint_f1"] - original_score["balanced_joint_f1"]
        results.append({"threshold": threshold, "narration_predictions": count,
                        "false_on_no_narration_sequences": false,
                        "balanced_joint_f1": report["balanced_joint_f1"],
                        "text_order_score": report["text_order_score"],
                        "delta": delta})
        print(f"{threshold:9.2f} | {count:14d} | {false:31d} | "
              f"{report['balanced_joint_f1']:.4f}            | {delta:+.4f}")
        if best is None or report["balanced_joint_f1"] > best[0]:
            best = (report["balanced_joint_f1"], threshold, predictions)

    (VAL / "yolo_narration_sweep.json").write_text(
        json.dumps({"baseline": original_score, "results": results,
                    "note": "Exploratory validation tuning; not unseen test performance"},
                   indent=2) + "\n", encoding="utf-8")
    save_jsonl(VAL / "yolo_narration_predictions.jsonl", best[2])
    joblib.dump({"model": model, "vectorizer": vectorizer,
                 "features": "text_filter features + YOLO region overlaps",
                 "threshold_experimental": best[1],
                 "train_groups": len(y), "positive_groups": sum(y)},
                ROOT / "models/yolo_narration_classifier.joblib")
    print("\nSelected exploratory validation threshold:", best[1])
    print("Saved validation sweep, predictions, and experimental model")
    print("Production inference and test submission unchanged")


if __name__ == "__main__":
    main()
