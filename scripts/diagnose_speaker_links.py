"""Inspect OCR-to-balloon-to-panel-to-character geometry on BYTE validation pages.

No reference labels are used to choose matches. No speaker predictions are changed.
Run in the original .venv:
    python scripts/diagnose_speaker_links.py
"""
import json
from collections import Counter
from math import hypot
from pathlib import Path

import joblib
from evaluate_grouping import group_fragments
from predict import CACHE, ROOT, read_json, read_jsonl, select_sequences
from train_text_filter import features

VAL = ROOT / "experiments/validation"
SEGMENTATION = ROOT / "experiments/vision_detections/segmentation_val.json"
BASELINE = VAL / "pipeline_predictions.jsonl"
OUTPUT = VAL / "speaker_links_diagnostic.json"


def union_box(items):
    return [
        min(item["bbox"][0] for item in items),
        min(item["bbox"][1] for item in items),
        max(item["bbox"][2] for item in items),
        max(item["bbox"][3] for item in items),
    ]


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def box_coverage(inner, outer):
    x0, y0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x1, y1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    overlap = max(0, x1 - x0) * max(0, y1 - y0)
    return overlap / max(area(inner), 1)


def inside_polygon(x, y, points):
    """Ray-casting point-in-polygon; works with the export's pixel coordinates."""
    if len(points) < 3:
        return False
    inside = False
    j = len(points) - 1
    for i in range(len(points)):
        xi, yi = points[i]
        xj, yj = points[j]
        if (yi > y) != (yj > y):
            hit_x = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < hit_x:
                inside = not inside
        j = i
    return inside


def region_contains_point(region, x, y):
    x0, y0, x1, y1 = region["bbox"]
    if not (x0 <= x <= x1 and y0 <= y <= y1):
        return False
    polygon = region.get("polygon") or []
    return inside_polygon(x, y, polygon) if len(polygon) >= 3 else True


def find_balloon(group_box, regions):
    gx, gy = center(group_box)
    matches = []
    for index, region in enumerate(regions):
        if region["class"] != "speech_bubble":
            continue
        coverage = box_coverage(group_box, region["bbox"])
        contains = region_contains_point(region, gx, gy)
        if not contains and coverage < 0.65:
            continue
        # Prefer matching the group's center, strong coverage, and more
        # compact regions rather than an enormous enclosing false positive.
        compactness = min(1.0, area(group_box) / max(area(region["bbox"]), 1))
        rank = (int(contains), round(coverage, 4), round(compactness, 4),
                float(region["confidence"]))
        matches.append((rank, index))
    if not matches:
        return None
    return max(matches)[1]


def find_frame(target_box, regions):
    x, y = center(target_box)
    options = []
    for index, region in enumerate(regions):
        if region["class"] != "frame":
            continue
        if region_contains_point(region, x, y) or box_coverage(target_box, region["bbox"]) >= 0.7:
            options.append((area(region["bbox"]), index))
    # A nested small frame wins instead of the entire page-sized region.
    return min(options)[1] if options else None


def edge_distance(a, b):
    """Shortest straight-line separation between two bounding rectangles."""
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return hypot(dx, dy)


def character_candidates(frame_index, balloon_index, regions):
    if frame_index is None or balloon_index is None:
        return []
    frame = regions[frame_index]
    balloon = regions[balloon_index]
    diagonal = max(hypot(frame["bbox"][2] - frame["bbox"][0],
                         frame["bbox"][3] - frame["bbox"][1]), 1)
    candidates = []
    for index, character in enumerate(regions):
        if character["class"] != "character":
            continue
        x, y = center(character["bbox"])
        belongs = (region_contains_point(frame, x, y)
                   or box_coverage(character["bbox"], frame["bbox"]) >= 0.35)
        if not belongs:
            continue
        distance = edge_distance(balloon["bbox"], character["bbox"]) / diagonal
        candidates.append({
            "region_index": index,
            "distance_from_balloon": round(distance, 4),
            "confidence": character["confidence"],
            "bbox": character["bbox"],
        })
    # This is a CANDIDATE ranking, not a verified speaker assignment.
    return sorted(candidates, key=lambda c: (c["distance_from_balloon"], -c["confidence"]))


def main():
    segmentation = read_json(SEGMENTATION)
    if segmentation.get("split") != "val":
        raise RuntimeError("Expected validation detections")
    sequences = select_sequences("val")
    vision = {seq["sequence_id"]: seq["pages"]
              for seq in segmentation["sequences"]}
    baseline = {row["sequence_id"]: row["pages"] for row in read_jsonl(BASELINE)}
    ids = {seq["sequence_id"] for seq in sequences}
    if set(vision) != ids or set(baseline) != ids:
        raise RuntimeError("Validation sequence IDs do not match exactly")

    package = joblib.load(ROOT / "models/text_filter.joblib")
    model, vectorizer = package["model"], package["vectorizer"]
    threshold = float(package["threshold"])
    positive_index = list(model.classes_).index(1)

    stats = Counter()
    output_sequences = []
    examples = []
    for seq in sequences:
        sid = seq["sequence_id"]
        if len(vision[sid]) != 3:
            raise RuntimeError(f"Expected three pages: {sid}")
        pages = []
        for page_idx, relative in enumerate(seq["images"]):
            detection_page = vision[sid][page_idx]
            if detection_page["image"] != relative:
                raise RuntimeError(f"Wrong detection image: {sid} page {page_idx+1}")
            ocr_page = read_json(CACHE / "development" / sid / f"{page_idx+1:02d}.json")
            if (ocr_page["width"], ocr_page["height"]) != (detection_page["width"], detection_page["height"]):
                raise RuntimeError(f"Image dimension mismatch: {sid} page {page_idx+1}")
            detections = ocr_page["detections"]
            if detections:
                X = vectorizer.transform([
                    features(item, ocr_page["width"], ocr_page["height"])
                    for item in detections
                ])
                probs = model.predict_proba(X)[:, positive_index]
                kept = [item for item, p in zip(detections, probs)
                        if p >= threshold and item["text"].strip()]
            else:
                kept = []
            groups = group_fragments(kept, ocr_page["height"])
            reconstructed_texts = [" ".join(item["text"].strip() for item in group).strip()
                                   for group in groups]
            baseline_texts = [item["text"] for item in baseline[sid][page_idx]]
            if reconstructed_texts != baseline_texts:
                raise RuntimeError(f"OCR/baseline text mismatch: {sid} page {page_idx+1}")

            regions = detection_page["detections"]
            rows = []
            for idx, group in enumerate(groups):
                stats["groups"] += 1
                group_box = union_box(group)
                bubble_idx = find_balloon(group_box, regions)
                target_box = regions[bubble_idx]["bbox"] if bubble_idx is not None else group_box
                frame_idx = find_frame(target_box, regions)
                if bubble_idx is not None:
                    stats["with_balloon"] += 1
                if frame_idx is not None:
                    stats["with_frame"] += 1
                candidates = character_candidates(frame_idx, bubble_idx, regions)
                if candidates:
                    stats["with_character_candidates"] += 1
                    stats["with_1_character" if len(candidates) == 1 else "with_2plus_characters"] += 1
                elif bubble_idx is not None:
                    stats["balloon_without_character_candidates"] += 1
                row = {
                    "text": reconstructed_texts[idx],
                    "baseline_speaker": baseline[sid][page_idx][idx]["speaker"],
                    "ocr_group_bbox": group_box,
                    "balloon_region_index": bubble_idx,
                    "frame_region_index": frame_idx,
                    "character_candidates": candidates,
                }
                rows.append(row)
                if sid == "seq_ee880b9612205860" and len(examples) < 12:
                    examples.append((page_idx + 1, row))
            pages.append(rows)
        output_sequences.append({"sequence_id": sid, "pages": pages})

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "notes": "Geometry diagnostics only; no speaker labels inferred or changed.",
        "summary": dict(stats),
        "sequences": output_sequences,
    }, indent=2), encoding="utf-8")
    print("Baseline text reconstruction: PASS")
    print("Pages analyzed:", len(sequences) * 3)
    for k in ("groups", "with_balloon", "with_frame", "with_character_candidates",
              "with_1_character", "with_2plus_characters",
              "balloon_without_character_candidates"):
        print(f"{k}: {stats[k]}")
    print("\nExample matches (one validation sequence):")
    for page_no, row in examples:
        chars = row["character_candidates"]
        nearest = f"region {chars[0]['region_index']}" if chars else "none"
        print(f"page {page_no}: balloon={row['balloon_region_index']}, "
              f"frame={row['frame_region_index']}, "
              f"candidate_characters={len(chars)}, closest={nearest}; "
              f"text={row['text'][:65]}")
    print("\nSaved:", OUTPUT)
    print("No test predictions or production code changed.")


if __name__ == "__main__":
    main()
