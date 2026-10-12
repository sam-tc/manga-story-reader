"""BYTE Manga Task: compare panel-aware reading order against the saved baseline.

Reads only saved validation predictions, speaker-link diagnostics and segmentation.
No model download, OCR rerun, label-dependent prediction or test modification.
Development reference labels are used only by the official scorer at the end.

Run from repository root, in .venv:
    python scripts/evaluate_panel_reading_order.py
"""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "experiments" / "validation"
VISION = ROOT / "experiments" / "vision_detections"
sys.path.insert(0, str(ROOT / "dataset"))
import score  # Official BYTE task scorer


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def center(box):
    return (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))


def sort_items_by_manga_rows(items, *, image_height, bands=6):
    """Approximate row-first/right-to-left order of rectangular items.

    `items` contains (bounding_box, payload) pairs. The number of rows is an
    experimental heuristic, not ground-truth panel information.
    """
    band_h = max(image_height / bands, 1)
    return sorted(items, key=lambda item: (
        int(center(item[0])[1] / band_h),
        -center(item[0])[0],
        item[0][1],
    ))


def reorder_page(base_page, link_page, segment_page, mode):
    if len(base_page) != len(link_page):
        raise ValueError("OCR group / diagnostic count mismatch")
    regions = segment_page["detections"]
    height = segment_page["height"]
    units = defaultdict(list)
    # Frame keys are unique per page. Groups without a detected frame are
    # independent units, not one giant miscellaneous frame.
    for row_idx, (prediction, link) in enumerate(zip(base_page, link_page)):
        if prediction["text"] != link["text"] or prediction["speaker"] != link["baseline_speaker"]:
            raise ValueError("Speaker link diagnostic differs from baseline")
        box = link["ocr_group_bbox"]
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("Missing OCR group coordinates")
        frame_idx = link["frame_region_index"]
        if frame_idx is not None:
            if not (0 <= frame_idx < len(regions)) or regions[frame_idx]["class"] != "frame":
                raise ValueError("Invalid frame index")
            key = ("frame", frame_idx)
        else:
            key = ("unframed", row_idx)
        units[key].append((row_idx, prediction, link))

    sorted_units = []
    for key, rows in units.items():
        if key[0] == "frame":
            bbox = regions[key[1]]["bbox"]
        else:
            bbox = rows[0][2]["ocr_group_bbox"]
        sorted_units.append((bbox, key, rows))

    if mode == "panel_top":
        # Strict panel-first order based on the top edge, with a coarse row band.
        sorted_units.sort(key=lambda unit: (
            int(unit[0][1] / max(height / 6, 1)),
            -center(unit[0])[0],
            unit[0][1],
        ))
    elif mode in ("panel_center", "panel_balloon"):
        # Panel-first order based on vertical center and right-to-left position.
        sorted_units.sort(key=lambda unit: (
            int(center(unit[0])[1] / max(height / 6, 1)),
            -center(unit[0])[0],
            unit[0][1],
        ))
    else:
        raise ValueError(mode)

    output_rows = []
    for _box, _key, rows in sorted_units:
        if mode == "panel_balloon":
            # Keep OCR fragments belonging to one detected balloon together,
            # preserving their pre-existing order inside that balloon.
            balloons = defaultdict(list)
            for row in rows:
                row_idx, _prediction, link = row
                bubble = link["balloon_region_index"]
                key = ("bubble", bubble) if bubble is not None else ("unlinked", row_idx)
                balloons[key].append(row)
            bubble_units = []
            for bubble_key, bubble_rows in balloons.items():
                if bubble_key[0] == "bubble":
                    index = bubble_key[1]
                    if not (0 <= index < len(regions)) or regions[index]["class"] != "speech_bubble":
                        raise ValueError("Invalid speech balloon index")
                    bbox = regions[index]["bbox"]
                else:
                    bbox = bubble_rows[0][2]["ocr_group_bbox"]
                bubble_units.append((bbox, bubble_rows))
            bubble_units = sort_items_by_manga_rows(bubble_units, image_height=height, bands=12)
            rows = [row for _, group in bubble_units for row in group]
        output_rows.extend(rows)

    if sorted(r[0] for r in output_rows) != list(range(len(base_page))):
        raise RuntimeError("Panel ordering dropped or repeated OCR groups")
    return output_rows


def build_predictions(baseline, links, segmentation, mode):
    output = {}
    changed_pages = 0
    changed_sequences = 0
    displaced = 0
    for sid, original_pages in baseline.items():
        sequence_changed = False
        sequence_turn = 0
        new_pages = []
        for page_idx, original in enumerate(original_pages):
            ordered = reorder_page(original, links[sid][page_idx], segmentation[sid][page_idx], mode)
            orig_indices = [row[0] for row in ordered]
            if orig_indices != list(range(len(original))):
                changed_pages += 1
                sequence_changed = True
                displaced += sum(i != old for i, old in enumerate(orig_indices))
            new_page = []
            for _old_index, original_row, _link in ordered:
                # After changing reading order, rebuild the same simple
                # two-speaker baseline, exactly as predict.py does.
                speaker = f"character{sequence_turn % 2 + 1}"
                sequence_turn += 1
                new_page.append({"speaker": speaker, "text": original_row["text"]})
            if Counter(r["text"] for r in new_page) != Counter(r["text"] for r in original):
                raise RuntimeError("Text changed during panel reordering")
            new_pages.append(new_page)
        output[sid] = new_pages
        changed_sequences += sequence_changed
    return output, {"changed_sequences": changed_sequences,
                    "changed_pages": changed_pages,
                    "displaced_groups": displaced}


def main():
    paths = [
        VAL / "validation_references.jsonl",
        VAL / "pipeline_predictions.jsonl",
        VAL / "speaker_links_diagnostic.json",
        VISION / "segmentation_val.json",
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing prerequisite: {path}")
    references = score.load_jsonl(paths[0], reference=True)
    baseline = score.load_jsonl(paths[1])
    link_data = read_json(paths[2])
    seg_data = read_json(paths[3])
    if seg_data.get("split") != "val":
        raise ValueError("Expected validation segmentation, not training/test")
    links = {s["sequence_id"]: s["pages"] for s in link_data["sequences"]}
    seg = {s["sequence_id"]: s["pages"] for s in seg_data["sequences"]}
    if not (len(baseline) == len(references) == len(links) == len(seg) == 16
            and set(baseline) == set(references) == set(links) == set(seg)):
        raise ValueError("Validation sequence IDs do not match")
    original = score.evaluate(references, baseline)["macro"]
    if abs(original["balanced_joint_f1"] - 0.29931233548051034) > 1e-7:
        raise ValueError("Baseline no longer matches the previously verified score")
    print("Original BYTE baseline: PASS")
    print(f"Text order: {original['text_order_score']:.4f}, "
          f"balanced joint F1: {original['balanced_joint_f1']:.4f}")
    print("\nMethod         | Changed pages | Displaced groups | Text order | Balanced F1 | Delta text")
    print("-" * 95)
    results = {"baseline": original, "variants": []}
    for mode in ("panel_top", "panel_center", "panel_balloon"):
        predictions, counts = build_predictions(baseline, links, seg, mode)
        metrics = score.evaluate(references, predictions)["macro"]
        delta = metrics["text_order_score"] - original["text_order_score"]
        print(f"{mode:14s} | {counts['changed_pages']:13d} | "
              f"{counts['displaced_groups']:16d} | "
              f"{metrics['text_order_score']:.4f}     | "
              f"{metrics['balanced_joint_f1']:.4f}      | {delta:+.4f}")
        out_path = VAL / f"{mode}_predictions.jsonl"
        with out_path.open("w", encoding="utf-8") as stream:
            for sid, pages in predictions.items():
                stream.write(json.dumps({"sequence_id": sid, "pages": pages}, ensure_ascii=False) + "\n")
        results["variants"].append({"method": mode, "counts": counts,
                                    "scores": metrics, "predictions": out_path.name})
    results_path = VAL / "panel_reading_order_sweep.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("\nSaved experiment:", results_path)
    print("Production inference and test predictions were not modified.")


if __name__ == "__main__":
    main()
