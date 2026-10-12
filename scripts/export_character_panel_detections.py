"""Export open-weight manga instance-segmentation predictions for BYTE development splits.

The output contains panels, characters and balloons with bounding boxes and
segmentation polygons in original image coordinates. No BYTE labels or test
images are used, and no existing inference/submission file is modified.

Run in the separately created manga-vision-env:
    python scripts/export_character_panel_detections.py --split train --limit 1
    python scripts/export_character_panel_detections.py --split train
    python scripts/export_character_panel_detections.py --split val
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
SPLITS = ROOT / "experiments" / "splits"
OUTPUT_DIR = ROOT / "experiments" / "vision_detections"
MODEL_REPO = "anonimkaq4/manga-page-element-segmentation"
MODEL_FILE = "snapshots/yolo11m_1024_synthetic_mangaseg/best.pt"
MODEL_REVISION = "1f30cd14d255569d431d787d7eea7c1874892241"
EXPECTED_CLASSES = {"character", "frame", "speech_bubble"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))



def select_sequences(split):
    manifest = read_json(DATASET / "sequences.json")

    if split == "test":
        sample_path = DATASET / "sample_submission.jsonl"

        with sample_path.open(encoding="utf-8") as file:
            sample = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        ids = [row["sequence_id"] for row in sample]

        by_id = {
            item["sequence_id"]: item
            for item in manifest
            if item["split"] == "test"
        }

        if (
            len(ids) != 15
            or len(set(ids)) != 15
            or set(ids) != set(by_id)
        ):
            raise RuntimeError(
                "Official test sequence IDs do not match"
            )

        return [by_id[sid] for sid in ids]

    ids = set(read_json(SPLITS / f"{split}_ids.json"))

    selected = [
        item for item in manifest
        if item["split"] == "development"
        and item["sequence_id"] in ids
    ]

    expected = 64 if split == "train" else 16

    if (
        len(selected) != expected
        or {s["sequence_id"] for s in selected} != ids
    ):
        raise RuntimeError(
            f"Incorrect {split} split"
        )

    return selected



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        required=True,
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only this many sequences for a smoke test")
    parser.add_argument("--device", default="mps", help="mps or cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=768)
    args = parser.parse_args()

    selected = select_sequences(args.split)
    if args.limit is not None:
        if not 1 <= args.limit <= len(selected):
            parser.error("--limit must be between 1 and the split sequence count")
        selected = selected[:args.limit]

    checkpoint = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        revision=MODEL_REVISION,
    )
    model = YOLO(checkpoint)
    names = {int(k): str(v) for k, v in model.names.items()}
    if set(names.values()) != EXPECTED_CLASSES:
        raise RuntimeError(f"Unexpected model classes: {names}")

    counters = Counter()
    result_sequences = []

    for number, sequence in enumerate(selected, 1):
        pages = []
        for page_no, relative_path in enumerate(sequence["images"], 1):
            image_path = DATASET / relative_path
            if not image_path.is_file():
                raise FileNotFoundError(image_path)

            result = model.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
                save=False,
            )[0]
            height, width = (int(v) for v in result.orig_shape)
            boxes = result.boxes
            polygons = result.masks.xy if result.masks is not None else []
            if len(boxes) and len(boxes) != len(polygons):
                raise RuntimeError(
                    f"Mask/box mismatch: {relative_path}: "
                    f"{len(boxes)} boxes vs {len(polygons)} polygons"
                )

            detections = []
            for i, box in enumerate(boxes):
                label = names[int(box.cls.item())]
                coords = [round(float(v), 2) for v in box.xyxy[0].tolist()]
                outline = [
                    [round(float(x), 2), round(float(y), 2)]
                    for x, y in polygons[i]
                ]
                detections.append({
                    "class": label,
                    "confidence": round(float(box.conf.item()), 6),
                    "bbox": coords,
                    "polygon": outline,
                })
                counters[label] += 1

            pages.append({
                "page": page_no,
                "image": relative_path,
                "width": width,
                "height": height,
                "detections": detections,
            })

        result_sequences.append({
            "sequence_id": sequence["sequence_id"],
            "pages": pages,
        })
        if number % 8 == 0 or number == len(selected):
            print(f"Processed {number}/{len(selected)} sequences")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.limit is not None else ""
    output_path = OUTPUT_DIR / f"segmentation_{args.split}{suffix}.json"
    output = {
        "split": args.split,
        "source_model": MODEL_REPO,
        "source_revision": MODEL_REVISION,
        "image_size": args.imgsz,
        "confidence_threshold": args.conf,
        "classes": names,
        "sequences": result_sequences,
    }
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print("\nSequences:", len(result_sequences))
    print("Pages:", sum(len(s["pages"]) for s in result_sequences))
    print("Detected regions:", dict(sorted(counters.items())))
    print("Saved:", output_path)
    print("No training labels, test images, or final submission were changed.")


if __name__ == "__main__":
    main()
