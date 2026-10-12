"""Export manga detector boxes for a sequence-level BYTE development split.

Run with the separate manga-vision-env (Ultralytics installed). This script
creates independent JSON artifacts; it does not change the main OCR model,
inference script, or submission.

Example:
    python scripts/export_yolo_detections.py --split train
    python scripts/export_yolo_detections.py --split val
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
SPLITS = ROOT / "experiments" / "splits"
OUTPUT = ROOT / "experiments" / "vision_detections"
DEFAULT_WEIGHTS = ROOT.parent / "manga-vision-assets" / "best.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="mps", help="mps or cpu")
    parser.add_argument("--confidence", type=float, default=0.20)
    args = parser.parse_args()

    manifest = json.loads((DATASET / "sequences.json").read_text(encoding="utf-8"))
    split_ids = set(json.loads((SPLITS / f"{args.split}_ids.json").read_text(encoding="utf-8")))
    selected = [
        sequence for sequence in manifest
        if sequence["split"] == "development"
        and sequence["sequence_id"] in split_ids
    ]
    assert len(selected) == (64 if args.split == "train" else 16)
    assert {item["sequence_id"] for item in selected} == split_ids

    if not args.weights.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.weights}\n"
            "Specify --weights /full/path/to/best.pt if it is elsewhere."
        )

    print("Loading pretrained manga detector:", args.weights)
    detector = YOLO(str(args.weights))
    names = detector.names
    print("Model classes:", names)

    output_rows = []
    counts = Counter()
    for number, sequence in enumerate(selected, start=1):
        sid = sequence["sequence_id"]
        predicted_pages = []
        for page_number, rel_path in enumerate(sequence["images"], start=1):
            image_path = DATASET / rel_path
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            # Ultralytics returns boxes in ORIGINAL image pixel coordinates,
            # even though the input is resized internally to imgsz=960.
            results = detector.predict(
                source=str(image_path),
                imgsz=960,
                conf=args.confidence,
                device=args.device,
                verbose=False,
                save=False,
            )
            assert len(results) == 1
            result = results[0]
            height, width = result.orig_shape
            boxes = []
            for b in result.boxes:
                cls_id = int(b.cls.item())
                cls_name = str(names[cls_id]).lower()
                confidence = float(b.conf.item())
                x0, y0, x1, y1 = (float(v) for v in b.xyxy[0].tolist())
                boxes.append({
                    "class": cls_name,
                    "confidence": round(confidence, 6),
                    "bbox": [round(v, 2) for v in (x0, y0, x1, y1)],
                })
                counts[cls_name] += 1
            predicted_pages.append({
                "page": page_number,
                "image": rel_path,
                "width": width,
                "height": height,
                "boxes": boxes,
            })
        output_rows.append({"sequence_id": sid, "pages": predicted_pages})
        if number % 8 == 0 or number == len(selected):
            print(f"Processed {number}/{len(selected)} sequences")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / f"{args.split}.json"
    payload = {
        "split": args.split,
        "weights_filename": args.weights.name,
        "detection_confidence": args.confidence,
        "image_size": 960,
        "model_classes": names,
        "sequences": output_rows,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\nSequences:", len(output_rows))
    print("Pages:", sum(len(row["pages"]) for row in output_rows))
    print("Detected boxes by class:", dict(sorted(counts.items())))
    print("Saved:", output_path)
    print("Original BYTE test submission unchanged.")


if __name__ == "__main__":
    main()
