
import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from rapidocr import RapidOCR

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
CACHE = ROOT / "experiments" / "ocr_cache"


def normalize(text):
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def box_iou(a, b):
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])

    intersection = (
        max(0, right - left)
        * max(0, bottom - top)
    )

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0


def deduplicate(detections):
    # Higher-confidence detections take priority.
    ordered = sorted(
        detections,
        key=lambda item: item["confidence"],
        reverse=True,
    )

    unique = []

    for candidate in ordered:
        duplicate = any(
            normalize(candidate["text"])
            == normalize(existing["text"])
            and box_iou(
                candidate["bbox"],
                existing["bbox"],
            ) >= 0.35
            for existing in unique
        )

        if not duplicate:
            unique.append(candidate)

    # This is only for inspection, NOT manga reading order.
    return sorted(
        unique,
        key=lambda item: (
            item["bbox"][1],
            item["bbox"][0],
        ),
    )


def extract_page(engine, image_path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    overlap_x = int(width * 0.08)
    overlap_y = int(height * 0.08)

    detections = []

    for row in range(2):
        for col in range(2):
            x0 = max(0, col * width // 2 - overlap_x)
            y0 = max(0, row * height // 2 - overlap_y)
            x1 = min(
                width,
                (col + 1) * width // 2 + overlap_x,
            )
            y1 = min(
                height,
                (row + 1) * height // 2 + overlap_y,
            )

            tile = image.crop((x0, y0, x1, y1))
            result = engine(np.asarray(tile))

            if (
                result.boxes is None
                or result.txts is None
                or result.scores is None
            ):
                continue

            for box, text, score in zip(
                result.boxes,
                result.txts,
                result.scores,
            ):
                xs = [float(p[0]) + x0 for p in box]
                ys = [float(p[1]) + y0 for p in box]

                detections.append({
                    "text": text,
                    "confidence": float(score),
                    "bbox": [
                        min(xs), min(ys),
                        max(xs), max(ys),
                    ],
                })

    return {
        "width": width,
        "height": height,
        "raw_count": len(detections),
        "detections": deduplicate(detections),
    }


def load_sequences(split):
    manifest = json.loads(
        (DATASET / "sequences.json").read_text(
            encoding="utf-8"
        )
    )

    if split == "test":
        return [
            s for s in manifest
            if s["split"] == "test"
        ]

    filename = (
        "train_ids.json"
        if split == "train"
        else "val_ids.json"
    )

    ids = set(json.loads(
        (ROOT / "experiments" / "splits" / filename)
        .read_text(encoding="utf-8")
    ))

    return [
        s for s in manifest
        if s["split"] == "development"
        and s["sequence_id"] in ids
    ]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
    )
    parser.add_argument(
        "--limit-sequences",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    if args.limit_sequences < 1:
        parser.error("--limit-sequences must be positive")

    sequences = load_sequences(args.split)
    sequences = sequences[:args.limit_sequences]

    print("Selected sequences:", len(sequences))
    print("Loading local OCR model...")

    engine = RapidOCR(
        params={"Rec.lang_type": "en"}
    )

    for sequence in sequences:
        sequence_id = sequence["sequence_id"]

        for page_number, relative_path in enumerate(
            sequence["images"], start=1
        ):
            image_path = DATASET / relative_path

            output_dir = (
                CACHE
                / sequence["split"]
                / sequence_id
            )
            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path = (
                output_dir / f"{page_number:02d}.json"
            )

            if output_path.exists() and not args.force:
                print(
                    f"{sequence_id} page {page_number}: cached"
                )
                continue

            result = extract_page(engine, image_path)

            result["sequence_id"] = sequence_id
            result["page_number"] = page_number

            output_path.write_text(
                json.dumps(result, indent=2),
                encoding="utf-8",
            )

            print(
                f"{sequence_id} page {page_number}: "
                f"{result['raw_count']} raw, "
                f"{len(result['detections'])} unique"
            )

    print("\nOCR processing completed.")


if __name__ == "__main__":
    main()
