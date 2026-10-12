import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from rapidocr import RapidOCR

ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = (
    ROOT / "dataset/development/images"
    / "seq_2032620aa4e4ac7f/01.png"
)

OUTPUT_DIR = ROOT / "experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

engine = RapidOCR(params={"Rec.lang_type": "en"})
image = Image.open(IMAGE_PATH).convert("RGB")

width, height = image.size
overlap_x = int(width * 0.08)
overlap_y = int(height * 0.08)


def normalize(text):
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def box_iou(a, b):
    """Measure how much two rectangles overlap."""
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])

    intersection = max(0, right - left) * max(0, bottom - top)

    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    union = area_a + area_b - intersection
    return intersection / union if union else 0


detections = []

for row in range(2):
    for col in range(2):
        x0 = max(0, col * width // 2 - overlap_x)
        y0 = max(0, row * height // 2 - overlap_y)
        x1 = min(width, (col + 1) * width // 2 + overlap_x)
        y1 = min(height, (row + 1) * height // 2 + overlap_y)

        tile = np.asarray(image.crop((x0, y0, x1, y1)))
        result = engine(tile)

        if (
            result.boxes is None
            or result.txts is None
            or result.scores is None
        ):
            continue

        for box, text, score in zip(
            result.boxes, result.txts, result.scores
        ):
            # Convert tile coordinates back to page coordinates.
            xs = [float(point[0]) + x0 for point in box]
            ys = [float(point[1]) + y0 for point in box]

            detections.append({
                "text": text,
                "confidence": float(score),
                "bbox": [
                    min(xs), min(ys),
                    max(xs), max(ys)
                ],
            })


# Keep higher-confidence detections first.
detections.sort(
    key=lambda item: item["confidence"],
    reverse=True
)

unique = []

for candidate in detections:
    duplicate = False

    for existing in unique:
        same_text = (
            normalize(candidate["text"])
            == normalize(existing["text"])
        )

        same_location = (
            box_iou(candidate["bbox"], existing["bbox"]) >= 0.35
        )

        if same_text and same_location:
            duplicate = True
            break

    if not duplicate:
        unique.append(candidate)


# Spatial sorting for inspection only.
# This is NOT final manga reading order.
unique.sort(key=lambda item: (
    item["bbox"][1],
    item["bbox"][0]
))

raw_path = OUTPUT_DIR / "tiled_ocr_located.json"
clean_path = OUTPUT_DIR / "tiled_ocr_deduplicated.json"

raw_path.write_text(
    json.dumps(detections, indent=2),
    encoding="utf-8"
)
clean_path.write_text(
    json.dumps(unique, indent=2),
    encoding="utf-8"
)


# Create a visual debugging image.
visual = image.copy()
draw = ImageDraw.Draw(visual)

for index, item in enumerate(unique, start=1):
    draw.rectangle(item["bbox"], outline="red", width=2)
    draw.text(
        (item["bbox"][0], max(0, item["bbox"][1] - 12)),
        str(index),
        fill="red"
    )

visual_path = OUTPUT_DIR / "tiled_ocr_visualization.png"
visual.save(visual_path)

print("Original detections:", len(detections))
print("After deduplication:", len(unique))
print("Duplicates removed:", len(detections) - len(unique))

print("\nRemaining text fragments:")
for index, item in enumerate(unique, start=1):
    print(f"{index}. {item['text']}")

print("\nSaved visualization:", visual_path)
print("Saved structured OCR:", clean_path)
