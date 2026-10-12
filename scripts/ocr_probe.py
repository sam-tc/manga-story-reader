import json
from pathlib import Path
from rapidocr import RapidOCR

ROOT = Path(__file__).resolve().parents[1]

image_path = (
    ROOT
    / "dataset"
    / "development"
    / "images"
    / "seq_2032620aa4e4ac7f"
    / "01.png"
)

output_path = ROOT / "experiments" / "first_page_ocr.json"
output_path.parent.mkdir(parents=True, exist_ok=True)

# Load an open-weight OCR model locally.
engine = RapidOCR(params={"Rec.lang_type": "en"})

# Run text detection and recognition.
result = engine(str(image_path))

detections = []

if result.boxes is not None and result.txts is not None:
    for box, text, score in zip(
        result.boxes, result.txts, result.scores
    ):
        detections.append({
            "text": text,
            "confidence": float(score),
            "box": [
                [float(x), float(y)]
                for x, y in box
            ],
        })

with output_path.open("w", encoding="utf-8") as file:
    json.dump(detections, file, indent=2)

print("Image:", image_path.name)
print("Detected text regions:", len(detections))

print("\nRaw OCR results:")
for item in detections:
    print(
        f"{item['confidence']:.2f} | "
        f"{item['text']}"
    )

print("\nSaved results to:", output_path)
