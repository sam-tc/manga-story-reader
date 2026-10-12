"""Probe a pretrained manga instance-segmentation model on three BYTE development pages.

Run from project directory in the separate vision virtual environment:
    python scripts/probe_manga_character_panels.py

Downloads pretrained weights from the cited original model card (not trained here).
This script does not load any labels, change the BYTE submission, or train models.
"""
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_ID = "seq_ee880b9612205860"
INPUT = ROOT / "dataset" / "development" / "images" / SEQUENCE_ID
OUTPUT = ROOT.parent / "manga-vision-assets" / "character_panel_probe"

MODEL_REPO = "anonimkaq4/manga-page-element-segmentation"
MODEL_FILE = "snapshots/yolo11m_1024_synthetic_mangaseg/best.pt"


def main():
    images = [INPUT / f"{page:02d}.png" for page in range(1, 4)]
    for path in images:
        if not path.is_file():
            raise FileNotFoundError(f"Cannot find dataset image: {path}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("Downloading/loading pretrained manga segmentation checkpoint...")
    checkpoint = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    print("Checkpoint:", checkpoint)
    model = YOLO(checkpoint)
    print("Classes:", model.names)

    expected = {"character", "frame", "speech_bubble"}
    if set(model.names.values()) != expected:
        raise RuntimeError(f"Unexpected model classes: {model.names}")

    for image_path in images:
        result = model.predict(
            source=str(image_path),
            imgsz=768,
            conf=0.25,
            device="mps",
            verbose=False,
        )[0]
        counts = Counter(
            result.names[int(cls)]
            for cls in result.boxes.cls.cpu().tolist()
        )
        output_path = OUTPUT / image_path.name
        result.save(filename=str(output_path))
        print(f"{image_path.name}: {dict(counts)} -> {output_path}")

    print("Experiment complete. Original BYTE inference/submission unchanged.")


if __name__ == "__main__":
    main()
