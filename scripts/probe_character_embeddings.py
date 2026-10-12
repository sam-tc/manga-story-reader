"""Probe visual similarities between character appearances in one BYTE development sequence.

This is a diagnostic, NOT a character-identity predictor: closest pairs may be
wrong. Uses pretrained torchvision ResNet18 ImageNet weights (no BYTE labels).
Does not read test images or overwrite final inference output.

Run in the independent manga-vision-env, from the manga-story-reader root:
  python scripts/probe_character_embeddings.py --split train

The script saves comparison images under ../manga-vision-assets/character_embedding_probe
so that no manga images are accidentally added to Git.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
SEGMENTATION_DIR = ROOT / "experiments" / "vision_detections"
OUTDIR = ROOT.parent / "manga-vision-assets" / "character_embedding_probe"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def crop_character(image, region):
    """Crop the segmented character while whitening pixels outside its polygon."""
    width, height = image.size
    x0, y0, x1, y1 = region["bbox"]
    box = (
        max(0, int(x0)), max(0, int(y0)),
        min(width, int(x1 + 0.999)), min(height, int(y1 + 0.999)),
    )
    if box[2] - box[0] < 12 or box[3] - box[1] < 12:
        return None

    crop = image.crop(box).convert("RGB")
    polygon = region.get("polygon") or []
    if len(polygon) >= 3:
        mask = Image.new("L", crop.size, color=0)
        pts = [(float(x) - box[0], float(y) - box[1]) for x, y in polygon]
        ImageDraw.Draw(mask).polygon(pts, fill=255)
        crop = Image.composite(crop, Image.new("RGB", crop.size, "white"), mask)
    return crop


def tile(crop, title, size=140):
    """A labeled square preview; makes it easy to compare appearance pairs."""
    cell = Image.new("RGB", (size + 12, size + 34), "white")
    preview = ImageOps.contain(crop, (size, size))
    cell.paste(preview, (6 + (size - preview.width) // 2,
                         6 + (size - preview.height) // 2))
    ImageDraw.Draw(cell).text((6, size + 13), title[:23], fill="black")
    return cell


def make_grid(cells, columns, destination):
    if not cells:
        return
    w, h = cells[0].size
    rows = (len(cells) + columns - 1) // columns
    image = Image.new("RGB", (columns * w, rows * h), "#dddddd")
    for i, cell in enumerate(cells):
        image.paste(cell, ((i % columns) * w, (i // columns) * h))
    image.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--sequence-id", default=None,
                        help="Default: first sequence in the saved detection export")
    parser.add_argument("--max-characters", type=int, default=64)
    args = parser.parse_args()
    if not 2 <= args.max_characters <= 200:
        parser.error("--max-characters must be between 2 and 200")

    source = SEGMENTATION_DIR / f"segmentation_{args.split}.json"
    data = read_json(source)
    if data.get("split") != args.split:
        raise RuntimeError("Segmentation export has the wrong split")
    sequence_list = data["sequences"]
    if not sequence_list:
        raise RuntimeError("No sequences found in segmentation export")
    if args.sequence_id:
        sequence = next((s for s in sequence_list
                         if s["sequence_id"] == args.sequence_id), None)
        if sequence is None:
            raise ValueError("Sequence ID is not in the chosen split")
    else:
        sequence = sequence_list[0]
    if len(sequence["pages"]) != 3:
        raise RuntimeError("BYTE sequences must contain three pages")

    crops = []
    names = []
    page_indices = []
    for page_num, page in enumerate(sequence["pages"], start=1):
        image_path = DATASET / page["image"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        if image.size != (page["width"], page["height"]):
            raise RuntimeError(f"Image dimensions differ: {image_path}")
        for region_index, region in enumerate(page["detections"]):
            if region["class"] != "character":
                continue
            crop = crop_character(image, region)
            if crop is None:
                continue
            crops.append(crop)
            names.append(f"p{page_num}-r{region_index}")
            page_indices.append(page_num)
            if len(crops) >= args.max_characters:
                break
        if len(crops) >= args.max_characters:
            break

    if len(crops) < 2:
        raise RuntimeError("Not enough character crops to compare")
    print("Sequence:", sequence["sequence_id"])
    print("Character appearances:", len(crops))
    print("Loading pretrained ResNet18 ImageNet weights (download on first run)...")

    weights = ResNet18_Weights.IMAGENET1K_V1
    processor = weights.transforms()
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()  # 512-number description instead of 1000 classes
    model.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    print("Inference device:", device)

    vectors = []
    with torch.inference_mode():
        for start in range(0, len(crops), 8):
            batch = torch.stack([processor(crop)
                                 for crop in crops[start:start + 8]]).to(device)
            embeddings = model(batch).float().cpu()
            vectors.append(F.normalize(embeddings, dim=1))
    all_vectors = torch.cat(vectors, dim=0)
    cosine = (all_vectors @ all_vectors.T).numpy()

    # Only compare appearances from DIFFERENT pages for this initial probe.
    # This tests cross-page retrieval; same-page duplicates cannot dominate.
    pairs = []
    for i in range(len(crops)):
        for j in range(i + 1, len(crops)):
            if page_indices[i] == page_indices[j]:
                continue
            pairs.append({"first": names[i], "second": names[j],
                          "similarity": round(float(cosine[i, j]), 4),
                          "first_index": i, "second_index": j})
    pairs.sort(key=lambda item: item["similarity"], reverse=True)
    if not pairs:
        print("No cross-page pairs in this sequence; try another train sequence.")

    output = OUTDIR / sequence["sequence_id"]
    output.mkdir(parents=True, exist_ok=True)
    make_grid([tile(crop, name) for crop, name in zip(crops, names)],
              columns=6, destination=output / "character_crops.jpg")

    comparisons = []
    for item in pairs[:10]:
        i, j = item["first_index"], item["second_index"]
        comparisons.extend([tile(crops[i], names[i]),
                            tile(crops[j], f"{names[j]} ({item['similarity']:.2f})")])
    make_grid(comparisons, columns=2, destination=output / "closest_cross_page_pairs.jpg")

    summary = {"sequence_id": sequence["sequence_id"],
               "model": "torchvision ResNet18 pretrained ImageNet1K V1",
               "embedding_dim": int(all_vectors.shape[1]),
               "num_character_crops": len(crops),
               "pairs": [{key: val for key, val in p.items()
                          if key not in ("first_index", "second_index")}
                         for p in pairs[:30]]}
    (output / "similarity_pairs.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\nTop cross-page appearance similarities:")
    for item in pairs[:10]:
        print(f"{item['similarity']:.3f}: {item['first']} vs {item['second']}")
    print("\nSaved previews to:", output)
    print("WARNING: Similarity does NOT prove the characters are identical.")
    print("No labels, trained BYTE models, or test predictions were modified.")


if __name__ == "__main__":
    main()
