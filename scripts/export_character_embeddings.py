"""Export deterministic ResNet18 appearance descriptors for segmented manga characters.

Experimental diagnostics only. Does not assign speakers or modify test predictions.

Run from repo root with ../manga-vision-env activated:
  python scripts/export_character_embeddings.py --split val
  python scripts/export_character_embeddings.py --split train

Requires segmentation_{split}.json produced by export_character_panel_detections.py,
original BYTE development images, and cached torchvision ResNet18 IMAGENET1K_V1 weights.
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

from probe_character_embeddings import (
    DATASET,
    ROOT,
    SEGMENTATION_DIR,
    crop_character,
    read_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 32:
        parser.error("--batch-size must be between 1 and 32")

    data = read_json(SEGMENTATION_DIR / f"segmentation_{args.split}.json")
    if data.get("split") != args.split:
        raise ValueError("Incorrect segmentation split")
    expected = 64 if args.split == "train" else 16
    if len(data["sequences"]) != expected:
        raise ValueError(f"Expected {expected} sequences")

    # No labels are used. Images and segmentation data are from the selected split.
    print("Loading ResNet18 ImageNet weights from PyTorch's local cache...")
    weights = ResNet18_Weights.IMAGENET1K_V1
    processor = weights.transforms()
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    print("Inference device:", device)

    embeddings = []
    sequence_ids = []
    page_numbers = []
    region_indices = []
    seen = set()
    with torch.inference_mode():
        for seq_number, sequence in enumerate(data["sequences"], start=1):
            sid = sequence["sequence_id"]
            if len(sequence["pages"]) != 3:
                raise ValueError(f"Wrong page count: {sid}")
            for page_idx, page in enumerate(sequence["pages"]):
                image_file = DATASET / page["image"]
                with Image.open(image_file) as image_handle:
                    image = image_handle.convert("RGB")
                if image.size != (page["width"], page["height"]):
                    raise ValueError(f"Dimension mismatch: {image_file}")

                crops = []
                indices = []
                for region_index, region in enumerate(page["detections"]):
                    if region["class"] != "character":
                        continue
                    crop = crop_character(image, region)
                    if crop is None:
                        continue
                    crops.append(crop)
                    indices.append(region_index)

                for start in range(0, len(crops), args.batch_size):
                    batch = torch.stack([
                        processor(crop)
                        for crop in crops[start:start + args.batch_size]
                    ]).to(device)
                    vectors = model(batch).float().cpu()
                    vectors = F.normalize(vectors, p=2, dim=1).numpy()
                    for vector, region_index in zip(
                        vectors, indices[start:start + args.batch_size]
                    ):
                        key = (sid, page_idx, region_index)
                        if key in seen:
                            raise RuntimeError(f"Duplicate region key: {key}")
                        seen.add(key)
                        sequence_ids.append(sid)
                        page_numbers.append(page_idx)
                        region_indices.append(region_index)
                        embeddings.append(vector)
            if seq_number % 8 == 0 or seq_number == expected:
                print(f"Processed {seq_number}/{expected} sequences")

    if not embeddings:
        raise RuntimeError("No character embeddings generated")
    vectors = np.stack(embeddings).astype("float32", copy=False)
    if vectors.shape[1] != 512 or not np.isfinite(vectors).all():
        raise RuntimeError(f"Unexpected embedding values/shape: {vectors.shape}")

    output = SEGMENTATION_DIR / f"appearance_embeddings_{args.split}.npz"
    np.savez_compressed(
        output,
        vectors=vectors,
        sequence_ids=np.asarray(sequence_ids, dtype="str"),
        page_indices=np.asarray(page_numbers, dtype="int16"),
        region_indices=np.asarray(region_indices, dtype="int32"),
    )
    # Read it back to verify all entries were saved without object pickling.
    with np.load(output, allow_pickle=False) as check:
        if check["vectors"].shape != vectors.shape:
            raise RuntimeError("Saved embedding file failed validation")

    print("\nSequences:", len(data["sequences"]))
    print("Character appearances exported:", len(embeddings))
    print("Embedding dimensions:", vectors.shape[1])
    print("Saved:", output)
    print("Original BYTE submission unchanged. No labels used.")


if __name__ == "__main__":
    main()
