
import json
import re
from pathlib import Path

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
CACHE = ROOT / "experiments" / "ocr_cache"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "experiments" / "training"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def normalize(text):
    """Normalize text for approximate annotation matching."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def weak_label(text, reference_lines):
    """
    1: OCR fragment occurs in annotated story text.
    0: OCR fragment appears unrelated to story text.
    None: uncertain; exclude from training.
    """
    candidate = normalize(text)

    if len(candidate) < 4:
        return None

    references = [
        normalize(line["text"])
        for line in reference_lines
    ]

    if any(candidate in ref for ref in references):
        return 1

    reference_words = {
        token
        for line in reference_lines
        for token in words(line["text"])
        if len(token) >= 4
    }

    candidate_words = {
        token for token in words(text)
        if len(token) >= 4
    }

    # Partially matching fragments may contain OCR errors.
    if candidate_words & reference_words:
        return None

    return 0


def features(detection, width, height):
    """Convert an OCR detection into numerical/text features."""
    text = detection["text"]
    x0, y0, x1, y1 = detection["bbox"]

    width = max(width, 1)
    height = max(height, 1)

    cleaned = normalize(text)
    tokens = words(text)

    result = {
        "confidence": float(detection["confidence"]),
        "x_center": (x0 + x1) / (2 * width),
        "y_center": (y0 + y1) / (2 * height),
        "box_width": (x1 - x0) / width,
        "box_height": (y1 - y0) / height,
        "text_length": min(len(text), 100) / 100,
        "word_count": min(len(tokens), 20) / 20,
        "has_question": float("?" in text),
        "has_exclamation": float("!" in text),
        "has_ellipsis": float("..." in text),
        "digit_ratio": sum(c.isdigit() for c in text)
        / max(len(text), 1),
    }

    # Textual features help the model learn language patterns.
    for token in set(tokens):
        if len(token) >= 3:
            result[f"word:{token}"] = 1.0

    # Character trigrams capture short text patterns.
    for index in range(max(0, len(cleaned) - 2)):
        trigram = cleaned[index:index + 3]
        result[f"char3:{trigram}"] = 1.0

    return result


def main():
    train_ids = json.loads(
        (ROOT / "experiments/splits/train_ids.json")
        .read_text(encoding="utf-8")
    )

    references = {}

    label_path = DATASET / "development/labels.jsonl"

    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                references[row["sequence_id"]] = row["pages"]

    assert len(train_ids) == 64
    assert len(set(train_ids)) == 64

    X = []
    y = []

    stats = {
        "training_sequences": len(train_ids),
        "positive": 0,
        "negative": 0,
        "uncertain": 0,
        "total_detections": 0,
    }

    for sequence_id in train_ids:
        reference_pages = references[sequence_id]

        for page_number in range(1, 4):
            cache_path = (
                CACHE / "development" / sequence_id
                / f"{page_number:02d}.json"
            )

            if not cache_path.exists():
                raise FileNotFoundError(cache_path)

            page = json.loads(
                cache_path.read_text(encoding="utf-8")
            )

            for detection in page["detections"]:
                stats["total_detections"] += 1

                label = weak_label(
                    detection["text"],
                    reference_pages[page_number - 1],
                )

                if label is None:
                    stats["uncertain"] += 1
                    continue

                X.append(features(
                    detection,
                    page["width"],
                    page["height"],
                ))

                y.append(label)

                if label == 1:
                    stats["positive"] += 1
                else:
                    stats["negative"] += 1

    if set(y) != {0, 1}:
        raise RuntimeError(
            "Training requires both positive and negative samples."
        )

    print("Training sequences:", stats["training_sequences"])
    print("Total OCR detections:", stats["total_detections"])
    print("Positive samples:", stats["positive"])
    print("Negative samples:", stats["negative"])
    print("Uncertain samples skipped:", stats["uncertain"])

    # Convert feature dictionaries to a sparse matrix.
    vectorizer = DictVectorizer(sparse=True)
    X_vectorized = vectorizer.fit_transform(X)

    print("Learned feature dimensions:", X_vectorized.shape[1])
    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_vectorized, y)

    model_path = MODEL_DIR / "text_filter.joblib"

    joblib.dump({
        "model": model,
        "vectorizer": vectorizer,
        "threshold": 0.5,
        "stats": stats,
    }, model_path)

    stats_path = RESULT_DIR / "text_filter_stats.json"

    stats_path.write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )

    print("\nTraining completed.")
    print("Saved model:", model_path)
    print("Saved training stats:", stats_path)


if __name__ == "__main__":
    main()
