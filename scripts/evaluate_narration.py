
import json
import subprocess
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
VAL = ROOT / "experiments/validation"
MODELS = ROOT / "models"

THRESHOLD = 0.5


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def main():
    train_ids = set(load_json(
        ROOT / "experiments/splits/train_ids.json"
    ))

    references = load_jsonl(
        DATASET / "development/labels.jsonl"
    )

    texts = []
    labels = []

    # Training labels only. No validation references.
    for sequence in references:
        if sequence["sequence_id"] not in train_ids:
            continue

        for page in sequence["pages"]:
            for entry in page:
                if entry["speaker"] == "UNKNOWN":
                    continue

                texts.append(entry["text"])
                labels.append(
                    1 if entry["speaker"] == "NARRATION"
                    else 0
                )

    print("Training samples:", len(texts))
    print("Narration samples:", sum(labels))
    print("Other story text:", len(labels) - sum(labels))

    assert len(train_ids) == 64
    assert set(labels) == {0, 1}

    # TF-IDF converts text patterns into numerical features.
    # Logistic Regression learns to classify those features.
    detector = make_pipeline(
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=12000,
            sublinear_tf=True,
        ),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1500,
            random_state=42,
        ),
    )

    print("\nTraining narration classifier...")
    detector.fit(texts, labels)

    MODELS.mkdir(parents=True, exist_ok=True)

    model_path = MODELS / "narration_detector.joblib"

    joblib.dump({
        "model": detector,
        "threshold": THRESHOLD,
    }, model_path)

    # Read predictions generated independently of references.
    predictions = load_jsonl(
        VAL / "pipeline_predictions.jsonl"
    )

    val_ids = set(load_json(
        ROOT / "experiments/splits/val_ids.json"
    ))

    assert len(predictions) == 16
    assert {
        row["sequence_id"] for row in predictions
    } == val_ids

    changed = 0
    candidates = []

    for sequence in predictions:
        for page in sequence["pages"]:
            if not page:
                continue

            page_texts = [entry["text"] for entry in page]

            probabilities = detector.predict_proba(
                page_texts
            )[:, 1]

            for entry, probability in zip(
                page, probabilities
            ):
                candidates.append((
                    float(probability),
                    sequence["sequence_id"],
                    entry["text"],
                ))

                if probability >= THRESHOLD:
                    entry["speaker"] = "NARRATION"
                    changed += 1

    prediction_path = (
        VAL / "narration_predictions.jsonl"
    )

    with prediction_path.open(
        "w", encoding="utf-8"
    ) as file:
        for sequence in predictions:
            file.write(
                json.dumps(
                    sequence,
                    ensure_ascii=False,
                ) + "\n"
            )

    scores_path = VAL / "narration_scores.json"

    print("Predicted narration entries:", changed)
    print("\nHighest narration probabilities:")

    for probability, sequence_id, text in sorted(
        candidates, reverse=True
    )[:8]:
        print(
            f"{probability:.3f} | "
            f"{sequence_id} | {text[:90]}"
        )

    print("\nRunning official BYTE scorer...\n")

    subprocess.run([
        sys.executable,
        str(DATASET / "score.py"),
        "--references",
        str(VAL / "validation_references.jsonl"),
        "--predictions",
        str(prediction_path),
        "--output",
        str(scores_path),
    ], check=True)

    baseline = load_json(
        VAL / "pipeline_scores.json"
    )["macro"]

    current = load_json(scores_path)["macro"]

    print("\nCOMPARISON")
    print("-" * 55)

    for metric in [
        "text_order_score",
        "balanced_joint_f1",
        "joint_f1",
        "matched_token_coverage",
    ]:
        old = baseline[metric]
        new = current[metric]

        print(
            f"{metric}: "
            f"{old:.4f} -> {new:.4f} "
            f"({new - old:+.4f})"
        )

    # Changing speakers should not alter text ordering.
    assert abs(
        current["text_order_score"]
        - baseline["text_order_score"]
    ) < 1e-9

    print("\nSaved model:", model_path)
    print("Saved predictions:", prediction_path)
    print("Saved scores:", scores_path)


if __name__ == "__main__":
    main()
