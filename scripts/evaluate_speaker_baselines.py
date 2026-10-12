
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
OUTPUT = ROOT / "experiments" / "validation"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def assign_speakers(sequences, num_characters):
    """
    Assign alternating character labels across all
    three pages of each sequence.

    This is an experimental baseline, not visual
    character recognition.
    """
    predictions = []

    for sequence in sequences:
        turn_index = 0
        pages = []

        for page in sequence["pages"]:
            predicted_page = []

            for entry in page:
                speaker_number = (
                    turn_index % num_characters
                ) + 1

                predicted_page.append({
                    "speaker": f"character{speaker_number}",
                    "text": entry["text"],
                })

                turn_index += 1

            pages.append(predicted_page)

        predictions.append({
            "sequence_id": sequence["sequence_id"],
            "pages": pages,
        })

    return predictions


def save_jsonl(path, sequences):
    with path.open("w", encoding="utf-8") as file:
        for sequence in sequences:
            file.write(
                json.dumps(
                    sequence,
                    ensure_ascii=False,
                ) + "\n"
            )


def evaluate(predictions, name):
    prediction_path = OUTPUT / f"{name}_predictions.jsonl"
    scores_path = OUTPUT / f"{name}_scores.json"

    save_jsonl(prediction_path, predictions)

    subprocess.run([
        sys.executable,
        str(DATASET / "score.py"),
        "--references",
        str(OUTPUT / "validation_references.jsonl"),
        "--predictions",
        str(prediction_path),
        "--output",
        str(scores_path),
    ], check=True, capture_output=True, text=True)

    return load_json(scores_path)["macro"]


def main():
    grouped = []

    with (
        OUTPUT / "grouped_predictions.jsonl"
    ).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                grouped.append(json.loads(line))

    assert len(grouped) == 16

    original = load_json(
        OUTPUT / "grouped_scores.json"
    )["macro"]

    print("Evaluating speaker-label baselines...\n")

    results = {
        "single_speaker": original,
    }

    for n in [2, 3]:
        name = f"alternate_{n}_speakers"

        predictions = assign_speakers(grouped, n)
        results[name] = evaluate(predictions, name)

    print("RESULTS")
    print("-" * 65)

    for name, scores in results.items():
        print(f"\n{name}")
        print(
            "  text_order_score:",
            round(scores["text_order_score"], 4)
        )
        print(
            "  balanced_joint_f1:",
            round(scores["balanced_joint_f1"], 4)
        )
        print(
            "  joint_f1:",
            round(scores["joint_f1"], 4)
        )

    # Reassigning speaker labels must not alter text.
    for scores in results.values():
        assert abs(
            scores["text_order_score"]
            - original["text_order_score"]
        ) < 1e-9

    print("\nSpeaker baseline comparison completed.")


if __name__ == "__main__":
    main()
