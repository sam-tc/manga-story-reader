"""Test narration detection on panel-aware validation predictions.

Combines previously tested visual candidates with a trained
text classifier. Never modifies test submissions.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "experiments" / "validation"

sys.path.insert(0, str(ROOT / "dataset"))
import score


def main():
    original = score.load_jsonl(
        VAL / "pipeline_predictions.jsonl"
    )
    visual = score.load_jsonl(
        VAL / "visual_narration_predictions.jsonl"
    )
    panel = score.load_jsonl(
        VAL / "panel_top_live_predictions.jsonl"
    )
    references = score.load_jsonl(
        VAL / "validation_references.jsonl",
        reference=True,
    )

    assert (
        original.keys()
        == visual.keys()
        == panel.keys()
        == references.keys()
    ), "Validation sequence IDs do not match"

    package = joblib.load(
        ROOT / "models/narration_detector.joblib"
    )
    model = package["model"]
    positive_index = list(model.classes_).index(1)

    baseline_score = score.evaluate(
        references, panel
    )["macro"]

    print(
        "Panel baseline:",
        round(baseline_score["balanced_joint_f1"], 4)
    )

    results = []

    for threshold in (0.25, 0.35, 0.50, 0.60):
        candidate = {}
        narration_count = 0
        false_sequence_count = 0
        ambiguous_skipped = 0

        for sid, panel_pages in panel.items():
            candidate_pages = []

            gold_has_narration = any(
                entry["speaker"] == "NARRATION"
                for page in references[sid]
                for entry in page
            )

            for old_page, visual_page, panel_page in zip(
                original[sid],
                visual[sid],
                panel_pages,
            ):
                assert len(old_page) == len(visual_page)

                old_counts = Counter(
                    row["text"] for row in old_page
                )
                panel_counts = Counter(
                    row["text"] for row in panel_page
                )

                assert old_counts == panel_counts, (
                    "Panel ordering changed recognized text"
                )

                visual_candidates = set()
                ambiguous_texts = set()

                for old_row, visual_row in zip(
                    old_page, visual_page
                ):
                    assert old_row["text"] == visual_row["text"]

                    if visual_row["speaker"] != "NARRATION":
                        continue

                    text = old_row["text"]

                    if old_counts[text] == 1:
                        visual_candidates.add(text)
                    else:
                        ambiguous_texts.add(text)

                texts = [
                    row["text"] for row in panel_page
                ]

                probabilities = (
                    model.predict_proba(texts)[
                        :, positive_index
                    ]
                    if texts else []
                )

                new_page = []

                for row, probability in zip(
                    panel_page, probabilities
                ):
                    text = row["text"]

                    if text in ambiguous_texts:
                        ambiguous_skipped += 1

                    is_narration = (
                        text in visual_candidates
                        and probability >= threshold
                    )

                    speaker = (
                        "NARRATION"
                        if is_narration
                        else row["speaker"]
                    )

                    if is_narration:
                        narration_count += 1
                        if not gold_has_narration:
                            false_sequence_count += 1

                    new_page.append({
                        "speaker": speaker,
                        "text": text,
                    })

                candidate_pages.append(new_page)

            candidate[sid] = candidate_pages

        metrics = score.evaluate(
            references, candidate
        )["macro"]

        assert abs(
            metrics["text_order_score"]
            - baseline_score["text_order_score"]
        ) < 1e-9, "Reading order changed unexpectedly"

        result = {
            "threshold": threshold,
            "narration_entries": narration_count,
            "narration_in_no_narration_sequences": (
                false_sequence_count
            ),
            "ambiguous_entries_skipped": ambiguous_skipped,
            "text_order_score": metrics["text_order_score"],
            "balanced_joint_f1": metrics["balanced_joint_f1"],
            "delta": (
                metrics["balanced_joint_f1"]
                - baseline_score["balanced_joint_f1"]
            ),
        }

        results.append(result)

        if threshold == 0.50:
            print("\nNarration predictions at threshold 0.50:")

            for sid, pages in candidate.items():
                for page_index, page in enumerate(pages):
                    for entry in page:
                        if entry["speaker"] != "NARRATION":
                            continue

                        reference_narration = [
                            item["text"]
                            for item in references[sid][page_index]
                            if item["speaker"] == "NARRATION"
                        ]

                        print(f"\nSequence: {sid}")
                        print(f"Page: {page_index + 1}")
                        print(f"Predicted: {entry['text']!r}")
                        print(
                            "Actual narration on that page:",
                            reference_narration,
                        )

        print(
            f"Threshold {threshold:.2f}: "
            f"NARRATION={narration_count}, "
            f"F1={metrics['balanced_joint_f1']:.4f}, "
            f"delta={result['delta']:+.4f}"
        )

    output = VAL / "panel_narration_results.json"

    output.write_text(
        json.dumps(
            {
                "baseline": baseline_score,
                "experiments": results,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("\nSaved:", output)
    print("Original and test predictions unchanged.")


if __name__ == "__main__":
    main()
