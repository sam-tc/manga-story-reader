"""Experiment: require BOTH a visual-caption candidate and text-model evidence.

Uses previously saved validation predictions; does not edit final inference or test output.
Runs the official BYTE scorer on the held-out development sequences.
"""
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "experiments" / "validation"
sys.path.insert(0, str(ROOT / "dataset"))
import score  # noqa: E402


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def narration_alignment(references, predictions):
    """Weighted correct/wrong speaker labels among text-aligned narration predictions."""
    matched_correct = matched_wrong = 0.0
    for sid, gold_pages in references.items():
        for gold, predicted in zip(gold_pages, predictions[sid]):
            gold_tokens = score.tokens(gold)
            pred_tokens = score.tokens(predicted)
            for (_, real_speaker), (_, predicted_speaker), weight in score.align_tokens(
                gold_tokens, pred_tokens
            ):
                if predicted_speaker == "NARRATION":
                    if real_speaker == "NARRATION":
                        matched_correct += weight
                    elif real_speaker is not None:
                        matched_wrong += weight
    return round(matched_correct, 2), round(matched_wrong, 2)


def main():
    original = score.load_jsonl(VAL / "pipeline_predictions.jsonl")
    visual = score.load_jsonl(VAL / "visual_narration_predictions.jsonl")
    references = score.load_jsonl(VAL / "validation_references.jsonl", reference=True)
    assert len(original) == len(visual) == len(references) == 16
    assert original.keys() == visual.keys() == references.keys(), "Sequence IDs differ"

    # The visual experiment is allowed to change speaker labels, NEVER the text.
    visual_candidates = {}
    for sid in original:
        original_pages, visual_pages = original[sid], visual[sid]
        assert len(original_pages) == len(visual_pages) == 3
        visual_candidates[sid] = []
        for base_page, image_page in zip(original_pages, visual_pages):
            assert len(base_page) == len(image_page)
            flags = []
            for base, image in zip(base_page, image_page):
                assert base["text"] == image["text"], "OCR text/order mismatch"
                assert image["speaker"] in (base["speaker"], "NARRATION")
                flags.append(image["speaker"] == "NARRATION")
            visual_candidates[sid].append(flags)
    print("Visual/base predictions align exactly: PASS")

    model_package = joblib.load(ROOT / "models/narration_detector.joblib")
    detector = model_package["model"]
    yes_index = list(detector.classes_).index(1)

    text_probs = {}
    for sid, pages in original.items():
        texts = [entry["text"] for page in pages for entry in page]
        probs = (
            detector.predict_proba(texts)[:, yes_index].tolist()
            if texts else []
        )
        position = 0
        text_probs[sid] = []
        for page in pages:
            text_probs[sid].append(probs[position:position + len(page)])
            position += len(page)

    baseline_report = score.evaluate(references, original)["macro"]
    print("Baseline joint F1:", round(baseline_report["balanced_joint_f1"], 4))
    print("\nThreshold | Turn handling | NARRATION | False in zero-narration sequences | Balanced joint F1 | Delta")
    print("-" * 102)

    results = []
    for threshold in (0.25, 0.35, 0.50, 0.60):
        for mode in ("preserve", "skip_narration"):
            candidate = {}
            total_narration = 0
            guaranteed_false = 0
            for sid, pages in original.items():
                turn_index = 0
                new_pages = []
                gold_has_narration = any(
                    entry["speaker"] == "NARRATION"
                    for page in references[sid] for entry in page
                )
                for page_index, page in enumerate(pages):
                    new_page = []
                    for row_index, entry in enumerate(page):
                        is_narration = (
                            visual_candidates[sid][page_index][row_index]
                            and text_probs[sid][page_index][row_index] >= threshold
                        )
                        if is_narration:
                            speaker = "NARRATION"
                            total_narration += 1
                            guaranteed_false += int(not gold_has_narration)
                        elif mode == "preserve":
                            speaker = entry["speaker"]
                        else:
                            speaker = f"character{turn_index % 2 + 1}"

                        # Narration is not itself a turn spoken by a character.
                        if mode == "skip_narration" and not is_narration:
                            turn_index += 1
                        new_page.append({"speaker": speaker, "text": entry["text"]})
                    new_pages.append(new_page)
                candidate[sid] = new_pages

            macro = score.evaluate(references, candidate)["macro"]
            assert abs(
                macro["text_order_score"] - baseline_report["text_order_score"]
            ) < 1e-9, "Speaker labels unexpectedly changed text score"
            matched_correct, matched_wrong = narration_alignment(references, candidate)
            delta = macro["balanced_joint_f1"] - baseline_report["balanced_joint_f1"]
            results.append({
                "threshold": threshold,
                "turn_handling": mode,
                "narration_entries": total_narration,
                "entries_in_sequences_with_no_narration": guaranteed_false,
                "correctly_aligned_narration_token_weight": matched_correct,
                "wrongly_aligned_narration_token_weight": matched_wrong,
                "text_order_score": macro["text_order_score"],
                "balanced_joint_f1": macro["balanced_joint_f1"],
                "joint_f1": macro["joint_f1"],
                "delta_balanced_joint_f1": delta,
            })
            print(
                f"{threshold:8.2f} | {mode:14s} | {total_narration:9d} | "
                f"{guaranteed_false:32d} | {macro['balanced_joint_f1']:.4f}            | {delta:+.4f}"
            )

    path = VAL / "narration_fusion_results.json"
    save_json(path, {
        "baseline": baseline_report,
        "experiments": results,
        "note": "Exploratory validation sweep; not evidence of generalization to hidden test.",
    })
    best = max(results, key=lambda row: row["balanced_joint_f1"])
    print("\nBest experimental variant:", best["threshold"], best["turn_handling"])
    print("Correct narration-aligned token weight:", best["correctly_aligned_narration_token_weight"])
    print("Wrong narration-aligned token weight:", best["wrongly_aligned_narration_token_weight"])
    print("Saved experiment results:", path)
    print("Test predictions and production pipeline were NOT modified.")


if __name__ == "__main__":
    main()
