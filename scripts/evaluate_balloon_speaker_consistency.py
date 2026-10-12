"""Explore whether repeated OCR groups in one detected balloon should share a speaker.

Requires saved validation OCR/segmentation links from diagnose_speaker_links.py.
Uses validation references ONLY for final scoring, not for making predictions.
Does not modify scripts/predict.py or submissions/test_predictions.jsonl.

Run from the project's main .venv:
    python scripts/evaluate_balloon_speaker_consistency.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "experiments" / "validation"
sys.path.insert(0, str(ROOT / "dataset"))
import score  # Official BYTE scoring script, provided with the dataset


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path, predictions):
    with path.open("w", encoding="utf-8") as file:
        for sequence_id, pages in predictions.items():
            file.write(json.dumps({
                "sequence_id": sequence_id,
                "pages": pages,
            }, ensure_ascii=False) + "\n")


def build_predictions(baseline, links, mode):
    """Alter speaker labels only; never modify predicted text or its order."""
    if mode not in {"first_occurrence", "balloon_turn"}:
        raise ValueError(f"Unknown mode: {mode}")

    predictions = {}
    summary = Counter()

    for sid, base_pages in baseline.items():
        link_pages = links[sid]
        if len(base_pages) != 3 or len(link_pages) != 3:
            raise ValueError(f"Expected exactly three pages: {sid}")

        sequence_turn = 0
        new_pages = []

        for page_index, (base_page, link_page) in enumerate(zip(base_pages, link_pages)):
            if len(base_page) != len(link_page):
                raise ValueError(f"Row-count mismatch: {sid} page {page_index + 1}")

            # A region_index is only unique within a single page.
            seen_balloon_speakers = {}
            new_page = []

            for base_row, link_row in zip(base_page, link_page):
                if (base_row["text"] != link_row["text"]
                        or base_row["speaker"] != link_row["baseline_speaker"]):
                    raise ValueError(f"Diagnostic differs from baseline: {sid} page {page_index + 1}")

                balloon_id = link_row["balloon_region_index"]
                if balloon_id is not None:
                    summary["linked_groups"] += 1
                else:
                    summary["unlinked_groups"] += 1

                if mode == "first_occurrence":
                    # Control: change ONLY repeated groups inside an already-seen balloon.
                    speaker = (seen_balloon_speakers.get(balloon_id, base_row["speaker"])
                               if balloon_id is not None else base_row["speaker"])
                    if balloon_id is not None and balloon_id not in seen_balloon_speakers:
                        seen_balloon_speakers[balloon_id] = speaker
                else:
                    # Make each newly observed balloon one conversational turn.
                    # No balloon match: fall back to creating a separate turn.
                    if balloon_id is not None and balloon_id in seen_balloon_speakers:
                        speaker = seen_balloon_speakers[balloon_id]
                    else:
                        speaker = f"character{sequence_turn % 2 + 1}"
                        sequence_turn += 1
                        if balloon_id is not None:
                            seen_balloon_speakers[balloon_id] = speaker

                if balloon_id is not None and seen_balloon_speakers[balloon_id] != speaker:
                    raise AssertionError("A detected balloon acquired inconsistent labels")
                if speaker != base_row["speaker"]:
                    summary["speaker_labels_changed"] += 1
                new_page.append({"speaker": speaker, "text": base_row["text"]})

            new_pages.append(new_page)
        predictions[sid] = new_pages

    return predictions, dict(summary)


def main():
    reference_path = VAL / "validation_references.jsonl"
    baseline_path = VAL / "pipeline_predictions.jsonl"
    diagnostic_path = VAL / "speaker_links_diagnostic.json"
    for path in (reference_path, baseline_path, diagnostic_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")

    references = score.load_jsonl(reference_path, reference=True)
    baseline = score.load_jsonl(baseline_path)
    diagnostic = read_json(diagnostic_path)
    links = {seq["sequence_id"]: seq["pages"] for seq in diagnostic["sequences"]}

    if not (set(references) == set(baseline) == set(links) and len(links) == 16):
        raise RuntimeError("Reference, baseline and diagnostic sequence IDs differ")

    baseline_report = score.evaluate(references, baseline)
    baseline_f1 = baseline_report["macro"]["balanced_joint_f1"]
    if abs(baseline_f1 - 0.29931233548051034) > 1e-7:
        raise RuntimeError(f"Unexpected baseline score: {baseline_f1}")

    print("Original BYTE baseline: PASS")
    print(f"Balanced joint F1: {baseline_f1:.4f}")
    print()

    results = {"baseline": baseline_report["macro"], "variants": {}}
    print("Method             | Changed labels | Text order | Balanced F1 | Difference")
    print("-" * 76)

    for mode in ("first_occurrence", "balloon_turn"):
        predictions, counts = build_predictions(baseline, links, mode)
        # Never accept a variant that changes transcript content/order.
        for sid in baseline:
            for baseline_page, changed_page in zip(baseline[sid], predictions[sid]):
                if [r["text"] for r in baseline_page] != [r["text"] for r in changed_page]:
                    raise RuntimeError("Text or order changed unexpectedly")

        report = score.evaluate(references, predictions)
        metrics = report["macro"]
        if abs(metrics["text_order_score"] - baseline_report["macro"]["text_order_score"]) > 1e-12:
            raise RuntimeError("Text/order score should be identical")
        delta = metrics["balanced_joint_f1"] - baseline_f1
        print(f"{mode:18s} | {counts.get('speaker_labels_changed', 0):14d} | "
              f"{metrics['text_order_score']:.4f}     | {metrics['balanced_joint_f1']:.4f}     | {delta:+.4f}")

        write_jsonl(VAL / f"{mode}_predictions.jsonl", predictions)
        (VAL / f"{mode}_scores.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        results["variants"][mode] = {"metrics": metrics, "diagnostics": counts}

    (VAL / "balloon_speaker_comparison.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print("\nSaved experimental predictions, official scores and comparison report.")
    print("Final test JSONL and production inference remain untouched.")


if __name__ == "__main__":
    main()
