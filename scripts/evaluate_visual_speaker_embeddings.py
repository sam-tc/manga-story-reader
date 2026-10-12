"""Experimental speaker assignment using segmentation-linked ResNet18 embeddings.

Run from project root with original .venv activated:
    python scripts/evaluate_visual_speaker_embeddings.py

IMPORTANT: Reference annotations are loaded ONLY for evaluation. No validation labels
are used to decide character clusters, speaker labels, or candidate associations.
Does not change scripts/predict.py or submissions/test_predictions.jsonl.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "experiments" / "validation"
VISION = ROOT / "experiments" / "vision_detections"
sys.path.insert(0, str(ROOT / "dataset"))
import score  # BYTE's provided official scorer


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path, predictions):
    with path.open("w", encoding="utf-8") as stream:
        for sid, pages in predictions.items():
            stream.write(json.dumps({"sequence_id": sid, "pages": pages},
                                    ensure_ascii=False) + "\n")


def embedding_table(path):
    """Key every embedding by (sequence ID, zero-indexed page, region index)."""
    with np.load(path, allow_pickle=False) as data:
        vectors = data["vectors"]
        ids = data["sequence_ids"]
        pages = data["page_indices"]
        regions = data["region_indices"]
    if vectors.ndim != 2 or vectors.shape[1] != 512 or not np.isfinite(vectors).all():
        raise ValueError(f"Invalid embedding shape/values: {vectors.shape}")
    if not all(len(arr) == len(vectors) for arr in (ids, pages, regions)):
        raise ValueError("Embedding metadata length mismatch")
    table = {}
    for sid, page, region, vector in zip(ids, pages, regions, vectors):
        key = (str(sid), int(page), int(region))
        if key in table:
            raise ValueError(f"Duplicate embedding key: {key}")
        table[key] = np.asarray(vector, dtype=np.float32)
    return table


def cluster_characters(sid, pages, table, threshold):
    """Greedy cosine clustering within one sequence; no reference labels involved.

    Prototype is the normalized mean of already assigned vectors. Results depend on
    detection order; this is a baseline, not a learned manga re-identification model.
    """
    clusters = []
    assigned = {}
    for page_idx, page in enumerate(pages):
        for region_idx, region in enumerate(page["detections"]):
            if region["class"] != "character":
                continue
            key = (sid, page_idx, region_idx)
            if key not in table:
                continue  # crop was discarded by the embedding exporter
            vector = table[key]
            norm = np.linalg.norm(vector)
            if norm <= 0:
                raise ValueError(f"Zero embedding: {key}")
            vector = vector / norm
            if clusters:
                similarities = [float(np.dot(vector, c["centroid"])) for c in clusters]
                best = int(np.argmax(similarities))
                if similarities[best] >= threshold:
                    cluster = clusters[best]
                    cluster["total"] += vector
                    cluster["count"] += 1
                    cluster["page_set"].add(page_idx)
                    cluster["centroid"] = cluster["total"] / np.linalg.norm(cluster["total"])
                    assigned[(page_idx, region_idx)] = best
                    continue
            clusters.append({"total": vector.copy(), "centroid": vector.copy(),
                             "count": 1, "page_set": {page_idx}})
            assigned[(page_idx, region_idx)] = len(clusters) - 1
    return assigned, clusters


def visual_predictions(baseline, links, segmentation, table, *,
                       threshold, mode, max_distance=0.22):
    """Use candidate geometry and visual clusters to assign repeatable speaker IDs.

    `unique`: only balloons with ONE candidate character can get visual labels.
    `closest`: accepts the closest candidate even in multi-character panels, but
    requires normalized balloon-to-character box distance <= max_distance.
    For unassigned balloons, preserve first-occurrence speaker labels, making this a
    controlled comparison against the previous experiment. Balloons always share a label.
    """
    if mode not in {"unique", "closest"}:
        raise ValueError(mode)
    outputs = {}
    counters = Counter()
    for sid, original_pages in baseline.items():
        link_pages = links[sid]
        seg_pages = segmentation[sid]
        clusters_by_region, clusters = cluster_characters(sid, seg_pages, table, threshold)
        counters["identity_clusters"] += len(clusters)
        counters["cross_page_clusters"] += sum(len(c["page_set"]) > 1 for c in clusters)
        speaker_for_balloon = [dict() for _ in range(3)]
        speaker_for_cluster = {}
        next_speaker = 1
        changed_pages = []
        for page_idx, (original, linked) in enumerate(zip(original_pages, link_pages)):
            if len(original) != len(linked):
                raise ValueError(f"Link count mismatch: {sid} page {page_idx+1}")
            new_page = []
            for base_row, link_row in zip(original, linked):
                if base_row["text"] != link_row["text"]:
                    raise ValueError(f"Text mismatch: {sid} page {page_idx+1}")
                balloon = link_row["balloon_region_index"]
                cached = speaker_for_balloon[page_idx]
                if balloon is not None and balloon in cached:
                    speaker = cached[balloon]
                else:
                    speaker = base_row["speaker"]
                    candidates = link_row["character_candidates"]
                    eligible = candidates if mode == "closest" else candidates[:1] if len(candidates) == 1 else []
                    if eligible and float(eligible[0]["distance_from_balloon"]) <= max_distance:
                        region_idx = int(eligible[0]["region_index"])
                        cluster_idx = clusters_by_region.get((page_idx, region_idx))
                        if cluster_idx is not None:
                            # Only trust clusters seen on more than one PAGE.
                            # Singleton/single-page crops fall back to the prior baseline.
                            if len(clusters[cluster_idx]["page_set"]) >= 2:
                                if cluster_idx not in speaker_for_cluster:
                                    speaker_for_cluster[cluster_idx] = f"visual{next_speaker}"
                                    next_speaker += 1
                                speaker = speaker_for_cluster[cluster_idx]
                                counters["visual_groups"] += 1
                    if balloon is not None:
                        cached[balloon] = speaker
                if speaker != base_row["speaker"]:
                    counters["changed_labels"] += 1
                new_page.append({"speaker": speaker, "text": base_row["text"]})
            changed_pages.append(new_page)
        outputs[sid] = changed_pages
    return outputs, dict(counters)


def main():
    required = [
        VAL / "validation_references.jsonl",
        VAL / "pipeline_predictions.jsonl",
        VAL / "first_occurrence_predictions.jsonl",
        VAL / "speaker_links_diagnostic.json",
        VISION / "segmentation_val.json",
        VISION / "appearance_embeddings_val.npz",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing prerequisite: {path}")

    references = score.load_jsonl(required[0], reference=True)
    baseline = score.load_jsonl(required[1])
    first_occurrence = score.load_jsonl(required[2])
    links_data = read_json(required[3])
    seg_data = read_json(required[4])
    table = embedding_table(required[5])
    if seg_data.get("split") != "val":
        raise ValueError("Incorrect segmentation split")
    links = {s["sequence_id"]: s["pages"] for s in links_data["sequences"]}
    vision = {s["sequence_id"]: s["pages"] for s in seg_data["sequences"]}
    if not (len(references) == len(baseline) == len(first_occurrence) == len(links) == len(vision) == 16):
        raise ValueError("Expected 16 validation sequences in every input")
    if not (set(references) == set(baseline) == set(first_occurrence) == set(links) == set(vision)):
        raise ValueError("Validation sequence IDs differ across inputs")
    for sid in baseline:
        if len(vision[sid]) != 3:
            raise ValueError(f"Expected three pages: {sid}")

    baseline_report = score.evaluate(references, baseline)["macro"]
    control_report = score.evaluate(references, first_occurrence)["macro"]
    if abs(baseline_report["balanced_joint_f1"] - 0.29931233548051034) > 1e-7:
        raise ValueError("Original baseline validation score differs")
    print("Original baseline verification: PASS")
    print("Original balanced joint F1:", round(baseline_report["balanced_joint_f1"], 4))
    print("First-occurrence control F1:", round(control_report["balanced_joint_f1"], 4))
    print("Character embeddings loaded:", len(table))

    # Verify that disabling visual assignments exactly reproduces the
    # previously tested first-occurrence balloon-consistency baseline.
    control, _ = visual_predictions(
        baseline, links, vision, table, threshold=1.01, mode="unique")
    if control != first_occurrence:
        raise RuntimeError("First-occurrence control predictions do not match")
    print("First-occurrence reproduction: PASS")
    print("\nMethod | Similarity threshold | Changed labels | Visual groups | Cross-page clusters | Balanced F1 | Delta vs original")
    print("-" * 122)
    results = {"baseline": baseline_report, "first_occurrence": control_report, "variants": []}
    for threshold in (0.80, 0.87, 0.93):
        for mode in ("unique", "closest"):
            predictions, counts = visual_predictions(
                baseline, links, vision, table,
                threshold=threshold, mode=mode,
            )
            for sid in baseline:
                for original_page, new_page in zip(baseline[sid], predictions[sid]):
                    if [row["text"] for row in original_page] != [row["text"] for row in new_page]:
                        raise ValueError("A variant changed the text/order")
            report = score.evaluate(references, predictions)
            metrics = report["macro"]
            if abs(metrics["text_order_score"] - baseline_report["text_order_score"]) > 1e-12:
                raise ValueError("Unexpected reading-order score change")
            print(f"{mode:7s} | {threshold:20.2f} | {counts.get('changed_labels',0):14d} | "
                  f"{counts.get('visual_groups',0):13d} | {counts.get('cross_page_clusters',0):19d} | "
                  f"{metrics['balanced_joint_f1']:.4f}      | "
                  f"{metrics['balanced_joint_f1'] - baseline_report['balanced_joint_f1']:+.4f}")
            result_name = f"embedding_speakers_{mode}_{int(round(threshold * 100))}"
            path = VAL / f"{result_name}_predictions.jsonl"
            write_jsonl(path, predictions)
            results["variants"].append({"method": mode, "threshold": threshold,
                                        "counts": counts, "scores": metrics,
                                        "prediction_file": path.name})
    output = VAL / "embedding_speaker_sweep.json"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("\nSaved:", output)
    print("Experimental only: production inference and BYTE test submission are untouched.")


if __name__ == "__main__":
    main()
