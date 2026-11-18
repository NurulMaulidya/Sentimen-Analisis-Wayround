"""Build an isolated, reproducible preparation snapshot; never promote a model.

Run: python scripts/prepare_dataset_snapshot.py
Existing output directories are rejected, including partially written snapshots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.constants import FINAL_OUTPUT_COLUMNS, SLANG_DICT_PATH, NORMALIZATION_DICT_PATH
from shared.data_contract import filter_research_period
from shared.dataset_io import read_dataset
from shared.labeling import label_text, lexicon_info
from shared.preprocessing import preprocess_text
from shared.research_config import PREPROCESSING_VERSION, RESEARCH_START, RESEARCH_END_EXCLUSIVE, BASELINE_NEUTRAL_THRESHOLD
from shared.splitting import grouped_stratified_split, duplicate_report
from pipeline.integrate_expert_labels import (
    read_expert_lama, read_expert_baru, map_expert_lama, normalize_text,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def attach_experts(frame, old, new):
    """Accept exact text/ID matches only; export all other cases for review."""
    result = frame.copy()
    result["sentiment_label_expert"] = ""
    result["sentiment_label_expert_2"] = ""
    result["label_source"] = "Lexicon"
    result["label_source_detail"] = "automatic_baseline"
    result["expert_match_method"] = ""
    result["expert_independence_verified"] = False
    result["_content_norm"] = result.content.map(normalize_text)
    mapping, counters = map_expert_lama(old, result)
    pending = []
    # Aggregate before applying labels: duplicate/conflicting annotation rows
    # cannot be resolved by whichever row happens to be last in a file.
    accepted = {}
    for _, row in mapping.iterrows():
        if row.match_method == "content_exact":
            accepted.setdefault(row.review_id, set()).add(row.expert_label)
        else:
            pending.append({"source": "old", **row.to_dict()})
    for rid, labels in accepted.items():
        if len(labels) != 1:
            pending.append({"source": "old", "review_id": rid, "match_method": "conflicting_labels"})
            continue
        mask = result.reviewId.eq(rid)
        result.loc[mask, "sentiment_label_expert"] = next(iter(labels))
        result.loc[mask, "expert_match_method"] = "content_exact"
        result.loc[mask, "label_source_detail"] = "old_expert_exact"

    for rid, annotations in new.groupby("review_id", sort=True):
        filled = annotations[
            annotations.label_pakar_1.notna() | annotations.label_pakar_2.notna()
        ]
        if filled.empty:
            continue
        mask = result.reviewId.eq(rid)
        if not mask.any():
            pending.append({"source": "new", "review_id": rid, "match_method": "id_not_found"})
            continue
        content = result.loc[mask, "_content_norm"].iloc[0]
        if filled.text.eq("").any() or not filled.text.eq(content).all():
            pending.append({"source": "new", "review_id": rid, "match_method": "content_not_verified"})
            continue
        labels = set(filled.label_pakar_1.dropna()) | set(filled.label_pakar_2.dropna())
        existing = result.loc[mask, "sentiment_label_expert"].iloc[0]
        if len(labels) != 1 or (existing and existing not in labels):
            pending.append({"source": "new", "review_id": rid, "match_method": "conflicting_labels"})
            continue
        result.loc[mask, "sentiment_label_expert"] = next(iter(labels))
        second = filled.label_pakar_2.dropna()
        if not second.empty:
            result.loc[mask, "sentiment_label_expert_2"] = second.iloc[0]
        result.loc[mask, "expert_match_method"] = "reviewId_and_content_exact"
        result.loc[mask, "label_source_detail"] = "annotation_sheet_id_and_text_verified; independence_unverified"

    has_expert = result.sentiment_label_expert.ne("")
    result.loc[has_expert, "label_source"] = "Expert"
    result["sentiment_label_final"] = result.sentiment_label_inset.where(
        ~has_expert, result.sentiment_label_expert
    )
    result["expert_validation_status"] = has_expert.map({True: "Exact match", False: "Not validated"})
    return result.drop(columns="_content_norm"), mapping, pd.DataFrame(pending), counters


def build_snapshot(source, old_path, new_path, output_dir):
    source, old_path, new_path, output_dir = map(Path, (source, old_path, new_path, output_dir))
    output_dir = output_dir.resolve()
    if not any(output_dir.is_relative_to(ROOT / name) for name in ("data", "runs")):
        raise ValueError("Snapshot output must be inside the project data/ or runs/ directory")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite snapshot: {output_dir}")
    inputs = [source, old_path, new_path]
    before = {str(p.resolve()): sha256(p) for p in inputs}
    raw = read_dataset(source)
    initial, outside = filter_research_period(raw)
    initial = initial[FINAL_OUTPUT_COLUMNS].copy().reset_index(drop=True)
    initial["at"] = initial["at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    # Dedup normalized raw content with a stable order independent of CSV rows.
    initial = initial.sort_values(["at", "reviewId"], kind="stable").reset_index(drop=True)
    keys = initial.content.map(normalize_text)
    duplicates = initial.loc[keys.duplicated()].copy()
    initial = initial.loc[~keys.duplicated()].reset_index(drop=True)
    processed = initial.copy()
    processed["processed_text"] = processed.content.map(preprocess_text)
    empty = processed.loc[processed.processed_text.str.strip().eq("")].copy()
    processed = processed.loc[processed.processed_text.str.strip().ne("")].reset_index(drop=True)
    labeled = processed.copy()
    results = labeled.processed_text.map(lambda text: label_text(text, threshold=BASELINE_NEUTRAL_THRESHOLD))
    for column, key in [("sentiment_label_inset", "label"), ("sentiment_score", "raw_score"),
                        ("sentiment_score_normalized", "normalized_score"),
                        ("lexicon_version", "lexicon_version"), ("neutral_threshold", "threshold")]:
        labeled[column] = results.map(lambda r: r[key])
    labeled["preprocessing_version"] = PREPROCESSING_VERSION
    old, new = read_expert_lama(old_path), read_expert_baru(new_path)
    final, mapping, pending, mapping_stats = attach_experts(labeled, old, new)
    dev, hold, split = grouped_stratified_split(final)
    duplicates_report = duplicate_report(final)
    code_paths = [ROOT / p for p in [
        "shared/preprocessing.py", "shared/data_contract.py", "shared/labeling.py",
        "shared/lexicon.py", "shared/research_config.py", "shared/splitting.py",
        "scripts/prepare_dataset_snapshot.py", "pipeline/integrate_expert_labels.py",
    ]]
    lexicon_paths = sorted((ROOT / "data/lexicon").rglob("*.tsv"))
    outputs = {
        "stage1_scraped.csv": initial, "stage2_preprocessed_new.csv": processed,
        "stage3_labeled.csv": labeled, "stage3_final_labeled.csv": final,
        "audit/excluded_period.csv": outside, "audit/duplicate_content.csv": duplicates,
        "audit/empty_processed.csv": empty, "audit/expert_mapping_candidates.csv": mapping,
        "audit/annotation_review_queue.csv": pending,
        "stage5_split_v2/development.csv": dev, "stage5_split_v2/holdout.csv": hold,
    }
    after = {str(p.resolve()): sha256(p) for p in inputs}
    if before != after:
        raise RuntimeError("An input changed while preparing the snapshot")
    manifest = {
        "status": "prepared_not_promoted", "preprocessing_version": PREPROCESSING_VERSION,
        "period": {"start_inclusive": RESEARCH_START, "end_exclusive": RESEARCH_END_EXCLUSIVE},
        "lexicon": lexicon_info(), "threshold_status": "baseline_pending_experiment_2",
        "rows": {"source": len(raw), "in_period_unique_content": len(initial), "final": len(final),
                 "excluded_period": len(outside), "empty_processed": len(empty), "duplicate_content": len(duplicates)},
        "class_distribution": final.sentiment_label_final.value_counts().to_dict(),
        "expert_rows": int(final.sentiment_label_expert.ne("").sum()),
        "second_annotator_rows": int(final.sentiment_label_expert_2.ne("").sum()),
        "annotation_mapping": mapping_stats, "pending_annotation_cases": len(pending),
        "duplicates": duplicates_report, "split": split,
        "holdout_independence_verified": False,
        "holdout_note": "Group-isolated prospective split; historical lexicon/annotation development exposure is unknown. Not an independently annotated gold test set.",
        "input_hashes": before,
        "code_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in code_paths},
        "resource_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in [Path(SLANG_DICT_PATH), Path(NORMALIZATION_DICT_PATH), *lexicon_paths]},
    }
    manifest["lexicon"]["threshold"] = BASELINE_NEUTRAL_THRESHOLD
    output_dir.mkdir(parents=True)
    for name, frame in outputs.items():
        path = output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    split_path = output_dir / "stage5_split_v2/split_metadata.json"
    split_path.write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["output_hashes"] = {name: sha256(output_dir / name) for name in [*outputs, "stage5_split_v2/split_metadata.json"]}
    # Written last: a partial directory without this manifest is not a snapshot.
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/02_data_akhir.csv")
    parser.add_argument("--expert-old", type=Path, default=ROOT / "dataset/Hasil_Validasi_Pakar.tsv")
    parser.add_argument("--expert-new", type=Path, default=ROOT / "dataset/anotasi_tambahan.tsv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/pipeline_output_v3")
    args = parser.parse_args()
    report = build_snapshot(args.source, args.expert_old, args.expert_new, args.output)
    print(json.dumps({k: report[k] for k in ["status", "rows", "class_distribution", "expert_rows", "pending_annotation_cases"]}, indent=2))


if __name__ == "__main__":
    main()
