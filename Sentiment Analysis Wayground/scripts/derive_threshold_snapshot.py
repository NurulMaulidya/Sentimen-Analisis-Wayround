"""Derive labels from a selected development threshold, preserving split IDs."""

import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
from shared.labeling import label_text
from shared.splitting import assert_no_overlap, duplicate_report


def derive(source, destination, experiment):
    source, destination, experiment = map(lambda p: Path(p).resolve(), (source, destination, experiment))
    if not any(destination.is_relative_to(ROOT / name) for name in ("data", "runs")):
        raise ValueError("Output must be inside project data/ or runs/")
    if destination.exists():
        raise FileExistsError(destination)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    parent = json.loads((source / "dataset_manifest.json").read_text())
    for name, expected in parent["output_hashes"].items():
        if sha(source / name) != expected:
            raise ValueError(f"Source artifact changed: {name}")
    selection = json.loads(experiment.read_text())
    if selection["expert_validation"].get("selection_partition") != "development_only":
        raise ValueError("Threshold must be selected on development only")
    threshold = selection["configuration"]["selected_threshold"]
    frame = pd.read_csv(source / "stage3_final_labeled.csv", encoding="utf-8-sig")
    results = frame.processed_text.map(lambda t: label_text(t, threshold=threshold))
    frame["sentiment_label_inset"] = results.map(lambda r: r["label"])
    frame["neutral_threshold"] = threshold
    expert = frame.sentiment_label_expert.fillna("")
    frame["sentiment_label_final"] = frame.sentiment_label_inset.where(expert.eq(""), expert)
    dev_ids = set(pd.read_csv(source / "stage5_split_v2/development.csv").reviewId)
    hold_ids = set(pd.read_csv(source / "stage5_split_v2/holdout.csv").reviewId)
    dev = frame[frame.reviewId.isin(dev_ids)].copy()
    hold = frame[frame.reviewId.isin(hold_ids)].copy()
    assert_no_overlap(dev, hold)
    if len(dev) + len(hold) != len(frame) or dev_ids & hold_ids:
        raise ValueError("Incomplete or overlapping ID split")
    shutil.copytree(source, destination)
    # Remove the copied completion marker before updating any artifacts.
    (destination / "dataset_manifest.json").unlink()
    frame.to_csv(destination / "stage3_final_labeled.csv", index=False, encoding="utf-8-sig")
    initial = pd.read_csv(destination / "stage3_labeled.csv", encoding="utf-8-sig")
    initial["sentiment_label_inset"] = initial.reviewId.map(frame.set_index("reviewId").sentiment_label_inset)
    initial["neutral_threshold"] = threshold
    initial.to_csv(destination / "stage3_labeled.csv", index=False, encoding="utf-8-sig")
    for name, part in [("development", dev), ("holdout", hold)]:
        part.to_csv(destination / f"stage5_split_v2/{name}.csv", index=False, encoding="utf-8-sig")
    metadata = copy.deepcopy(parent["split"])
    for key, part in [("total", frame), ("development", dev), ("holdout", hold)]:
        metadata[f"distribusi_kelas_{key}"] = part.sentiment_label_final.value_counts().to_dict()
    metadata["partition_policy"] = "Inherited exact reviewId split from preparation snapshot; no holdout reselection after threshold tuning"
    metadata["duplikasi"] = duplicate_report(frame)
    (destination / "stage5_split_v2/split_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    manifest = copy.deepcopy(parent)
    manifest.update({"status": "threshold_selected_not_promoted", "split": metadata,
                     "threshold_status": "selected_on_development", "selected_threshold": threshold,
                     "class_distribution": frame.sentiment_label_final.value_counts().to_dict(),
                     "duplicates": duplicate_report(frame),
                     "parent_snapshot": str(source.relative_to(ROOT)),
                     "parent_manifest_sha256": sha(source / "dataset_manifest.json"),
                     "threshold_experiment": str(experiment.relative_to(ROOT)),
                     "threshold_experiment_sha256": sha(experiment),
                     "derivation_code_sha256": sha(__file__)})
    manifest["lexicon"]["threshold"] = threshold
    manifest["output_hashes"] = {name: sha(destination / name) for name in parent["output_hashes"]}
    (destination / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--experiment", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(derive(a.source, a.output, a.experiment)["class_distribution"], indent=2))
