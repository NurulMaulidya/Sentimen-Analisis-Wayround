"""Experiment inputs come from the snapshot, never from a final model config."""

import hashlib
import json
from pathlib import Path

from shared.dataset_io import read_canonical_dataset
from shared.splitting import split_parts_for_dataset


def snapshot_sizes(path):
    path = Path(path)
    manifest_path = path.parent / "dataset_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        for name in ["stage3_final_labeled.csv", "stage5_split_v2/development.csv", "stage5_split_v2/holdout.csv"]:
            actual = hashlib.sha256((path.parent / name).read_bytes()).hexdigest()
            if actual != manifest["output_hashes"][name]:
                raise ValueError(f"Snapshot artifact changed: {name}")
        return (manifest["rows"]["final"], manifest["split"]["development_rows"], manifest["split"]["holdout_rows"])
    frame = read_canonical_dataset(path)
    dev, hold, _ = split_parts_for_dataset(frame)
    return len(frame), len(dev), len(hold)
