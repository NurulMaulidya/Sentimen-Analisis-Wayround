"""
Verifikasi alur data end-to-end pada state CANONICAL v4.

CATATAN SEJARAH
---------------
Versi test ini pernah memverifikasi path legacy:

    data/pipeline_output/stage*.csv
    data/pipeline_output/stage5_split/{train,test}.csv
    models/svm_model_latest.pkl + models/vectorizer_latest.pkl
    models/metadata_latest.json

Path tersebut sudah SUPERSEDED. Artefak canonical sekarang:

    data/pipeline_output_v4/stage3_final_labeled.csv      (3078 baris)
    data/pipeline_output_v4/stage5_split_v2/development.csv (2474 baris)
    data/pipeline_output_v4/stage5_split_v2/holdout.csv     (604 baris)
    models/final_v4/final_pipeline.joblib
    models/final_v4/metadata.json

Test ini sengaja TIDAK memakai dataset lama (3065 / 3060 baris).
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.constants import (
    CANONICAL_DATASET_PATH,
    PIPELINE_OUTPUT_DIR,
)
from shared.deployment_integrity import sha256, canonical
from shared.final_config import (
    FINAL_MODEL_CONFIG,
    FINAL_PIPELINE_PATH,
    LABEL_COLUMN,
    PROJECT_ROOT,
    TEXT_COLUMN,
)
from shared.labels import normalize_label

EXPECTED_TOTAL = 3078
EXPECTED_DEVELOPMENT = 2474
EXPECTED_HOLDOUT = 604

SPLIT_DIR = PIPELINE_OUTPUT_DIR / "stage5_split_v2"
DEVELOPMENT_PATH = SPLIT_DIR / "development.csv"
HOLDOUT_PATH = SPLIT_DIR / "holdout.csv"


def test_canonical_dataset_exists_and_has_expected_rows():
    """Dataset canonical v4 ada dan memiliki 3078 baris."""
    assert CANONICAL_DATASET_PATH.exists(), (
        f"Dataset canonical tidak ditemukan: {CANONICAL_DATASET_PATH}"
    )
    df = pd.read_csv(CANONICAL_DATASET_PATH)
    assert len(df) == EXPECTED_TOTAL, (
        f"Dataset canonical harus {EXPECTED_TOTAL} baris, ditemukan {len(df)}"
    )
    assert TEXT_COLUMN in df.columns
    assert LABEL_COLUMN in df.columns


def test_canonical_dataset_label_distribution():
    """Distribusi label canonical sesuai metadata final v4."""
    df = pd.read_csv(CANONICAL_DATASET_PATH)
    counts = df[LABEL_COLUMN].map(normalize_label).value_counts().to_dict()
    assert counts == {"positif": 1947, "netral": 556, "negatif": 575}, counts


def test_split_files_match_dataset():
    """Split development/holdout utuh dan berjumlah ke total dataset."""
    assert DEVELOPMENT_PATH.exists(), f"Development split tidak ditemukan: {DEVELOPMENT_PATH}"
    assert HOLDOUT_PATH.exists(), f"Holdout split tidak ditemukan: {HOLDOUT_PATH}"

    development = pd.read_csv(DEVELOPMENT_PATH)
    holdout = pd.read_csv(HOLDOUT_PATH)

    assert len(development) == EXPECTED_DEVELOPMENT, len(development)
    assert len(holdout) == EXPECTED_HOLDOUT, len(holdout)
    assert len(development) + len(holdout) == EXPECTED_TOTAL


def test_split_has_no_overlap():
    """Tidak ada baris yang sama di development dan holdout."""
    development = pd.read_csv(DEVELOPMENT_PATH)
    holdout = pd.read_csv(HOLDOUT_PATH)

    key = "reviewId" if "reviewId" in development.columns else TEXT_COLUMN
    development_keys = set(development[key].astype(str))
    holdout_keys = set(holdout[key].astype(str))

    overlap = development_keys & holdout_keys
    assert not overlap, (
        f"Development dan holdout bertumpuk pada {len(overlap)} baris: "
        f"{sorted(overlap)[:5]}"
    )


def test_final_artifact_and_metadata():
    """Artefak final canonical ada dan terikat ke metadata."""
    assert FINAL_PIPELINE_PATH.exists(), (
        f"Artefak final tidak ditemukan: {FINAL_PIPELINE_PATH}"
    )

    metadata_path = FINAL_PIPELINE_PATH.parent / "metadata.json"
    assert metadata_path.exists(), f"Metadata tidak ditemukan: {metadata_path}"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    assert metadata["status"] == "FINAL"
    assert metadata["holdout_used_for_selection"] is False
    # Perbandingan memakai normalisasi JSON supaya tuple/list dianggap sama.
    assert canonical(metadata["model"]) == canonical(FINAL_MODEL_CONFIG)

    dataset = metadata["dataset"]
    assert dataset["total_samples"] == EXPECTED_TOTAL
    assert dataset["development_samples"] == EXPECTED_DEVELOPMENT
    assert dataset["holdout_samples"] == EXPECTED_HOLDOUT


def test_final_artifact_hash_matches_metadata():
    """Hash artefak final masih cocok dengan metadata (artifact tidak berubah)."""
    metadata_path = FINAL_PIPELINE_PATH.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))

    recorded = metadata["artifact_hashes"]
    key = str(
        (PROJECT_ROOT / FINAL_PIPELINE_PATH).resolve().relative_to(PROJECT_ROOT.resolve())
    )
    if key not in recorded:
        # Metadata mungkin memakai separator path platform lain.
        key = next(k for k in recorded if k.replace("\\", "/").endswith("final_pipeline.joblib"))

    assert sha256(FINAL_PIPELINE_PATH) == recorded[key], (
        "Artefak final_pipeline.joblib berubah sejak metadata dibuat."
    )


def test_dataset_hash_matches_metadata():
    """Hash dataset canonical masih cocok dengan metadata."""
    metadata_path = FINAL_PIPELINE_PATH.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))

    recorded = metadata["artifact_hashes"]
    key = next(
        k for k in recorded
        if k.replace("\\", "/").endswith("stage3_final_labeled.csv")
    )
    assert sha256(CANONICAL_DATASET_PATH) == recorded[key], (
        "Dataset canonical berubah sejak metadata dibuat."
    )


if __name__ == '__main__':
    print('=== END-TO-END DATA FLOW VERIFICATION (CANONICAL v4) ===')
    test_canonical_dataset_exists_and_has_expected_rows()
    print('  [STEP 1] Canonical dataset (3078 rows): OK')
    test_split_files_match_dataset()
    print('  [STEP 2] Split (2474 development / 604 holdout): OK')
    test_split_has_no_overlap()
    print('  [STEP 3] Split has no overlap: OK')
    test_final_artifact_and_metadata()
    print('  [STEP 4] Final artifact + metadata: OK')
    print('=== DATA FLOW VERIFICATION COMPLETE: ALL PASS ===')
