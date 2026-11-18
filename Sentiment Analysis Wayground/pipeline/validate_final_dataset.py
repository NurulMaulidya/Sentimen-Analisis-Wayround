from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.constants import AUDIT_DIR  # noqa: E402
from shared.dataset_io import read_dataset
from shared.final_config import DATASET_RELATIVE_PATH, LABEL_COLUMN, TEXT_COLUMN, SENTIMENT_CLASSES

DATASET_PATH = PROJECT_ROOT / DATASET_RELATIVE_PATH
REPORT_PATH = AUDIT_DIR / "final_dataset_validation.json"
DISTRIBUTION_PATH = AUDIT_DIR / "final_dataset_distribution.csv"
DUPLICATES_PATH = AUDIT_DIR / "final_dataset_duplicates.csv"
CONFLICTS_PATH = AUDIT_DIR / "final_dataset_label_conflicts.csv"

EXPECTED_CLASSES = {str(x).casefold() for x in SENTIMENT_CLASSES}
LABEL_ALIASES = {"positive": "positif", "neutral": "netral", "negative": "negatif"}


def norm_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().casefold().split())


def clean_label(value: Any) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip().casefold()
    return LABEL_ALIASES.get(value, value)


def main() -> int:
    print("=" * 80)
    print("FINAL DATASET VALIDATION - WAYGROUND")
    print("=" * 80)
    print(f"Dataset : {DATASET_PATH.relative_to(PROJECT_ROOT)}")

    if not DATASET_PATH.exists():
        print(f"[FAIL] Dataset tidak ditemukan: {DATASET_PATH}")
        return 1

    try:
        df = read_dataset(DATASET_PATH).copy()
    except Exception as exc:
        print(f"[FAIL] Dataset gagal dibaca: {exc}")
        return 1

    required = {TEXT_COLUMN, LABEL_COLUMN}
    missing = sorted(required - set(df.columns))
    if missing:
        print(f"[FAIL] Kolom wajib tidak ditemukan: {missing}")
        print(f"       Kolom tersedia: {list(df.columns)}")
        return 1

    original_rows = len(df)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str).str.strip()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].map(clean_label)

    empty_text = int(df[TEXT_COLUMN].eq("").sum())
    empty_label = int(df[LABEL_COLUMN].eq("").sum())
    invalid_labels = sorted(set(df[LABEL_COLUMN]) - EXPECTED_CLASSES - {""})
    missing_classes = sorted(EXPECTED_CLASSES - set(df[LABEL_COLUMN]))

    duplicate_review_ids = 0
    duplicate_review_id_groups = 0
    duplicate_rows = []

    if "reviewId" in df.columns:
        review_id = df["reviewId"].fillna("").astype(str).str.strip()
        valid_review_id = review_id.ne("")
        dup_mask = valid_review_id & review_id.duplicated(keep=False)
        duplicate_review_ids = int(dup_mask.sum())
        if duplicate_review_ids:
            duplicate_review_id_groups = int(review_id[dup_mask].nunique())
            duplicate_rows.append(df.loc[dup_mask].assign(_duplicate_type="reviewId"))

    normalized_text = df[TEXT_COLUMN].map(norm_text)
    duplicate_mask = normalized_text.duplicated(keep=False)
    duplicate_text_rows = int(duplicate_mask.sum())
    duplicate_text_groups = int(normalized_text[duplicate_mask].nunique()) if duplicate_text_rows else 0

    if duplicate_text_rows:
        duplicate_rows.append(df.loc[duplicate_mask].assign(_duplicate_type="processed_text"))

    # Hanya duplicate processed_text dengan lebih dari satu label final yang dianggap konflik.
    tmp = pd.DataFrame({"_normalized_text": normalized_text, "_label": df[LABEL_COLUMN]})
    label_nunique = tmp.groupby("_normalized_text")["_label"].nunique()
    conflict_texts = set(label_nunique[label_nunique > 1].index) - {""}
    conflict_mask = normalized_text.isin(conflict_texts)
    conflict_rows = int(conflict_mask.sum())
    conflict_groups = len(conflict_texts)

    conflicts_df = df.loc[conflict_mask].copy()
    if not conflicts_df.empty:
        conflicts_df["_normalized_text"] = normalized_text[conflict_mask]
        conflicts_df = conflicts_df.sort_values(["_normalized_text", LABEL_COLUMN], kind="stable")

    distribution = (
        df[LABEL_COLUMN].value_counts()
        .reindex(sorted(EXPECTED_CLASSES), fill_value=0)
        .rename_axis("label").reset_index(name="count")
    )
    distribution["percentage"] = (distribution["count"] / max(original_rows, 1) * 100).round(4)

    expert_label_col = "sentiment_label_expert"
    expert_status_col = "expert_validation_status"
    expert_rows = None
    expert_coverage = None

    if expert_label_col in df.columns:
        expert_values = df[expert_label_col].fillna("").astype(str).str.strip()
        expert_rows = int(expert_values.ne("").sum())
    elif expert_status_col in df.columns:
        status = df[expert_status_col].fillna("").astype(str).str.casefold().str.strip()
        expert_rows = int((~status.isin({"", "none", "nan", "false", "belum", "tidak"})).sum())

    if expert_rows is not None and original_rows:
        expert_coverage = round(expert_rows / original_rows * 100, 4)

    duplicates_df = pd.concat(duplicate_rows, ignore_index=True).drop_duplicates() if duplicate_rows else pd.DataFrame()

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    distribution.to_csv(DISTRIBUTION_PATH, index=False, encoding="utf-8-sig")
    if not duplicates_df.empty:
        duplicates_df.to_csv(DUPLICATES_PATH, index=False, encoding="utf-8-sig")
    elif DUPLICATES_PATH.exists():
        DUPLICATES_PATH.unlink()

    if not conflicts_df.empty:
        conflicts_df.to_csv(CONFLICTS_PATH, index=False, encoding="utf-8-sig")
    elif CONFLICTS_PATH.exists():
        CONFLICTS_PATH.unlink()

    errors = []
    warnings = []
    if empty_text:
        errors.append(f"{empty_text} baris memiliki {TEXT_COLUMN} kosong")
    if empty_label:
        errors.append(f"{empty_label} baris memiliki {LABEL_COLUMN} kosong")
    if invalid_labels:
        errors.append(f"label tidak valid: {invalid_labels}")
    if missing_classes:
        errors.append(f"kelas tidak ditemukan: {missing_classes}")
    if duplicate_review_ids:
        errors.append(f"duplicate reviewId: {duplicate_review_ids} baris dalam {duplicate_review_id_groups} kelompok")
    if duplicate_text_rows:
        warnings.append(f"{duplicate_text_rows} baris memiliki duplicate {TEXT_COLUMN} dalam {duplicate_text_groups} kelompok; ini bukan otomatis duplicate review.")
    if conflict_rows:
        errors.append(f"{conflict_rows} baris dalam {conflict_groups} kelompok memiliki konflik label final pada {TEXT_COLUMN} yang sama")
    if expert_rows is None:
        warnings.append("Kolom validasi pakar tidak ditemukan; cakupan pakar tidak dapat dihitung.")

    report = {
        "status": "FAIL" if errors else "PASS",
        "dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "rows": original_rows,
        "columns": list(df.columns),
        "required_columns": sorted(required),
        "text_column": TEXT_COLUMN,
        "label_column": LABEL_COLUMN,
        "checks": {
            "empty_text": empty_text,
            "empty_label": empty_label,
            "invalid_labels": invalid_labels,
            "missing_classes": missing_classes,
            "duplicate_review_id_rows": duplicate_review_ids,
            "duplicate_review_id_groups": duplicate_review_id_groups,
            "duplicate_processed_text_rows": duplicate_text_rows,
            "duplicate_processed_text_groups": duplicate_text_groups,
            "label_conflict_rows": conflict_rows,
            "label_conflict_groups": conflict_groups,
        },
        "class_distribution": {
            str(row.label): {"count": int(row.count), "percentage": float(row.percentage)}
            for row in distribution.itertuples()
        },
        "expert_validation": {
            "label_column_present": expert_label_col in df.columns,
            "status_column_present": expert_status_col in df.columns,
            "validated_rows": expert_rows,
            "coverage_percentage": expert_coverage,
        },
        "errors": errors,
        "warnings": warnings,
        "outputs": {
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "distribution": str(DISTRIBUTION_PATH.relative_to(PROJECT_ROOT)),
            "duplicates": str(DUPLICATES_PATH.relative_to(PROJECT_ROOT)),
            "label_conflicts": str(CONFLICTS_PATH.relative_to(PROJECT_ROOT)),
        },
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[DATASET]")
    print(f"  Rows                    : {original_rows:,}")
    print(f"  Text column             : {TEXT_COLUMN}")
    print(f"  Label column            : {LABEL_COLUMN}")
    print(f"  Empty text              : {empty_text}")
    print(f"  Empty label             : {empty_label}")
    print(f"  Duplicate reviewId      : {duplicate_review_ids}")
    print(f"  Duplicate processed     : {duplicate_text_rows} baris / {duplicate_text_groups} kelompok")
    print(f"  Konflik label           : {conflict_rows} baris / {conflict_groups} kelompok")

    print("\n[DISTRIBUSI LABEL]")
    for row in distribution.itertuples():
        print(f"  {str(row.label):<8}: {int(row.count):>6} ({float(row.percentage):>7.3f}%)")

    print("\n[VALIDASI PAKAR]")
    if expert_rows is None:
        print("  Status                  : tidak dapat dihitung")
    else:
        print(f"  Baris tervalidasi       : {expert_rows:,}")
        print(f"  Coverage                : {expert_coverage:.3f}%")

    print("\n[PEMERIKSAAN DUPLICATE]")
    if duplicate_text_rows:
        print(f"  [WARN] {duplicate_text_rows} baris memiliki processed_text yang sama.")
        print("         Duplicate processed_text tidak otomatis dianggap duplicate review.")
    else:
        print("  [PASS] Tidak ada duplicate processed_text.")

    if conflict_rows:
        print(f"  [FAIL] Ditemukan {conflict_rows} baris dengan konflik label.")
        print(f"         Detail: {CONFLICTS_PATH}")
    else:
        print("  [PASS] Tidak ada konflik label pada processed_text yang sama.")

    for warning in warnings:
        print(f"[WARN] {warning}")

    if errors:
        print("\n[FAIL] FINAL DATASET INVALID")
        for error in errors:
            print(f"  - {error}")
        print(f"\nReport      : {REPORT_PATH}")
        print(f"Distribution: {DISTRIBUTION_PATH}")
        if not conflicts_df.empty:
            print(f"Conflicts   : {CONFLICTS_PATH}")
        return 1

    print("\n[PASS] FINAL DATASET VALID")
    print("  Duplicate processed_text tanpa konflik dipertahankan sebagai WARNING.")
    print(f"  Report       : {REPORT_PATH}")
    print(f"  Distribution : {DISTRIBUTION_PATH}")
    if duplicate_text_rows:
        print(f"  Duplicates   : {DUPLICATES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
