"""
Wayground - Split Berkelompok yang Stratified (Tahap 3)
======================================================

Mengapa perlu:
- Pada split lama (`train_test_split` stratified per baris), **52 teks
  `processed_text` identik muncul di development DAN holdout** (69 baris
  holdout, 11,3%). Itu kebocoran data: model bisa "menghafal" teks
  holdout.
- Duplikat berasal dari ulasan berbeda yang isinya sama persis
  (ulasan copy-paste / ulasan yang sama diunggah dua kali).

Solusi:
- Baris dikelompokkan berdasarkan `processed_text`.
- Setiap grup (duplikat) seluruhnya masuk ke satu split saja.
- Stratifikasi sejauh mungkin dijaga: grup diberi label dari label mayoritas
  grupnya, lalu `StratifiedGroupKFold` dipakai untuk membagi grup.

Split yang dihasilkan:
- development 80% baris, holdout 20% baris (dengan toleransi kecil
  karena grup tidak dapat dipecah).
- `random_state=42` -> dapat direproduksi.
- Holdout TIDAK pernah dipakai untuk pemilihan lexicon, hyperparameter,
  atau konfigurasi.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


DEFAULT_GROUP_COL = "processed_text"
DEFAULT_LABEL_COL = "sentiment_label_final"


def _normalise_group(value: object) -> str:
    """Kunci grup: teks kosong dikelompokkan sebagai satu grup."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "__group_kosong__"
    text = str(value).strip()
    return text if text else "__group_kosong__"


def build_groups(
    df: pd.DataFrame,
    group_col: str = DEFAULT_GROUP_COL,
) -> pd.Series:
    """Series kunci grup untuk setiap baris."""
    if group_col not in df.columns:
        raise KeyError(
            f"Kolom grup '{group_col}' tidak ada. "
            f"Kolom tersedia: {list(df.columns)}"
        )
    return df[group_col].map(_normalise_group)


def group_label(
    df: pd.DataFrame,
    group_col: str = DEFAULT_GROUP_COL,
    label_col: str = DEFAULT_LABEL_COL,
) -> pd.Series:
    """
    Label untuk setiap grup: label yang paling sering muncul di grup.

    Dipakai hanya untuk stratifikasi, bukan sebagai label data.
    """
    work = pd.DataFrame({
        "group": build_groups(df, group_col),
        "label": df[label_col].astype(str),
    })
    majority = (
        work.groupby("group")["label"]
        .agg(lambda s: s.value_counts().idxmax())
    )
    return work["group"].map(majority)


def duplicate_report(
    df: pd.DataFrame,
    group_col: str = DEFAULT_GROUP_COL,
) -> Dict[str, object]:
    """Statistik duplikasi berdasarkan `group_col`."""
    groups = build_groups(df, group_col)
    sizes = groups.value_counts()

    duplicate_groups = sizes[sizes > 1]
    rows_in_duplicate_groups = int(duplicate_groups.sum())
    rows_removed_if_dedup = rows_in_duplicate_groups - len(duplicate_groups)

    return {
        "total_baris": int(len(df)),
        "jumlah_grup": int(len(sizes)),
        "grup_duplikat": int(len(duplicate_groups)),
        "baris_di_grup_duplikat": rows_in_duplicate_groups,
        "baris_yang_akan_hilang_bila_dideduplikasi": int(rows_removed_if_dedup),
        "duplikat_maks_per_grup": int(sizes.max()) if len(sizes) else 0,
        "grup_dengan_kolase_label": int(
            df.assign(_g=groups)
            .groupby("_g")["sentiment_label_final"]
            .nunique()
            .gt(1)
            .sum()
        ) if DEFAULT_LABEL_COL in df.columns else 0,
    }


def grouped_stratified_split(
    df: pd.DataFrame,
    group_col: str = DEFAULT_GROUP_COL,
    label_col: str = DEFAULT_LABEL_COL,
    test_size: float = 0.20,
    random_state: int = 42,
    n_splits: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """
    Split berkelompok yang stratified semampunya.

    Mengembalikan (development, holdout, info).
    """
    if label_col not in df.columns:
        raise KeyError(f"Kolom label '{label_col}' tidak ada.")

    groups = build_groups(df, group_col)
    group_labels = group_label(df, group_col, label_col)

    # Jumlah fold = 1 / test_size (80:20 -> 5 fold, 1 fold = holdout).
    n_splits = max(2, int(round(1.0 / test_size)))

    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    indices = np.arange(len(df))
    folds = list(
        splitter.split(indices, group_labels.to_numpy(), groups.to_numpy())
    )

    # Pilih fold yang jumlah barisnya paling dekat dengan target holdout.
    target = test_size * len(df)
    best = min(
        folds,
        key=lambda f: abs(len(f[1]) - target),
    )
    dev_idx, hold_idx = best

    development = df.iloc[dev_idx].reset_index(drop=True)
    holdout = df.iloc[hold_idx].reset_index(drop=True)

    dev_groups = set(build_groups(development, group_col))
    hold_groups = set(build_groups(holdout, group_col))
    leaking = dev_groups & hold_groups

    info: Dict[str, object] = {
        "method": "StratifiedGroupKFold (grup = processed_text)",
        "group_column": group_col,
        "label_column": label_col,
        "random_state": random_state,
        "n_splits": n_splits,
        "test_size": test_size,
        "total_rows": int(len(df)),
        "development_rows": int(len(development)),
        "holdout_rows": int(len(holdout)),
        "holdout_used_for_selection": False,
        "jumlah_grup_total": int(groups.nunique()),
        "jumlah_grup_development": int(len(dev_groups)),
        "jumlah_grup_holdout": int(len(hold_groups)),
        "grup_tumpang_tindih": int(len(leaking)),
        "distribusi_kelas_total": df[label_col].value_counts().to_dict(),
        "distribusi_kelas_development": (
            development[label_col].value_counts().to_dict()
        ),
        "distribusi_kelas_holdout": (
            holdout[label_col].value_counts().to_dict()
        ),
    }
    info["duplikasi"] = duplicate_report(df, group_col)

    if leaking:
        raise AssertionError(
            "Split berkelompok bocor: "
            f"{len(leaking)} grup muncul di kedua split."
        )

    return development, holdout, info


def assert_no_overlap(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    group_col: str = DEFAULT_GROUP_COL,
) -> None:
    """Pastikan tidak ada `group_col` yang sama di kedua split."""
    dev = set(build_groups(development, group_col))
    hold = set(build_groups(holdout, group_col))
    overlap = dev & hold
    if overlap:
        raise AssertionError(
            f"{len(overlap)} nilai '{group_col}' muncul di development "
            "dan holdout. Ini kebocoran data."
        )


# ============================================================================
# TAHAP 5 - PEMUATAN SPLIT CANONICAL
# ============================================================================
#
# Temuan Tahap 5 (28 Okt 2025): `scripts/experiments/common.py` dan
# `scripts/train_final_model.py` masing-masing membuat split sendiri dengan
# `train_test_split` per baris. Akibatnya:
#   * holdout yang dipakai eksperimen BUKAN holdout canonical Stage 5,
#   * split per baris, dan mengembalikan kebocoran `processed_text` yang
#     sengaja dihilangkan Tahap 3.
#
# Fungsi ini menjadi SATU-SATUNYA sumber split:
#   1. Membaca file Stage 5 (development.csv / holdout.csv) bila ada.
#   2. Bila belum ada, menghitung ulang dengan `grouped_stratified_split`
#      (deterministik, random_state=42).
#
# Pemanggil WAJIB memakai fungsi ini, bukan `train_test_split` sendiri.

CANONICAL_SPLIT_DIRNAME = "stage5_split_v2"


def load_canonical_split(
    pipeline_output_dir: Optional[Path] = None,
    group_col: str = DEFAULT_GROUP_COL,
    label_col: str = DEFAULT_LABEL_COL,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """
    Memuat split canonical (berkelompok) dari Stage 5.

    Mengembalikan (development, holdout, info).
    """
    if pipeline_output_dir is None:
        from shared.constants import PIPELINE_OUTPUT_DIR

        pipeline_output_dir = PIPELINE_OUTPUT_DIR

    pipeline_output_dir = Path(pipeline_output_dir)
    split_dir = pipeline_output_dir / CANONICAL_SPLIT_DIRNAME

    dev_path = split_dir / "development.csv"
    hold_path = split_dir / "holdout.csv"
    meta_path = split_dir / "split_metadata.json"

    if dev_path.exists() and hold_path.exists():
        from shared.dataset_io import read_canonical_dataset

        development = read_canonical_dataset(dev_path, group_col, label_col)
        holdout = read_canonical_dataset(hold_path, group_col, label_col)

        # Sanity check: tidak boleh ada grup yang sama di kedua split.
        assert_no_overlap(development, holdout, group_col)

        info: Dict[str, object] = {
            "source": "stage5_files",
            "path": str(split_dir),
            "development_rows": int(len(development)),
            "holdout_rows": int(len(holdout)),
            "group_column": group_col,
            "label_column": label_col,
            "holdout_used_for_selection": False,
        }

        if meta_path.exists():
            try:
                info["metadata"] = json.loads(
                    meta_path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError:
                info["metadata"] = {"status": "tidak_valid"}

        return development, holdout, info

    # Fallback: hitung ulang split berkelompok secara deterministik.
    from shared.constants import CANONICAL_DATASET_PATH

    from shared.dataset_io import read_canonical_dataset

    df = read_canonical_dataset(CANONICAL_DATASET_PATH, group_col, label_col)
    development, holdout, info = grouped_stratified_split(
        df,
        group_col=group_col,
        label_col=label_col,
        test_size=test_size,
        random_state=random_state,
    )
    info["source"] = "recomputed"
    return development, holdout, info


# ============================================================================
# PEMECAHAN SPLIT UNTUK DATASET TERTENTU
# ============================================================================


def split_parts_for_dataset(
    df: pd.DataFrame,
    group_col: str = DEFAULT_GROUP_COL,
    label_col: str = DEFAULT_LABEL_COL,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """
    Mengembalikan bagian development/holdout dari split canonical yang
    DISELARASKAN dengan baris `df`.

    Baris dicocokkan lewat `reviewId` (bukan indeks) supaya aman terhadap
    urutan baris yang berbeda.

    Bila `reviewId` tidak tersedia atau himpunan reviewId tidak cocok,
    split dihitung ulang dengan `grouped_stratified_split` (deterministik).
    """
    try:
        development, holdout, info = load_canonical_split(
            group_col=group_col,
            label_col=label_col,
            test_size=test_size,
            random_state=random_state,
        )
    except (OSError, ValueError, KeyError, AssertionError) as exc:
        info = {"source": "tidak_tersedia", "error": str(exc)}
        development = df.iloc[0:0].copy()
        holdout = df.iloc[0:0].copy()

    if (
        "reviewId" in df.columns
        and "reviewId" in development.columns
        and "reviewId" in holdout.columns
    ):
        dev_ids = set(development["reviewId"].astype(str))
        hold_ids = set(holdout["reviewId"].astype(str))
        df_ids = set(df["reviewId"].astype(str))

        frames = (df, development, holdout)
        unique_ids = all(
            frame["reviewId"].notna().all()
            and frame["reviewId"].astype(str).str.strip().ne("").all()
            and not frame["reviewId"].astype(str).duplicated().any()
            for frame in frames
        )
        same_rows = (
            unique_ids and dev_ids and hold_ids and not (dev_ids & hold_ids)
            and (dev_ids | hold_ids) == df_ids
        )
        # Matching IDs alone is insufficient after preprocessing/relabeling.
        same_values = False
        if same_rows:
            def identity(frame):
                work = frame[["reviewId", group_col, label_col]].copy()
                work["reviewId"] = work["reviewId"].astype(str)
                work[group_col] = build_groups(frame, group_col)
                work[label_col] = work[label_col].astype(str).str.strip().str.lower()
                return work.set_index("reviewId").sort_index()

            same_values = identity(df).equals(
                identity(pd.concat([development, holdout], ignore_index=True))
            )

        if same_values:
            dev = df[df["reviewId"].astype(str).isin(dev_ids)].copy()
            hold = df[df["reviewId"].astype(str).isin(hold_ids)].copy()
            info = dict(info)
            info["source"] = "stage5_files"
            return dev, hold, info

    # Fallback: hitung split berkelompok pada dataset yang diberikan.
    development, holdout, info = grouped_stratified_split(
        df,
        group_col=group_col,
        label_col=label_col,
        test_size=test_size,
        random_state=random_state,
    )
    info = dict(info)
    info["source"] = "recomputed"
    info["cache_note"] = "Saved split unavailable or does not match all dataset IDs, texts and labels."
    return development, holdout, info


__all__ = [
    "DEFAULT_GROUP_COL",
    "DEFAULT_LABEL_COL",
    "assert_no_overlap",
    "build_groups",
    "duplicate_report",
    "group_label",
    "grouped_stratified_split",
    "load_canonical_split",
    "split_parts_for_dataset",
]
