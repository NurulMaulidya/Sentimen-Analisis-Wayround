"""
Common utilities untuk Experiment 1-6.

Tujuan:
- Menyatukan path dataset canonical.
- Menyatukan validasi dataset.
- Menyatukan deteksi kolom.
- Menyatukan normalisasi label.
- Menyatukan fixed split 80:20.
- Menyatukan utilitas output JSON dan distribusi label.
- Menghindari duplikasi kode antar eksperimen.

Catatan metodologis:
- Dataset canonical adalah stage3_final_labeled.csv.
- Split final ditetapkan 80:20 dengan random_state=42 dan stratifikasi.
- Data holdout 20% tidak digunakan untuk pemilihan konfigurasi eksperimen.
- Konfigurasi TF-IDF/SVM final tidak disimpan di file ini.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from shared.constants import (  # noqa: E402
    CANONICAL_DATASET_PATH,
    EXPERIMENTS_DIR,
)

# Dataset canonical mengikuti PIPELINE_OUTPUT_DIR (Tahap 5) supaya
# eksperimen baru memakai dataset berversi tanpa mengubah kode.
DATASET_PATH = CANONICAL_DATASET_PATH

OUTPUT_DIR = EXPERIMENTS_DIR


# ============================================================
# FIXED METHODOLOGY
# ============================================================

RANDOM_STATE = 42
CV_FOLDS = 5

FINAL_TRAIN_RATIO = 0.80
FINAL_TEST_RATIO = 0.20

# Kolom yang dipakai sebagai grup pada split berkelompok (Tahap 3/5).
# Semua baris dengan `processed_text` sama wajib masuk satu split saja.
GROUP_COLUMN = "processed_text"

TEXT_COLUMN_CANDIDATES = (
    "processed_text",
    "preprocessed_text",
    "clean_text",
    "content",
    "review",
    "text",
)

LABEL_COLUMN_CANDIDATES = (
    "sentiment_label_final",
    "label",
    "sentiment",
    "sentiment_normalized",
)

VALID_LABELS = (
    "positif",
    "netral",
    "negatif",
)


# ============================================================
# HEADER
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# COLUMN DETECTION
# ============================================================

def detect_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    column_type: str,
) -> str:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "detect_column() membutuhkan pandas.DataFrame, "
            f"tetapi menerima {type(df).__name__}."
        )

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Kolom {column_type} tidak ditemukan.\n"
        f"Kandidat: {list(candidates)}\n"
        f"Kolom tersedia: {list(df.columns)}"
    )


def detect_text_column(df: pd.DataFrame) -> str:
    return detect_column(
        df,
        TEXT_COLUMN_CANDIDATES,
        "teks",
    )


def detect_label_column(df: pd.DataFrame) -> str:
    return detect_column(
        df,
        LABEL_COLUMN_CANDIDATES,
        "label sentimen",
    )


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_label(value) -> str | None:
    if value is None or pd.isna(value):
        return None

    value = str(value).strip().lower()

    mapping = {
        "positif": "positif",
        "positive": "positif",
        "pos": "positif",

        "netral": "netral",
        "neutral": "netral",
        "neu": "netral",

        "negatif": "negatif",
        "negative": "negatif",
        "neg": "negatif",
    }

    return mapping.get(value)


def normalize_labels(series: pd.Series) -> pd.Series:
    return series.map(normalize_label)


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset(
    data_path: str | Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    """
    Membaca dataset dan mengembalikan DataFrame serta path aktual.

    sep=None + engine='python' dipertahankan karena dataset
    pipeline saat ini perlu autodetection delimiter.
    """

    if data_path is None:
        path = DATASET_PATH
    else:
        path = Path(data_path)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan:\n{path}"
        )

    try:
        from shared.dataset_io import read_dataset

        df = read_dataset(path)
    except Exception as exc:
        raise RuntimeError(
            f"Gagal membaca dataset:\n{path}\n"
            f"Error: {exc}"
        ) from exc

    if df.empty:
        raise ValueError(
            f"Dataset kosong:\n{path}"
        )

    return df, path


# ============================================================
# DATASET VALIDATION
# ============================================================

def prepare_dataset(
    df: pd.DataFrame,
    text_column: str | None = None,
    label_column: str | None = None,
    normalize_label_values: bool = True,
) -> tuple[pd.DataFrame, str, str]:
    """
    Membersihkan dan memvalidasi dataset untuk eksperimen.

    Fungsi tidak melakukan:
    - deduplikasi,
    - balancing,
    - oversampling,
    - undersampling,
    - perubahan threshold,
    - feature extraction.

    Hal-hal tersebut harus dilakukan pada tahap eksperimen
    masing-masing agar metodologi tetap dapat dilacak.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "prepare_dataset() membutuhkan pandas.DataFrame, "
            f"tetapi menerima {type(df).__name__}."
        )

    df = df.copy()

    if text_column is None:
        text_column = detect_text_column(df)

    if label_column is None:
        label_column = detect_label_column(df)

    if text_column not in df.columns:
        raise ValueError(
            f"Kolom teks '{text_column}' tidak ditemukan."
        )

    if label_column not in df.columns:
        raise ValueError(
            f"Kolom label '{label_column}' tidak ditemukan."
        )

    df[text_column] = (
        df[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if normalize_label_values:
        df[label_column] = normalize_labels(
            df[label_column]
        )
    else:
        df[label_column] = (
            df[label_column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    df = df[
        (df[text_column] != "")
        & df[label_column].isin(VALID_LABELS)
    ].copy()

    if df.empty:
        raise ValueError(
            "Tidak ada data valid setelah validasi dataset."
        )

    class_counts = df[label_column].value_counts()

    missing_classes = [
        label
        for label in VALID_LABELS
        if label not in class_counts.index
    ]

    if missing_classes:
        raise ValueError(
            "Dataset tidak memiliki seluruh kelas sentimen. "
            f"Kelas yang hilang: {missing_classes}"
        )

    if class_counts.min() < CV_FOLDS:
        raise ValueError(
            f"Kelas terkecil hanya memiliki "
            f"{class_counts.min()} data, sedangkan "
            f"CV membutuhkan minimal {CV_FOLDS} data per kelas."
        )

    return df, text_column, label_column


# ============================================================
# CANONICAL DATASET
# ============================================================

def load_canonical_dataset() -> tuple[pd.DataFrame, str, str]:
    """
    Memuat canonical dataset penelitian.

    Return:
        df, text_column, label_column
    """

    df, path = load_dataset(DATASET_PATH)

    df, text_column, label_column = prepare_dataset(
        df,
        text_column="processed_text",
        label_column="sentiment_label_final",
        normalize_label_values=True,
    )

    print()
    print("=" * 70)
    print("CANONICAL DATASET")
    print("=" * 70)
    print(f"Path       : {path}")
    print(f"Rows       : {len(df)}")
    print(f"Columns    : {len(df.columns)}")
    print(f"Text       : {text_column}")
    print(f"Label      : {label_column}")

    print()
    print("Distribusi label:")
    print(df[label_column].value_counts().to_string())

    return df, text_column, label_column


# ============================================================
# FIXED 80:20 SPLIT
# ============================================================

def make_fixed_split(
    df: pd.DataFrame,
    text_column: str | None = None,
    label_column: str | None = None,
    return_groups: bool = False,
):
    """
    Membuat fixed 80:20 development/holdout split.

    TAHAP 5: split ini sekarang SAMA dengan split canonical Stage 5
    (berkelompok berdasarkan `processed_text`), bukan `train_test_split`
    per baris. Alasannya:
      * holdout eksperimen harus sama dengan holdout canonical, supaya
        hasil eksperimen dan model final dibandingkan apples-to-apples;
      * split per baris mengembalikan kebocoran `processed_text` yang
        sengaja dihilangkan Tahap 3.

    Split diambil dari file Stage 5 bila tersedia. Bila dataset yang
    diberikan berbeda dari dataset canonical, split dihitung ulang dengan
    `grouped_stratified_split` (deterministik, random_state=42).

    Development 80%:
        Digunakan oleh eksperimen untuk cross-validation.

    Holdout 20%:
        Tidak digunakan untuk pemilihan konfigurasi model.

    random_state:
        42
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "make_fixed_split() membutuhkan pandas.DataFrame, "
            f"tetapi menerima {type(df).__name__}."
        )

    if text_column is None:
        text_column = detect_text_column(df)

    if label_column is None:
        label_column = detect_label_column(df)

    if text_column not in df.columns:
        raise ValueError(
            f"Kolom teks '{text_column}' tidak ditemukan."
        )

    if label_column not in df.columns:
        raise ValueError(
            f"Kolom label '{label_column}' tidak ditemukan."
        )

    X = df[text_column].astype(str)
    y = df[label_column].astype(str)

    # ------------------------------------------------------------------
    # Tahap 5: pakai split canonical Stage 5 (berkelompok).
    # ------------------------------------------------------------------
    from shared.splitting import split_parts_for_dataset

    development, holdout, split_info = split_parts_for_dataset(
        df,
        group_col=GROUP_COLUMN,
        label_col=label_column,
        test_size=FINAL_TEST_RATIO,
        random_state=RANDOM_STATE,
    )
    if (DATASET_PATH.parent / "dataset_manifest.json").exists() and split_info.get("source") != "stage5_files":
        raise ValueError("Versioned experiment must use the saved canonical split, not a recomputed partition")

    X_train = development[text_column].astype(str)
    y_train = development[label_column].astype(str)
    X_test = holdout[text_column].astype(str)
    y_test = holdout[label_column].astype(str)

    print()
    print("[FIXED SPLIT]")
    print(f"Train        : {len(X_train)} ({FINAL_TRAIN_RATIO:.0%})")
    print(f"Test         : {len(X_test)} ({FINAL_TEST_RATIO:.0%})")
    print(f"Random State : {RANDOM_STATE}")
    print("Stratify     : True")
    print(
        "Metode       : "
        f"{split_info.get('method', split_info.get('source', 'canonical'))}"
    )
    print(
        "Sumber       : "
        f"{split_info.get('source', 'stage5_files')} "
        f"({split_info.get('path', '')})"
    )

    if return_groups:
        # Grup baris development (kunci `processed_text`) untuk CV
        # berkelompok. Tanpa ini, duplikat `processed_text` bisa muncul
        # di fold latih dan fold validasi pada saat yang sama.
        from shared.splitting import build_groups

        groups_train = build_groups(development, GROUP_COLUMN).to_numpy()
        return X_train, X_test, y_train, y_test, groups_train

    return X_train, X_test, y_train, y_test


# ============================================================
# CROSS-VALIDATION BERKELOMPOK (Tahap 5)
# ============================================================

def make_grouped_cv(n_splits: int = CV_FOLDS):
    """
    Cross-validation berkelompok (grup = `processed_text`).

    TAHAP 5: sebelumnya memakai `StratifiedKFold` per baris. Karena ada
    124 grup duplikat (424 baris), baris dengan teks identik bisa muncul di
    fold latih dan fold validasi sekaligus sehingga estimasi CV terlalu
    optimistis. `StratifiedGroupKFold` memastikan seluruh grup duplikat
    hanya berada di satu fold.

    Pemakaian:
        cv = make_grouped_cv()
        scores = cross_val_score(pipe, X, y, cv=cv, groups=groups)
    """
    from sklearn.model_selection import StratifiedGroupKFold

    return StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


# ============================================================
# BACKWARD-COMPATIBLE HOLDOUT ALIAS
# ============================================================

def make_final_holdout(
    df: pd.DataFrame,
    text_column: str | None = None,
    label_column: str | None = None,
):
    """
    Alias kompatibilitas untuk fixed 80:20 split.

    Gunakan make_fixed_split() pada eksperimen baru.
    """

    return make_fixed_split(
        df,
        text_column=text_column,
        label_column=label_column,
    )


# ============================================================
# JSON UTILITIES
# ============================================================

def save_json(
    data: dict,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)

    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


# ============================================================
# DISTRIBUTION
# ============================================================

def get_distribution(
    series: pd.Series,
) -> dict:
    counts = series.value_counts()
    total = len(series)

    if total == 0:
        return {
            label: {
                "count": 0,
                "percentage": 0.0,
            }
            for label in VALID_LABELS
        }

    return {
        label: {
            "count": int(counts.get(label, 0)),
            "percentage": round(
                counts.get(label, 0) / total * 100,
                4,
            ),
        }
        for label in VALID_LABELS
    }


# ============================================================
# SELF TEST
# ============================================================

def validate_common_utilities() -> None:
    """
    Pemeriksaan dasar terhadap helper common.py.
    Tidak menjalankan eksperimen model.
    """

    if RANDOM_STATE != 42:
        raise ValueError("RANDOM_STATE harus 42.")

    if CV_FOLDS != 5:
        raise ValueError("CV_FOLDS harus 5.")

    if abs(
        FINAL_TRAIN_RATIO + FINAL_TEST_RATIO - 1.0
    ) > 1e-12:
        raise ValueError(
            "Train ratio + test ratio harus sama dengan 1.0."
        )

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Canonical dataset tidak ditemukan:\n{DATASET_PATH}"
        )

    print("COMMON UTILITIES VALID")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset      : {DATASET_PATH}")
    print(f"Random state : {RANDOM_STATE}")
    print(f"CV folds     : {CV_FOLDS}")
    print(
        f"Split        : "
        f"{FINAL_TRAIN_RATIO:.0%}/{FINAL_TEST_RATIO:.0%}"
    )


if __name__ == "__main__":
    validate_common_utilities()
