"""
FINAL CONFIGURATION - WAYGROUND SENTIMENT ANALYSIS

Single source of truth untuk konfigurasi final penelitian.

PENTING:
- File ini TIDAK membaca best_config.json saat import.
- File ini TIDAK melakukan pemilihan parameter.
- File ini menjadi sumber konfigurasi yang dibaca lock_final_config.py.
- lock_final_config.py kemudian membuat best_config.json sebagai snapshot.
- best_config.json BUKAN sumber konfigurasi untuk file ini.

Dengan desain ini, tidak ada circular dependency:
    final_config.py
        -> lock_final_config.py
        -> best_config.json
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


# ============================================================================
# PROJECT / DATASET
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Path keluaran dapat di-override lewat environment variable supaya
# hasil eksperimen baru tidak menimpa artefak lama (Tahap 5).
from shared.constants import (  # noqa: E402
    CANONICAL_DATASET_PATH,
    EXPERIMENTS_DIR,
    FINAL_MODEL_DIR,
)

# Path dataset canonical mengikuti PIPELINE_OUTPUT_DIR, sehingga dataset
# berversi (Tahap 5) dipakai tanpa mengubah kode.
DATASET_RELATIVE_PATH = CANONICAL_DATASET_PATH.relative_to(PROJECT_ROOT)

TEXT_COLUMN = "processed_text"
LABEL_COLUMN = "sentiment_label_final"


# ============================================================================
# REPRODUCIBILITY / SPLIT
# ============================================================================

RANDOM_STATE = 42
CV_FOLDS = 5

TRAIN_TEST_SPLIT_RATIO = 0.80
TEST_SIZE = 0.20
STRATIFY = True


# ============================================================================
# LABELING
# ============================================================================

LABELING_METHOD = "Lexicon v2 + ExpertValidation"

# TAHAP 5: label dasar sudah memakai lexicon v2 (shared/labeling.py),
# bukan InSetLexicon v1. Dipakai oleh lock_final_config.py agar snapshot
# konfigurasi tidak lagi menyebut "InSetLexicon" yang sudah usang.
LEXICON_BASE_METHOD = "Lexicon v2 (shared/labeling.py) + Expert validation"
from shared.research_config import DEPLOYMENT_NEUTRAL_THRESHOLD

# Threshold selected on preparation development only (Experiment 2).
NEUTRAL_THRESHOLD = DEPLOYMENT_NEUTRAL_THRESHOLD

SENTIMENT_CLASSES = [
    "Positif",
    "Netral",
    "Negatif",
]


# ============================================================================
# EVALUATION
# ============================================================================

EVAL_AVERAGE = "macro"
SELECTION_METRIC = "f1_macro"


# ============================================================================
# FINAL MODEL CONFIGURATION
# ============================================================================

# Parameter ini dikunci berdasarkan hasil eksperimen yang telah direview.
# Jangan mengubahnya otomatis dari best_config.json.

# Selected by the completed Stage 3 run; see output_v4/experiments/selected_configuration.json.
FINAL_MODEL_CONFIG: dict[str, Any] = {
    "tfidf": {"max_features": 5000, "ngram_range": (1, 3),
              "sublinear_tf": False, "min_df": 1, "max_df": 0.90},
    "svm": {"kernel": "linear", "C": 10.0, "gamma": "scale", "class_weight": None},
    "multiclass": {"strategy": "ovo"},
    "imbalance": {"method": "none"},
}


# ============================================================================
# ASPECT ANALYSIS
# ============================================================================

ASPECTS = [
    "Kemudahan Akses",
    "Gamifikasi",
    "Materi/Konten",
    "Performa Teknis",
]

ASPECT_DEFINITIONS = {
    "Kemudahan Akses": (
        "Penilaian terhadap kemudahan akses dan kecepatan pemuatan"
    ),
    "Gamifikasi": (
        "Penilaian terhadap unsur permainan sebagai pendorong motivasi belajar"
    ),
    "Materi/Konten": (
        "Penilaian terhadap kualitas dan kelengkapan konten pembelajaran"
    ),
    "Performa Teknis": (
        "Penilaian terhadap stabilitas dan keandalan aplikasi"
    ),
}


# ============================================================================
# ARTIFACT PATHS
# ============================================================================

OUTPUT_DIR = EXPERIMENTS_DIR

MODELS_DIR = FINAL_MODEL_DIR.parent
FINAL_DIR = FINAL_MODEL_DIR

BEST_CONFIG_PATH = OUTPUT_DIR / "best_config.json"
EXPERIMENT_RESULTS_PATH = OUTPUT_DIR / "experiment_results.json"

FINAL_PIPELINE_PATH = FINAL_DIR / "final_pipeline.joblib"

# TF-IDF sudah menjadi bagian dari final_pipeline.joblib.
MODEL_PATH = FINAL_PIPELINE_PATH
VECTORIZER_PATH = FINAL_PIPELINE_PATH

FINAL_METADATA_PATH = FINAL_DIR / "metadata.json"
FINAL_EVALUATION_PATH = OUTPUT_DIR / "final_model_evaluation.json"
FINAL_CLASSIFICATION_REPORT_PATH = (
    OUTPUT_DIR / "final_classification_report.json"
)
FINAL_CONFUSION_MATRIX_PATH = OUTPUT_DIR / "final_confusion_matrix.csv"
FINAL_PREDICTIONS_PATH = OUTPUT_DIR / "final_holdout_predictions.csv"


# ============================================================================
# CANONICAL DATASET CONTRACT
# ============================================================================

# Dataset final yang menjadi dasar locking.
# (Diperbarui oleh scripts/refresh_dataset_contract.py)
EXPECTED_DATASET_ROWS = 3078

# Distribusi kelas harus sama persis dengan dataset canonical yang digunakan
# untuk final model. Jika dataset aktual berbeda, locking HARUS berhenti.
EXPECTED_CLASS_DISTRIBUTION = {
    "negatif": 575,
    "netral": 556,
    "positif": 1947,
}

EXPECTED_DEVELOPMENT_ROWS = 2474
EXPECTED_HOLDOUT_ROWS = 604

# Berdasarkan validasi pakar yang tercatat pada konfigurasi final.
EXPECTED_EXPERT_VALIDATED_ROWS = 288


# ============================================================================
# PUBLIC API
# ============================================================================

def get_final_config() -> dict[str, Any]:
    """Mengembalikan konfigurasi final tanpa membaca best_config.json."""

    return {
        "dataset": {
            "path": str(PROJECT_ROOT / DATASET_RELATIVE_PATH),
            "text_column": TEXT_COLUMN,
            "label_column": LABEL_COLUMN,
        },
        "split": {
            "train_size": TRAIN_TEST_SPLIT_RATIO,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratify": STRATIFY,
            "cv_folds": CV_FOLDS,
        },
        "labeling": {
            "method": LABELING_METHOD,
            "neutral_threshold": NEUTRAL_THRESHOLD,
            "classes": list(SENTIMENT_CLASSES),
            "initial_label_column": "sentiment_label_inset",
            "expert_label_column": "sentiment_label_expert",
            "final_label_column": LABEL_COLUMN,
        },
        "model": deepcopy(FINAL_MODEL_CONFIG),
        "evaluation": {
            "average": EVAL_AVERAGE,
            "selection_metric": SELECTION_METRIC,
            "selection_rule": (
                "Highest CV Macro F1, then lowest CV standard deviation, "
                "then simpler model"
            ),
            "holdout_test_used_for_selection": False,
        },
        "artifacts": {
            "best_config": str(BEST_CONFIG_PATH),
            "experiment_results": str(EXPERIMENT_RESULTS_PATH),
            "pipeline": str(FINAL_PIPELINE_PATH),
            "model": str(MODEL_PATH),
            "vectorizer": str(VECTORIZER_PATH),
            "metadata": str(FINAL_METADATA_PATH),
            "evaluation": str(FINAL_EVALUATION_PATH),
            "classification_report": str(FINAL_CLASSIFICATION_REPORT_PATH),
            "confusion_matrix": str(FINAL_CONFUSION_MATRIX_PATH),
            "predictions": str(FINAL_PREDICTIONS_PATH),
        },
    }


# Backward-compatible helper.
# Penting: helper ini HANYA mengembalikan konfigurasi statis.
# Ia tidak membaca best_config.json.
def load_best_config(path: Path | None = None) -> dict[str, Any]:
    """
    Kompatibilitas untuk script lama.

    Jika path diberikan dan file tersedia, file dibaca sebagai snapshot.
    Jika tidak diberikan, fungsi mengembalikan konfigurasi final statis.

    Fungsi ini sengaja tidak dipanggil pada import module.
    """
    if path is None:
        return get_final_config()

    import json

    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(
            f"Snapshot konfigurasi tidak ditemukan: {target}"
        )

    with target.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Format JSON tidak valid: {target}")

    return data


def verify_best_config_agrees(
    path: Path | None = None,
) -> dict[str, Any]:
    """
    Verifikasi snapshot terhadap FINAL_MODEL_CONFIG.

    Fungsi ini hanya dipanggil setelah best_config.json dibuat.
    """
    import json

    target = Path(path) if path else BEST_CONFIG_PATH

    if not target.exists():
        raise FileNotFoundError(
            f"Snapshot konfigurasi belum tersedia: {target}"
        )

    with target.open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Format JSON tidak valid: {target}")

    source = FINAL_MODEL_CONFIG
    model = config.get("model")

    if not isinstance(model, dict):
        raise ValueError(
            "best_config.json tidak memiliki bagian 'model' yang valid."
        )

    problems: list[str] = []

    # JSON mengubah tuple menjadi list, sehingga perbandingan langsung
    # dapat menghasilkan false mismatch walaupun parameternya sama.
    snapshot_tfidf = dict(model.get("tfidf") or {})
    source_tfidf = dict(source["tfidf"])
    if "ngram_range" in snapshot_tfidf:
        snapshot_tfidf["ngram_range"] = tuple(snapshot_tfidf["ngram_range"])
    if "ngram_range" in source_tfidf:
        source_tfidf["ngram_range"] = tuple(source_tfidf["ngram_range"])
    if snapshot_tfidf != source_tfidf:
        problems.append("model.tfidf berbeda")

    if dict(model.get("svm") or {}) != dict(source["svm"]):
        problems.append("model.svm berbeda")

    # best_config.json boleh menyimpan metadata tambahan seperti
    # "implementation": "native_svc_ovo". Yang dibandingkan adalah
    # parameter strategi yang benar-benar menjadi source of truth.
    snapshot_multiclass = dict(model.get("multiclass") or {})
    source_multiclass = dict(source["multiclass"])
    if snapshot_multiclass.get("strategy") != source_multiclass.get("strategy"):
        problems.append("model.multiclass berbeda")

    if dict(model.get("imbalance") or {}) != dict(source["imbalance"]):
        problems.append("model.imbalance berbeda")

    if config.get("labeling", {}).get("neutral_threshold") != NEUTRAL_THRESHOLD:
        problems.append("labeling.neutral_threshold berbeda")

    if problems:
        raise ValueError(
            "best_config.json tidak konsisten dengan final_config.py:\n"
            + "\n".join(f"- {item}" for item in problems)
        )

    return config


def validate_config(require_dataset: bool = True) -> None:
    """
    Memvalidasi konfigurasi final tanpa melakukan pemilihan parameter.

    Kompatibel dengan train_final_model.py:
    - require_dataset=False: hanya validasi struktur/nilai konfigurasi.
    - require_dataset=True: sekaligus memvalidasi keberadaan dataset dan
      kontrak ukuran/distribusi dataset.
    """
    required_tfidf = {
        "max_features", "ngram_range", "sublinear_tf", "min_df", "max_df"
    }
    required_svm = {"kernel", "C", "gamma", "class_weight"}

    tfidf = FINAL_MODEL_CONFIG.get("tfidf")
    svm = FINAL_MODEL_CONFIG.get("svm")
    multiclass = FINAL_MODEL_CONFIG.get("multiclass")
    imbalance = FINAL_MODEL_CONFIG.get("imbalance")

    if not isinstance(tfidf, dict) or not required_tfidf.issubset(tfidf):
        raise ValueError(
            "FINAL_MODEL_CONFIG['tfidf'] tidak lengkap. "
            f"Wajib memiliki: {sorted(required_tfidf)}"
        )

    if not isinstance(svm, dict) or not required_svm.issubset(svm):
        raise ValueError(
            "FINAL_MODEL_CONFIG['svm'] tidak lengkap. "
            f"Wajib memiliki: {sorted(required_svm)}"
        )

    if not isinstance(multiclass, dict):
        raise ValueError("FINAL_MODEL_CONFIG['multiclass'] harus berupa dictionary.")

    if multiclass.get("strategy") not in {"ovo", "ovr"}:
        raise ValueError(
            "Strategi multiclass harus 'ovo' atau 'ovr', "
            f"ditemukan: {multiclass.get('strategy')!r}"
        )

    if not isinstance(imbalance, dict) or imbalance.get("method") not in {
        "none",
        "class_weight_balanced",
        "random_over",
        "random_under",
        "smote",
    }:
        raise ValueError(
            "Metode imbalance tidak valid: "
            f"{imbalance.get('method') if isinstance(imbalance, dict) else None!r}"
        )

    if int(tfidf["max_features"]) <= 0:
        raise ValueError("TF-IDF max_features harus > 0.")

    ngram = tuple(tfidf["ngram_range"])
    if len(ngram) != 2 or ngram[0] < 1 or ngram[0] > ngram[1]:
        raise ValueError(f"ngram_range tidak valid: {ngram!r}")

    if int(tfidf["min_df"]) < 1:
        raise ValueError("TF-IDF min_df harus >= 1.")

    if not 0 < float(tfidf["max_df"]) <= 1:
        raise ValueError("TF-IDF max_df harus berada pada rentang (0, 1].")

    if float(svm["C"]) <= 0:
        raise ValueError("SVM C harus > 0.")

    if TRAIN_TEST_SPLIT_RATIO <= 0 or TRAIN_TEST_SPLIT_RATIO >= 1:
        raise ValueError("TRAIN_TEST_SPLIT_RATIO harus berada di antara 0 dan 1.")

    if abs((TRAIN_TEST_SPLIT_RATIO + TEST_SIZE) - 1.0) > 1e-9:
        raise ValueError(
            "TRAIN_TEST_SPLIT_RATIO + TEST_SIZE harus sama dengan 1.0."
        )

    if CV_FOLDS < 2:
        raise ValueError("CV_FOLDS harus >= 2.")

    if require_dataset:
        dataset_path = PROJECT_ROOT / DATASET_RELATIVE_PATH
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset canonical tidak ditemukan: {dataset_path}"
            )

        from shared.dataset_io import read_canonical_dataset

        df = read_canonical_dataset(dataset_path, TEXT_COLUMN, LABEL_COLUMN)
        if len(df) != EXPECTED_DATASET_ROWS:
            raise ValueError(
                "Jumlah baris dataset tidak sesuai konfigurasi final: "
                f"expected={EXPECTED_DATASET_ROWS}, actual={len(df)}"
            )

        if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
            raise ValueError(
                f"Dataset wajib memiliki kolom {TEXT_COLUMN!r} dan {LABEL_COLUMN!r}."
            )

        labels = df[LABEL_COLUMN].astype("string").str.strip().str.lower()
        if not labels.isin(["negatif", "netral", "positif"]).all():
            raise ValueError("Dataset memiliki label kosong atau tidak dikenal.")
        if df[TEXT_COLUMN].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError("Dataset memiliki teks hasil preprocessing kosong.")

        actual_distribution = {
            label: int((labels == label).sum())
            for label in ("negatif", "netral", "positif")
        }

        if actual_distribution != EXPECTED_CLASS_DISTRIBUTION:
            raise ValueError(
                "Distribusi kelas dataset tidak sesuai konfigurasi final.\n"
                f"Expected: {EXPECTED_CLASS_DISTRIBUTION}\n"
                f"Actual:   {actual_distribution}"
            )



if __name__ == "__main__":
    print("=" * 80)
    print("WAYGROUND - FINAL CONFIGURATION")
    print("=" * 80)
    print("[OK] final_config.py dapat dijalankan tanpa best_config.json.")
    print("[INFO] File ini adalah single source of truth.")
    print()
    print("TF-IDF     :", FINAL_MODEL_CONFIG["tfidf"])
    print("SVM        :", FINAL_MODEL_CONFIG["svm"])
    print("Multiclass :", FINAL_MODEL_CONFIG["multiclass"])
    print("Imbalance  :", FINAL_MODEL_CONFIG["imbalance"])
    print("Split      :", f"{TRAIN_TEST_SPLIT_RATIO}:{TEST_SIZE}")
    print("Random     :", RANDOM_STATE)
    print("CV folds   :", CV_FOLDS)
