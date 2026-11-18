"""
Konfigurasi Backend
====================
Konfigurasi Flask/backend-specific berada di sini.

Konfigurasi ML final TIDAK didefinisikan ulang di backend.
Backend mengambil seluruh konfigurasi penelitian dari:

    shared.final_config.FINAL_MODEL_CONFIG

Dengan demikian shared/final_config.py menjadi single source of truth.
"""

import os
from pathlib import Path

# ============================================================================
# BASE PATHS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset"

# Database dan folder upload dapat di-override lewat environment variable.
#
# TUJUAN: seluruh test suite WAJIB berjalan terhadap database test, tidak
# pernah terhadap database produksi. Override ini dibaca SATU KALI saat
# modul di-import, jadi environment variable harus di-set SEBELUM
# `backend.config` di-import (lihat tests/conftest.py).
#
#   set WAYGROUND_DATABASE_URI=sqlite:///D:/path/ke/test.db
#   set WAYGROUND_DATABASE_PATH=D:/path/ke/test.db
#   set WAYGROUND_UPLOAD_DIR=D:/path/ke/test_uploads
DATABASE_PATH = Path(
    os.getenv(
        "WAYGROUND_DATABASE_PATH",
        str(DATA_DIR / "sentiment_analysis.db"),
    )
)
DATABASE_URI = (
    os.getenv("WAYGROUND_DATABASE_URI")
    or f"sqlite:///{DATABASE_PATH}"
)
UPLOAD_DIR = Path(
    os.getenv("WAYGROUND_UPLOAD_DIR", str(DATA_DIR / "uploads"))
)

# Buat direktori yang diperlukan backend
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Direktori database harus sudah ada agar SQLite dapat membuat file.
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

# ============================================================================
# SERVER
# ============================================================================

HOST = os.getenv("FLASK_HOST", "0.0.0.0")
PORT = int(os.getenv("FLASK_PORT", "5001"))
DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"

# ============================================================================
# FILE UPLOAD
# ============================================================================
# File uji hanya boleh berformat CSV/TSV (metodologi_final.md Bab 15).
# Daftar lengkap ada di backend/services/upload_service.py; nilai di sini
# hanya keptasan untuk kompatibilitas.
ALLOWED_EXTENSIONS = {"csv", "tsv", "txt"}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB

# ============================================================================
# ML CONFIGURATION
# ============================================================================
# Single source of truth:
# shared/final_config.py
#
# Jangan mendefinisikan ulang parameter final model di backend.
# ============================================================================

try:
    from shared.final_config import (
        PROJECT_ROOT,
        DATASET_RELATIVE_PATH,
        TEXT_COLUMN,
        LABEL_COLUMN,
        RANDOM_STATE,
        CV_FOLDS,
        TRAIN_TEST_SPLIT_RATIO,
        TEST_SIZE,
        STRATIFY,
        LABELING_METHOD,
        NEUTRAL_THRESHOLD,
        SENTIMENT_CLASSES,
        EVAL_AVERAGE,
        SELECTION_METRIC,
        FINAL_MODEL_CONFIG,
        ASPECTS,
        ASPECT_DEFINITIONS,
        FINAL_PIPELINE_PATH,
        MODEL_PATH,
        VECTORIZER_PATH,
    )

    # ------------------------------------------------------------------------
    # Final model configuration
    # ------------------------------------------------------------------------

    TFIDF_CONFIG = FINAL_MODEL_CONFIG["tfidf"]
    SVM_CONFIG = FINAL_MODEL_CONFIG["svm"]
    MULTICLASS_CONFIG = FINAL_MODEL_CONFIG["multiclass"]
    IMBALANCE_CONFIG = FINAL_MODEL_CONFIG["imbalance"]

    # ------------------------------------------------------------------------
    # Backward-compatible ML aliases
    #
    # Alias ini hanya untuk kode backend lama yang masih membaca nama
    # parameter individual. Nilainya tetap berasal dari FINAL_MODEL_CONFIG.
    # ------------------------------------------------------------------------

    TFIDF_MAX_FEATURES = TFIDF_CONFIG["max_features"]
    TFIDF_NGRAM_RANGE = TFIDF_CONFIG["ngram_range"]
    TFIDF_SUBLINEAR_TF = TFIDF_CONFIG["sublinear_tf"]
    TFIDF_MIN_DF = TFIDF_CONFIG["min_df"]
    TFIDF_MAX_DF = TFIDF_CONFIG["max_df"]

    SVM_KERNEL = SVM_CONFIG["kernel"]
    SVM_C = SVM_CONFIG["C"]
    SVM_GAMMA = SVM_CONFIG["gamma"]
    SVM_CLASS_WEIGHT = SVM_CONFIG["class_weight"]

    # Konfigurasi kernel yang tersedia untuk compatibility dengan backend lama.
    # Ini bukan konfigurasi pemilihan model final.
    SVM_KERNELS = ["linear", "rbf", "poly", "sigmoid"]

    # OVO adalah strategi multiclass final.
    MULTICLASS_STRATEGY = MULTICLASS_CONFIG["strategy"]

    # Final imbalance handling.
    IMBALANCE_METHOD = IMBALANCE_CONFIG["method"]

    # Legacy aliases
    TFIDF_MAX_FEATURES_ALIAS = TFIDF_MAX_FEATURES
    TFIDF_NGRAM_RANGE_ALIAS = TFIDF_NGRAM_RANGE
    TFIDF_SUBLINEAR_TF_ALIAS = TFIDF_SUBLINEAR_TF
    SVM_KERNELS_ALIAS = SVM_KERNELS
    RANDOM_STATE_ALIAS = RANDOM_STATE

    # Hanya untuk compatibility dengan kode lama.
    # Bukan konfigurasi final eksperimen.
    GRID_SEARCH_C = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

except ImportError as e:
    raise RuntimeError(
        "shared/final_config.py tidak dapat diimpor: "
        + str(e)
        + ". Pastikan project berjalan dari root directory."
    ) from e

# ============================================================================
# CORS
# ============================================================================

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

# ============================================================================
# API
# ============================================================================

API_PREFIX = "/api/v1"
API_VERSION = "1.0.0"
RATELIMIT_DEFAULT = "200 per day"
RATELIMIT_STORAGE_URL = "memory://"