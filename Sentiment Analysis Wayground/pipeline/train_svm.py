"""
pipeline/train_svm.py

Compatibility layer untuk backend/ml_engine.py.

CATATAN:
- Modul ini TIDAK melakukan training otomatis saat di-import.
- Konfigurasi final tetap berasal dari shared.final_config.
- Model final canonical adalah:
    models/final_v4/final_pipeline.joblib
- Pipeline final sudah mencakup:
    TF-IDF -> OneVsOneClassifier -> Linear SVM
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np

from shared.final_config import (
    PROJECT_ROOT,
    FINAL_PIPELINE_PATH,
    FINAL_MODEL_CONFIG,
)


# ============================================================
# PATH CANONICAL
# ============================================================

CANONICAL_PIPELINE_PATH = Path(FINAL_PIPELINE_PATH)

if not CANONICAL_PIPELINE_PATH.is_absolute():
    CANONICAL_PIPELINE_PATH = PROJECT_ROOT / CANONICAL_PIPELINE_PATH


# ============================================================
# FINAL PIPELINE LOADER
# ============================================================

def load_final_pipeline(path: Optional[str | Path] = None):
    """
    Memuat canonical final pipeline.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline final yang sudah berisi TF-IDF + classifier.
    """

    model_path = Path(path) if path else CANONICAL_PIPELINE_PATH

    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    if not model_path.exists():
        raise FileNotFoundError(
            f"Canonical final pipeline tidak ditemukan:\n{model_path}"
        )

    from shared.deployment_integrity import load_verified_pipeline
    pipeline, _ = load_verified_pipeline(model_path, model_path.parent / "metadata.json")
    return pipeline


# ============================================================
# PREDICTION
# ============================================================

def predict_with_final_pipeline(
    texts,
    pipeline=None,
):
    """
    Prediksi menggunakan canonical final pipeline.

    Parameters
    ----------
    texts : list[str] atau iterable[str]
        Teks yang akan diprediksi.

    pipeline : optional
        Pipeline sklearn. Jika None, pipeline final akan dimuat.

    Returns
    -------
    numpy.ndarray
        Label hasil prediksi.
    """

    if pipeline is None:
        pipeline = load_final_pipeline()

    if isinstance(texts, str):
        texts = [texts]

    texts = list(texts)

    if not texts:
        return np.array([], dtype=object)

    return pipeline.predict(texts)


def predict_proba_with_final_pipeline(
    texts,
    pipeline=None,
):
    """
    Menghasilkan probabilitas jika classifier mendukung predict_proba.

    Catatan:
    OneVsRestClassifier/SVC dapat tidak menyediakan predict_proba
    apabila probability=False pada base estimator.
    """

    if pipeline is None:
        pipeline = load_final_pipeline()

    if isinstance(texts, str):
        texts = [texts]

    texts = list(texts)

    if not texts:
        return np.empty((0, 0))

    if not hasattr(pipeline, "predict_proba"):
        raise AttributeError(
            "Canonical final pipeline tidak menyediakan predict_proba()."
        )

    return pipeline.predict_proba(texts)


# ============================================================
# METADATA
# ============================================================

def get_final_model_config() -> dict:
    """
    Mengembalikan konfigurasi final dari single source of truth.
    """
    return FINAL_MODEL_CONFIG.copy()


def get_final_pipeline_path() -> Path:
    """
    Mengembalikan lokasi canonical final pipeline.
    """
    return CANONICAL_PIPELINE_PATH


def verify_final_pipeline() -> dict:
    """
    Memeriksa struktur dasar canonical final pipeline.
    Tidak melakukan training.
    """

    from shared.deployment_integrity import load_verified_pipeline
    pipeline, metadata = load_verified_pipeline()

    result = {
        "path": str(CANONICAL_PIPELINE_PATH),
        "loaded": True,
        "pipeline_type": type(pipeline).__name__,
        "steps": [],
        "valid": False,
        "model_version": metadata["model_version"],
        "evaluation_provenance": metadata["evaluation_provenance"],
    }

    if hasattr(pipeline, "steps"):
        result["steps"] = [
            name for name, _ in pipeline.steps
        ]

    result["valid"] = hasattr(pipeline, "predict")

    return result


# ============================================================
# BACKWARD-COMPATIBILITY HELPERS
# ============================================================

def load_model(path=None):
    """
    Compatibility helper.

    Prior code mungkin memanggil load_model().
    Sekarang diarahkan ke canonical final pipeline.
    """

    return load_final_pipeline(path)


def load_pipeline(path=None):
    """
    Alias untuk load_final_pipeline().
    """

    return load_final_pipeline(path)


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "CANONICAL_PIPELINE_PATH",
    "load_final_pipeline",
    "load_pipeline",
    "load_model",
    "predict_with_final_pipeline",
    "predict_proba_with_final_pipeline",
    "get_final_model_config",
    "get_final_pipeline_path",
    "verify_final_pipeline",
]
