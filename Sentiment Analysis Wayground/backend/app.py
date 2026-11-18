"""
Flask Backend Application
=========================
REST API untuk Sentiment Analysis Wayground.

Arsitektur final:
- shared/final_config.py = single source of truth konfigurasi penelitian
- shared/preprocessing.py = preprocessing canonical
- models/final_v4/final_pipeline.joblib = satu-satunya artifact inference final
- data/pipeline_output_v4/stage3_final_labeled.csv = dataset canonical

Catatan penting:
Endpoint training dipertahankan untuk kompatibilitas API, tetapi sengaja
DINONAKTIFKAN agar model final yang sudah dikunci tidak dapat tertimpa dari API.
"""

import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

# -----------------------------------------------------------------------------
# PATH / IMPORTS
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend import config
from backend.database_models import (
    db,
    AnalysisSession,
    ReviewData,
    AspectResult,
    EvaluationResult,
    init_db, ensure_analysis_schema
)
from backend.ml_engine import (
    AspectDetectionEngine,
    SentimentLabelingEngine,
)
from pipeline.train_svm import load_final_pipeline, get_final_pipeline_path
from shared.preprocessing import preprocess_text as shared_preprocess_text
from shared.deployment_integrity import load_verified_pipeline, DeploymentIntegrityError
from shared.labels import normalize_label as shared_normalize_label
from shared.final_config import (
    DATASET_RELATIVE_PATH,
    FINAL_CLASSIFICATION_REPORT_PATH as SHARED_FINAL_REPORT_PATH,
    FINAL_CONFUSION_MATRIX_PATH as SHARED_FINAL_CM_PATH,
    FINAL_EVALUATION_PATH as SHARED_FINAL_EVALUATION_PATH,
    FINAL_PIPELINE_PATH as SHARED_FINAL_PIPELINE_PATH,
    LABEL_COLUMN as CANONICAL_LABEL_COLUMN,
    TEXT_COLUMN as CANONICAL_TEXT_COLUMN,
)


# -----------------------------------------------------------------------------
# CANONICAL PATHS / CONSTANTS
# -----------------------------------------------------------------------------
# Paths come from shared/final_config.py so the backend can never point at a
# different artifact than the one produced by scripts/train_final_model.py.

FINAL_DATASET_PATH = BASE_DIR / DATASET_RELATIVE_PATH
FINAL_EVALUATION_PATH = SHARED_FINAL_EVALUATION_PATH
FINAL_REPORT_PATH = SHARED_FINAL_REPORT_PATH
FINAL_CM_PATH = SHARED_FINAL_CM_PATH
FINAL_PIPELINE_PATH = SHARED_FINAL_PIPELINE_PATH

CANONICAL_LABELS = ["positif", "netral", "negatif"]


# -----------------------------------------------------------------------------
# APP INITIALIZATION
# -----------------------------------------------------------------------------


def create_app():
    """Create and configure Flask application."""
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = config.DATABASE_URI
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

    CORS(app, origins=config.CORS_ORIGINS)
    db.init_app(app)

    with app.app_context():
        ensure_analysis_schema()

    # Blueprint analisis inti (alur upload -> preprocess -> predict ->
    # evaluate -> aspect). Wajib terdaftar; jika gagal, backend tidak
    # berguna untuk pengujian model.
    try:
        from backend.api.analysis import analysis_bp
        app.register_blueprint(analysis_bp)
        from backend.api.web_flow import web_flow_bp
        app.register_blueprint(web_flow_bp)
    except Exception as exc:
        app.logger.exception("Analysis API tidak dapat didaftarkan: %s", exc)
        raise

    return app


app = create_app()


@app.errorhandler(DeploymentIntegrityError)
def deployment_unavailable(error):
    return jsonify({"success": False, "error": str(error), "model_available": False}), 503


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------


def allowed_file(filename):
    """Delegasi ke validator file uji yang sama dengan blueprint analysis."""
    from backend.services.upload_service import allowed_file as _allowed_file

    return _allowed_file(filename)


def save_uploaded_file(file):
    from backend.services.upload_service import allowed_file as _allowed_file

    if not file or not _allowed_file(file.filename):
        return None
    filename = secure_filename(file.filename) or "upload.csv"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = config.UPLOAD_DIR / f"{timestamp}_{filename}"
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file.save(filepath)
    return filepath


def _json_load(path, default=None):
    if default is None:
        default = {}
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception as exc:
        app.logger.warning("Gagal membaca %s: %s", path, exc)
    return default


def _load_canonical_dataset():
    """Load the locked canonical labeled dataset used by the final model."""
    if not FINAL_DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset canonical tidak ditemukan: {FINAL_DATASET_PATH}")
    df = pd.read_csv(FINAL_DATASET_PATH)
    required = {CANONICAL_TEXT_COLUMN, CANONICAL_LABEL_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Kolom canonical tidak lengkap: {sorted(missing)}")
    return df


# Compatibility alias for older internal calls.
def _load_final_dataset():
    return _load_canonical_dataset()


def _normalize_label(label):
    """
    Normalisasi label ke kelas canonical (lowercase).

    Delegasi ke shared/labels.py sebagai SATU-SATUNYA definisi normalisasi
    label, supaya backend, script, dan laporan selalu memakai aturan yang
    sama.
    """
    return shared_normalize_label(label)


def _label_distribution_from_dataset(df):
    counts = {label: 0 for label in CANONICAL_LABELS}
    if CANONICAL_LABEL_COLUMN in df.columns:
        raw = df[CANONICAL_LABEL_COLUMN].dropna().map(_normalize_label)
        for label, count in raw.value_counts().items():
            if label in counts:
                counts[label] = int(count)
    return counts


def _load_final_metrics():
    """
    Load the final holdout metrics from the verified evaluation artifacts.

    The evaluation JSON produced by scripts/train_final_model.py stores the
    holdout metrics under "metrics" (holdout_*) and, for backwards
    compatibility, under "holdout_metrics". The classification report stores
    sklearn's report layout ("macro avg" / "weighted avg").

    Every metric returned here comes from an artifact that was written by the
    same training run that produced the deployed pipeline. No number is
    hard-coded in this module.
    """
    data = _json_load(FINAL_EVALUATION_PATH, {})
    report = _json_load(FINAL_REPORT_PATH, {})

    metrics = data.get("metrics", {}) if isinstance(data, dict) else {}
    holdout = data.get("holdout_metrics", {}) if isinstance(data, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}
    if not isinstance(holdout, dict):
        holdout = {}
    if not isinstance(report, dict):
        report = {}

    def from_sources(*keys):
        for source in (metrics, holdout):
            for key in keys:
                value = source.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
        return None

    def from_report(macro_key, weighted_key=None):
        macro = report.get("macro avg")
        if isinstance(macro, dict):
            value = macro.get(macro_key)
            if isinstance(value, (int, float)):
                return float(value)
        if weighted_key:
            weighted = report.get("weighted avg")
            if isinstance(weighted, dict):
                value = weighted.get(weighted_key)
                if isinstance(value, (int, float)):
                    return float(value)
        return None

    accuracy = from_sources("holdout_accuracy", "accuracy")
    if accuracy is None:
        value = report.get("accuracy")
        accuracy = float(value) if isinstance(value, (int, float)) else 0.0

    precision_macro = from_sources("holdout_precision_macro", "precision_macro")
    if precision_macro is None:
        precision_macro = from_report("precision")
    if precision_macro is None:
        precision_macro = 0.0

    recall_macro = from_sources("holdout_recall_macro", "recall_macro")
    if recall_macro is None:
        recall_macro = from_report("recall")
    if recall_macro is None:
        recall_macro = 0.0

    f1_macro = from_sources("holdout_f1_macro", "f1_macro")
    if f1_macro is None:
        f1_macro = from_report("f1-score")
    if f1_macro is None:
        f1_macro = 0.0

    f1_weighted = from_sources("holdout_f1_weighted", "f1_weighted")
    if f1_weighted is None:
        f1_weighted = from_report("f1-score", weighted_key="f1-score")
    if f1_weighted is None:
        f1_weighted = 0.0

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "average": "macro",
        "raw": data,
        "report": report,
    }


def _load_confusion_matrix():
    if not FINAL_CM_PATH.exists():
        return None
    try:
        df = pd.read_csv(FINAL_CM_PATH, index_col=0)
        return df.values.tolist()
    except Exception as exc:
        app.logger.warning("Gagal membaca confusion matrix: %s", exc)
        return None


def _load_canonical_pipeline():
    """Always load the locked canonical pipeline, never legacy .pkl pairs."""
    if not FINAL_PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Canonical model tidak ditemukan: {FINAL_PIPELINE_PATH}")
    return load_final_pipeline()


def _predict_texts(texts):
    """Canonical inference: raw text -> shared preprocessing -> final pipeline."""
    if not isinstance(texts, list):
        texts = list(texts)

    cleaned = []
    processed = []
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Setiap teks harus berupa string yang tidak kosong")
        cleaned.append(text)
        processed.append(shared_preprocess_text(text))
        if not processed[-1].strip():
            raise ValueError("Teks kosong setelah preprocessing")

    pipeline = _load_canonical_pipeline()
    predictions = pipeline.predict(processed)

    results = []
    for original, processed_text, prediction in zip(cleaned, processed, predictions):
        label = _normalize_label(prediction)
        results.append(
            {
                "text": original,
                "processed_text": processed_text,
                "predicted_label": label,
                "sentiment_label": label,
                "model_version": pipeline._deployment_model_version,
            }
        )
    return results


def _get_model_config_payload():
    final = config.FINAL_MODEL_CONFIG
    return {
        "tfidf": {
            "max_features": final["tfidf"]["max_features"],
            "ngram_range": list(final["tfidf"]["ngram_range"]),
            "sublinear_tf": final["tfidf"]["sublinear_tf"],
            "min_df": final["tfidf"]["min_df"],
            "max_df": final["tfidf"]["max_df"],
        },
        "svm": {
            "kernel": final["svm"]["kernel"],
            "C": final["svm"]["C"],
            "gamma": final["svm"].get("gamma", "scale"),
            "class_weight": final["svm"].get("class_weight"),
        },
        "multiclass": final["multiclass"],
        "imbalance": final["imbalance"],
    }


def _safe_json(value):
    try:
        return json.loads(value) if isinstance(value, str) else value
    except Exception:
        return value


# -----------------------------------------------------------------------------
# BASIC ROUTES
# -----------------------------------------------------------------------------


@app.route("/")
def index():
    return jsonify(
        {
            "status": "ok",
            "service": "Sentiment Analysis Wayground API",
            "version": config.API_VERSION,
            "model": "final_pipeline.joblib",
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/api/v1/health")
def health():
    model_exists = FINAL_PIPELINE_PATH.exists()
    model_loadable = False
    identity = None
    if model_exists:
        try:
            _, metadata = load_verified_pipeline()
            identity = {"model_version": metadata["model_version"],
                        "integrity_verified": True,
                        "evaluation_provenance": metadata["evaluation_provenance"]}
            model_loadable = True
        except Exception as exc:
            app.logger.error("Canonical model gagal dimuat: %s", exc)

    return jsonify(
        {
            "status": "healthy" if model_loadable else "degraded",
            "database": "connected",
            "model_loaded": model_loadable,
            "identity": identity,
            "model_path": str(FINAL_PIPELINE_PATH),
            "canonical_model": "final_pipeline.joblib",
            "uploads_dir": str(config.UPLOAD_DIR),
        }
    ), (200 if model_loadable else 503)


# -----------------------------------------------------------------------------
# STATS / DISTRIBUTION / REVIEWS
# -----------------------------------------------------------------------------


@app.route("/api/v1/model/stats")
def get_model_stats():
    """
    Model-level statistics: verified holdout metrics of the final model.

    NOTE: /api/v1/stats belongs to the analysis blueprint and reports the
    test-history statistics used by the dashboard. The model metrics live
    here so the two payloads never collide.
    """
    try:
        _, deployment_metadata = load_verified_pipeline()
    except (ValueError, OSError, KeyError) as exc:
        return jsonify({"error": str(exc), "success": False}), 503
    metrics = _load_final_metrics()
    evaluation = metrics["raw"] if isinstance(metrics["raw"], dict) else {}
    split_info = evaluation.get("dataset", {}) if isinstance(evaluation, dict) else {}
    distribution = {label: 0 for label in CANONICAL_LABELS}
    total = 0
    preprocessed = 0
    raw_samples = 0
    after_dedup = 0
    aspects = {
        aspect: {"positif": 0, "netral": 0, "negatif": 0, "total": 0}
        for aspect in config.ASPECTS
    }

    # Canonical dataset is always available as a stable fallback/source.
    try:
        df = _load_canonical_dataset()
        distribution = _label_distribution_from_dataset(df)
        total = len(df)
        preprocessed = int(df[CANONICAL_TEXT_COLUMN].notna().sum()) if CANONICAL_TEXT_COLUMN in df.columns else 0
        after_dedup = total
    except Exception as exc:
        app.logger.warning("Canonical dataset gagal dibaca: %s", exc)

    # Model statistics describe the canonical research dataset. Upload history
    # remains under /stats and must not overwrite this denominator/distribution.
    manifest_path = FINAL_DATASET_PATH.parent / "dataset_manifest.json"
    dataset_manifest = _json_load(manifest_path, {})
    raw_samples = dataset_manifest.get("rows", {}).get("source", total)
    after_dedup = dataset_manifest.get("rows", {}).get("in_period_unique_content", total)

    # Research prevalence counts distinct reviews per aspect. Sentiment in this
    # report comes from inference on matched sentences, not inherited labels.
    aspect_report = _json_load(FINAL_EVALUATION_PATH.with_name("aspect_analysis.json"))
    aspect_counts = None
    aspect_sentiment = None
    if (aspect_report.get("model", {}).get("model_version") == deployment_metadata["model_version"]
            and aspect_report.get("dataset", {}).get("path") == split_info.get("path")
            and aspect_report.get("dataset", {}).get("rows") == total):
        try:
            entries = aspect_report["aspects"]
            aspect_counts = {name: entries[name]["reviews_with_aspect"] for name in config.ASPECTS}
            aspect_sentiment = {
                name: {label: entries[name]["model_sentiment_distribution"][label.capitalize()]["count"]
                       for label in CANONICAL_LABELS}
                for name in config.ASPECTS
            }
        except (KeyError, TypeError):
            aspect_counts = aspect_sentiment = None

    stats = {
        "aspect_statistics_available": aspect_counts is not None,
        "aspect_counts": aspect_counts,
        "aspect_sentiment": aspect_sentiment,
        "total_reviews": int(total),
        "processed": int(preprocessed),
        "positive": distribution["positif"],
        "neutral": distribution["netral"],
        "negative": distribution["negatif"],
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_weighted": metrics["f1_weighted"],
        "metric_average": metrics["average"],
        "metric_scope": "diagnostic_holdout_with_prior_exposure",
        "evaluation_provenance": deployment_metadata["evaluation_provenance"],
        "integrity_verified": True,
        "confusion_matrix_labels": ["negatif", "netral", "positif"],
        "model_loaded": FINAL_PIPELINE_PATH.exists(),
        "model_version": metrics["raw"].get("model_version") if isinstance(metrics["raw"], dict) else "final_pipeline",
        "confusion_matrix": _load_confusion_matrix(),
        "per_class": _safe_json(
            {
                label: metrics["report"][label]
                for label in CANONICAL_LABELS
                if isinstance(metrics.get("report"), dict) and label in metrics["report"]
            }
        ),
        "train_samples": split_info.get("development_samples", 0),
        "test_samples": split_info.get("holdout_samples", 0),
        "raw_samples": raw_samples,
        "after_dedup": after_dedup,
        "labeled_count": sum(distribution.values()),
        "preprocessed_count": preprocessed,
        "aspects": aspects,
        "split_ratio": "80:20",
    }

    return jsonify({"success": True, "stats": stats})


@app.route("/api/v1/distribution")
def get_distribution():
    """Return sentiment distribution from the canonical labeled dataset."""
    df = _load_canonical_dataset()
    counts = _label_distribution_from_dataset(df)
    total = sum(counts.values())
    distribution = {
        label: {
            "count": count,
            "percentage": round(100 * count / total, 2) if total else 0,
        }
        for label, count in counts.items()
    }
    return jsonify({"success": True, "distribution": distribution, "total": total})


@app.route("/api/v1/reviews")
def get_reviews():
    page = max(request.args.get("page", 1, type=int), 1)
    limit = min(max(request.args.get("limit", 50, type=int), 1), 500)
    sentiment = request.args.get("sentiment")

    df = _load_canonical_dataset()
    if sentiment:
        normalized = _normalize_label(sentiment)
        df = df[df[CANONICAL_LABEL_COLUMN].map(_normalize_label) == normalized]

    total = len(df)
    start = (page - 1) * limit
    page_df = df.iloc[start : start + limit].copy()

    # Frontend-compatible aliases while retaining canonical fields.
    if CANONICAL_TEXT_COLUMN in page_df.columns:
        page_df["processed_text"] = page_df[CANONICAL_TEXT_COLUMN]
    if CANONICAL_LABEL_COLUMN in page_df.columns:
        page_df["sentiment_label"] = page_df[CANONICAL_LABEL_COLUMN].map(_normalize_label)

    records = page_df.where(pd.notna(page_df), None).to_dict("records")
    return jsonify(
        {
            "success": True,
            "reviews": records,
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": (total + limit - 1) // limit if limit else 0,
        }
    )


# -----------------------------------------------------------------------------
# UPLOAD
# -----------------------------------------------------------------------------


@app.route("/api/v1/upload", methods=["POST"])
def upload_dataset():
    """
    Validasi file uji (CSV/TSV) tanpa menjalankan analisis.

    Endpoint ini memakai validasi yang sama dengan
    ``POST /api/v1/analysis/upload`` (backend/services/upload_service.py)
    sehingga file tidak bisa lolos validasi di satu endpoint tetapi gagal
    di endpoint lain.
    """
    from backend.services.upload_service import (
        UploadValidationError,
        allowed_file,
        find_content_column,
        read_uploaded_dataframe,
        validate_upload,
    )

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": (
                "Format file tidak didukung. Gunakan file .csv atau .tsv "
                "yang berisi kolom ulasan."
            )
        }), 400

    filepath = save_uploaded_file(file)
    if filepath is None:
        return jsonify({"error": "Failed to save file"}), 500

    try:
        try:
            df = read_uploaded_dataframe(filepath)
        except Exception as exc:
            return jsonify({
                "error": f"File tidak dapat dibaca: {exc}. "
                         "Pastikan file CSV/TSV yang valid dan tidak rusak."
            }), 400

        try:
            content_col, label_col, has_labels = validate_upload(df)
        except UploadValidationError as exc:
            payload = {"error": exc.message}
            payload.update(exc.extra)
            return jsonify(payload), 400

        return jsonify(
            {
                "success": True,
                "dataset": {
                    "filename": file.filename,
                    "filepath": str(filepath),
                    "rows": len(df),
                    "columns": list(df.columns),
                    "column_count": len(df.columns),
                    "detected_content_column": content_col,
                    "detected_label_column": label_col,
                    "has_actual_labels": has_labels,
                    "sample_content": df[content_col].iloc[0] if len(df) else None,
                },
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# -----------------------------------------------------------------------------
# PREPROCESSING / LABELING / ASPECTS
# -----------------------------------------------------------------------------


@app.route("/api/v1/preprocess", methods=["POST"])
def preprocess_data():
    data = request.get_json(silent=True) or {}
    texts = data.get("texts", [])
    if not texts and "text" in data:
        texts = [data["text"]]
    if not isinstance(texts, list) or not texts:
        return jsonify({"error": "No texts provided"}), 400

    include_steps = bool(data.get("include_steps", False))
    results = []
    for text in texts:
        if not isinstance(text, str):
            return jsonify({"error": "Each text must be a string"}), 400
        processed, steps = shared_preprocess_text(text, return_steps=True)
        results.append(
            {
                "original": text,
                "processed": processed,
                "stages": steps if include_steps else None,
            }
        )
    return jsonify({"success": True, "results": results})


@app.route("/api/v1/aspects", methods=["POST"])
def detect_aspects():
    data = request.get_json(silent=True) or {}
    texts = data.get("texts")
    if not isinstance(texts, list) or not texts:
        return jsonify({"error": "No texts provided"}), 400

    engine = AspectDetectionEngine()
    results = []
    for text in texts:
        aspects = engine.detect_in_text(text)
        results.append({"text": text, "aspects": {k: v for k, v in aspects.items() if v}})
    return jsonify({"success": True, "results": results})


@app.route("/api/v1/label", methods=["POST"])
def label_sentiment():
    """
    Pelabelan berbasis aturan lexicon (jalur RISET), bukan prediksi model.

    Endpoint ini memakai kamus lexicon untuk memberi label sentimen pada
    teks. Hasilnya BUKAN keluaran model SVM dan tidak boleh dibandingkan
    dengan metrik penelitian. Untuk prediksi model, gunakan
    ``POST /api/v1/predict`` atau ``POST /api/v1/predict-single``.
    """
    data = request.get_json(silent=True) or {}
    texts = data.get("texts")
    if not isinstance(texts, list) or not texts:
        return jsonify({"error": "No texts provided"}), 400

    engine = SentimentLabelingEngine()
    results = []
    labels = []
    for text in texts:
        processed = shared_preprocess_text(text)
        label, score, norm_score = engine.label_text(processed)
        labels.append(label)
        results.append(
            {
                "text": text,
                "processed_text": processed,
                "sentiment_label": label,
                "sentiment_score": float(score),
                "sentiment_normalized": float(norm_score),
            }
        )

    distribution = engine.get_distribution(pd.DataFrame({"sentiment_label": labels}))
    return jsonify({
        "success": True,
        "distribution": distribution,
        "results": results,
        "labeling_scope": "lexicon_rule_based",
        "is_model_prediction": False,
    })


@app.route("/api/v1/analyze", methods=["POST"])
def run_analysis():
    """Run canonical model inference plus keyword aspect detection."""
    data = request.get_json(silent=True) or {}
    texts = data.get("texts", [])
    if not texts and "text" in data:
        texts = [data["text"]]
    if not isinstance(texts, list) or not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
        return jsonify({"error": "Non-empty text strings are required"}), 400
    results = _predict_texts(texts)
    engine = AspectDetectionEngine()
    for result in results:
        aspects = engine.detect_in_text(result["text"])
        result["detected_aspects"] = {k: v for k, v in aspects.items() if v}
    counts = Counter(row["predicted_label"] for row in results)
    return jsonify({"success": True, "model_version": results[0]["model_version"],
        "total_reviews": len(results), "distribution": {
            label: {"count": counts[label], "percentage": counts[label] / len(results) * 100}
            for label in CANONICAL_LABELS}, "results": results,
        "prediction_scope": "svm_model_canonical",
        "is_model_prediction": True})


# -----------------------------------------------------------------------------
# CANONICAL MODEL INFERENCE
# -----------------------------------------------------------------------------


@app.route("/api/v1/predict", methods=["POST"])
def predict():
    """Predict using models/final_pipeline.joblib only."""
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    texts = data.get("texts")

    if text is not None:
        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "text must be a non-empty string"}), 400
        texts = [text]
    elif texts is not None:
        if not isinstance(texts, list) or not texts:
            return jsonify({"error": "texts must be a non-empty array"}), 400
    else:
        return jsonify({"error": "No text provided"}), 400

    if any(not isinstance(value, str) or not value.strip() for value in texts):
        return jsonify({"error": "Every text must be a non-empty string"}), 400
    try:
        results = _predict_texts(texts)
        aspect_engine = AspectDetectionEngine()
        for result in results:
            aspects = aspect_engine.detect_in_text(result["text"])
            result["detected_aspects"] = {k: v for k, v in aspects.items() if v}
        return jsonify(
            {
                "success": True,
                "model": "final_pipeline.joblib",
                "results": results,
                "prediction_scope": "svm_model_canonical",
                "is_model_prediction": True,
            }
        )
    except Exception as exc:
        app.logger.exception("Prediction gagal")
        return jsonify({"error": str(exc)}), (400 if isinstance(exc, ValueError) and "preprocessing" in str(exc) else 503)


@app.route("/api/v1/evaluate", methods=["POST"])
def evaluate_model():
    """Evaluate the locked final model on supplied labeled data, without training."""
    data = request.get_json(silent=True) or {}
    texts = data.get("texts", [])
    labels = data.get("labels", [])
    if not isinstance(texts, list) or not isinstance(labels, list):
        return jsonify({"error": "texts and labels must be arrays"}), 400
    if len(texts) != len(labels):
        return jsonify({"error": "Texts and labels must have same length"}), 400
    if not texts:
        return jsonify({"error": "No evaluation data provided"}), 400

    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )

        results = _predict_texts(texts)
        predicted = [item["predicted_label"] for item in results]
        actual = [_normalize_label(label) for label in labels]
        if any(label not in CANONICAL_LABELS for label in actual):
            return jsonify({"error": "Label aktual tidak dikenal"}), 400

        metrics = {
            "accuracy": float(accuracy_score(actual, predicted)),
            "precision_macro": float(
                precision_score(actual, predicted, labels=CANONICAL_LABELS, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(actual, predicted, labels=CANONICAL_LABELS, average="macro", zero_division=0)
            ),
            "f1_macro": float(
                f1_score(actual, predicted, labels=CANONICAL_LABELS, average="macro", zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(
                actual, predicted, labels=CANONICAL_LABELS
            ).tolist(),
            "confusion_matrix_labels": CANONICAL_LABELS,
        }
        return jsonify(
            {
                "success": True,
                "model": "final_pipeline.joblib",
                "metrics": metrics,
                "classification_report": classification_report(
                    actual,
                    predicted,
                    labels=CANONICAL_LABELS,
                    zero_division=0,
                ),
                # Metrik ini berlaku untuk data yang dikirim pada permintaan
                # ini saja, bukan untuk performa model pada penelitian.
                "evaluation_scope": "uploaded_dataset",
                "is_model_prediction": True,
            }
        )
    except Exception as exc:
        app.logger.exception("Evaluation gagal")
        return jsonify({"error": str(exc)}), (503 if isinstance(exc, DeploymentIntegrityError) else 500)


# -----------------------------------------------------------------------------
# TRAINING GUARD
# -----------------------------------------------------------------------------


@app.route("/api/v1/train", methods=["POST"])
def train_model():
    """Training endpoint intentionally disabled after final model lock."""
    return (
        jsonify(
            {
                "success": False,
                "error": "Training endpoint disabled",
                "message": (
                    "Model final sudah dikunci. Training/retraining dari API "
                    "dinonaktifkan untuk menjaga reproducibility."
                ),
                "canonical_model": str(FINAL_PIPELINE_PATH),
            }
        ),
        405,
    )


# -----------------------------------------------------------------------------
# FINAL CONFIG ENDPOINT
# -----------------------------------------------------------------------------


@app.route("/api/v1/config")
def get_config():
    """Return final configuration from shared.final_config/backend.config."""
    try:
        _, metadata = load_verified_pipeline()
    except (ValueError, OSError, KeyError) as exc:
        return jsonify({"error": str(exc), "success": False}), 503
    return jsonify(
        {
            "success": True,
            "model_version": metadata["model_version"],
            "integrity_verified": True,
            "evaluation_provenance": metadata["evaluation_provenance"],
            "config": config.get_final_config() if hasattr(config, "get_final_config") else _get_model_config_payload(),
            "model_path": str(FINAL_PIPELINE_PATH),
            "aspects": config.ASPECTS,
            "sentiment_classes": config.SENTIMENT_CLASSES,
            "neutral_threshold": config.NEUTRAL_THRESHOLD,
            "tfidf": {
                "max_features": config.TFIDF_MAX_FEATURES,
                "ngram_range": list(config.TFIDF_NGRAM_RANGE),
                "sublinear_tf": config.TFIDF_SUBLINEAR_TF,
                "min_df": config.TFIDF_MIN_DF,
                "max_df": config.TFIDF_MAX_DF,
            },
            "svm": {
                "kernel": config.SVM_KERNEL,
                "C": config.SVM_C,
                "gamma": config.SVM_GAMMA,
                "class_weight": config.SVM_CLASS_WEIGHT,
            },
            "multiclass": config.MULTICLASS_STRATEGY,
            "imbalance": config.IMBALANCE_METHOD,
            "split": {
                "train_size": config.TRAIN_TEST_SPLIT_RATIO,
                "test_size": config.TEST_SIZE,
                "random_state": config.RANDOM_STATE,
                "stratify": config.STRATIFY,
            },
            "cv_folds": config.CV_FOLDS,
            "selection_metric": config.SELECTION_METRIC,
            "labeling_method": config.LABELING_METHOD,
        }
    )


# -----------------------------------------------------------------------------
# WORDCLOUD / N-GRAM
# -----------------------------------------------------------------------------


@app.route("/api/v1/wordcloud", methods=["POST"])
def get_wordcloud():
    data = request.get_json(silent=True) or {}
    texts = data.get("texts")
    if not isinstance(texts, list) or not texts:
        return jsonify({"error": "No texts provided"}), 400

    words = re.findall(r"\b[\w]+\b", " ".join(map(str, texts)).lower(), flags=re.UNICODE)
    words = [word for word in words if len(word) > 2]
    frequencies = Counter(words)
    top_words = frequencies.most_common(50)
    max_freq = top_words[0][1] if top_words else 1

    return jsonify(
        {
            "success": True,
            "words": [
                {
                    "text": word,
                    "value": freq,
                    "size": round(10 + (freq / max_freq) * 40, 2),
                }
                for word, freq in top_words
            ],
            "total_words": len(words),
            "unique_words": len(frequencies),
        }
    )


@app.route("/api/v1/ngrams", methods=["POST"])
def get_ngrams():
    data = request.get_json(silent=True) or {}
    texts = data.get("texts")
    n = data.get("n", 2)
    if not isinstance(texts, list) or not texts:
        return jsonify({"error": "No texts provided"}), 400
    try:
        n = int(n)
    except (TypeError, ValueError):
        return jsonify({"error": "n must be an integer"}), 400
    if n < 1 or n > 5:
        return jsonify({"error": "n must be between 1 and 5"}), 400

    counter = Counter()
    for text in texts:
        words = re.findall(r"\b[\w]+\b", str(text).lower(), flags=re.UNICODE)
        counter.update(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))

    return jsonify(
        {
            "success": True,
            "ngrams": [
                {"ngram": ngram, "frequency": frequency}
                for ngram, frequency in counter.most_common(20)
            ],
            "n": n,
        }
    )


# -----------------------------------------------------------------------------
# ERROR HANDLERS / MAIN
# -----------------------------------------------------------------------------


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(_error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS WAYGROUND - BACKEND")
    print("=" * 60)
    print("Server: http://127.0.0.1:5001")
    print("API:    http://127.0.0.1:5001/api/v1")
    print("Model:  models/final_pipeline.joblib")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5001, debug=True)
