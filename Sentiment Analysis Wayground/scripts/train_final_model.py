"""
Train Final Model - Wayground Sentiment Analysis
================================================

CANONICAL training entry point for the final model.

This script is the single source of truth for producing the inference
artifact used by the website:

    models/final_v4/final_pipeline.joblib   (TF-IDF + OneVsOneClassifier + SVM)
    models/final_v4/metadata.json
    output_v4/experiments/final_model_evaluation.json
    output_v4/experiments/final_classification_report.json
    output_v4/experiments/final_confusion_matrix.csv
    output_v4/experiments/final_holdout_predictions.csv

Design rules (see metodologi_final.md):
  * Configuration comes from ``shared/final_config.py`` (single source of truth)
    and is cross-checked against ``output/experiments/best_config.json``.
  * The holdout is used for the FINAL evaluation ONLY. It is never used to
    select hyper-parameters. Selection was done by cross-validation on the
    development set (Experiment 3 and 4).
  * The model is trained on the DEVELOPMENT split only, so the deployed
    artifact is exactly the artifact that was evaluated on the holdout.
  * TF-IDF is fitted inside the Pipeline, therefore only on the training fold.

Run from the project root:
    python scripts/train_final_model.py
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from shared.constants import EXPERIMENTS_DIR  # noqa: E402

from shared.final_config import (  # noqa: E402
    DATASET_RELATIVE_PATH,
    EXPECTED_CLASS_DISTRIBUTION,
    EXPECTED_DATASET_ROWS,
    FINAL_CLASSIFICATION_REPORT_PATH,
    FINAL_CONFUSION_MATRIX_PATH,
    FINAL_EVALUATION_PATH,
    FINAL_METADATA_PATH,
    FINAL_MODEL_CONFIG,
    FINAL_PIPELINE_PATH,
    FINAL_PREDICTIONS_PATH,
    LABEL_COLUMN,
    RANDOM_STATE,
    STRATIFY,
    TEST_SIZE,
    TEXT_COLUMN,
    TRAIN_TEST_SPLIT_RATIO,
)
from shared.final_config import (
    verify_best_config_agrees as verify_shared_best_config_agrees,
)
from shared import final_config  # noqa: E402

BEST_CONFIG_PATH = EXPERIMENTS_DIR / "best_config.json"
DATASET_PATH = PROJECT_ROOT / DATASET_RELATIVE_PATH
EXPECTED_CLASSES = ["negatif", "netral", "positif"]

# Canonical mapping of the three label spellings that may appear in a file.
LABEL_ALIASES = {"positive": "positif", "neutral": "netral", "negative": "negatif"}


# =============================================================================
# HELPERS
# =============================================================================

def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Format JSON tidak valid: {path}")
    return data


def _read_dataset(path: Path) -> pd.DataFrame:
    """Read the canonical dataset, tolerating the delimiters used by the pipeline."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset canonical tidak ditemukan: {path}")
    required = {TEXT_COLUMN, LABEL_COLUMN}
    last_error: Exception | None = None
    for delimiter in (",", ";", "\t"):
        try:
            candidate = pd.read_csv(
                path, sep=delimiter, encoding="utf-8-sig", engine="python"
            )
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
            continue
        if required.issubset(candidate.columns):
            return candidate
    details = f" Kolom yang dicari: {sorted(required)}."
    if last_error:
        details += f" Error terakhir: {last_error}"
    raise ValueError(f"Gagal membaca dataset dengan delimiter yang didukung.{details}")


def _prepare_dataset(path: Path) -> pd.DataFrame:
    df = _read_dataset(path)

    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str).str.strip()
    df[LABEL_COLUMN] = (
        df[LABEL_COLUMN].fillna("").astype(str).str.strip().str.casefold()
    )
    df[LABEL_COLUMN] = df[LABEL_COLUMN].replace(LABEL_ALIASES)

    before = len(df)
    df = df.loc[(df[TEXT_COLUMN] != "") & (df[LABEL_COLUMN] != "")].copy()
    df = df.reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"[WARN] Baris dengan teks/label kosong dibuang: {removed}")

    actual_classes = set(df[LABEL_COLUMN].unique())
    invalid = actual_classes - set(EXPECTED_CLASSES)
    if invalid:
        raise ValueError(
            f"Label pada {LABEL_COLUMN} tidak valid: {sorted(invalid)}. "
            f"Label yang diizinkan: {EXPECTED_CLASSES}"
        )
    if actual_classes != set(EXPECTED_CLASSES):
        raise ValueError(
            f"Dataset wajib memiliki tiga kelas {EXPECTED_CLASSES}; "
            f"ditemukan {sorted(actual_classes)}"
        )
    return df


def _distribution(labels) -> dict[str, int]:
    counts = pd.Series(labels).value_counts()
    return {label: int(counts.get(label, 0)) for label in EXPECTED_CLASSES}


def _print_distribution(title: str, labels) -> dict[str, int]:
    dist = _distribution(labels)
    print(f"\n{title}")
    for label in EXPECTED_CLASSES:
        print(f"  {label:<8}: {dist[label]}")
    return dist


# =============================================================================
# CONFIGURATION CONSISTENCY
# =============================================================================

def verify_best_config_agrees() -> None:
    """
    The locked experiment config file must agree with shared/final_config.py.

    Delegasi ke ``shared.final_config.verify_best_config_agrees`` supaya
    sumber kebenaran hanya satu dan validasi tidak terduplikasi.
    """
    verify_shared_best_config_agrees()
    print("[OK] best_config.json agrees with shared/final_config.py")


def _selection_evidence() -> list[str]:
    """
    Daftar artefak eksperimen yang benar-benar dipakai untuk memilih
    konfigurasi final.

    TAHAP 5: sebelumnya daftar ini ditulis manual
    ("output/experiments/experiment_3_tfidf_results.json") sehingga
    bisa menunjuk artefak lama yang tidak lagi dipakai. Sekarang dibaca dari
    artifact chain yang sama dengan yang dipakai lock_final_config.py.
    """
    try:
        from shared.experiment_config_loader import load_experiment_chain

        chain = load_experiment_chain(include_exp7=True)

        paths = [
            str(chain["exp3"]["path"]),
            str(chain["exp4"]["path"]),
            str(chain["exp5"]["path"]),
            str(chain["exp6"]["path"]),
        ]

        if chain.get("exp7") is not None:
            paths.append(str(chain["exp7"]["path"]))

        return paths
    except Exception as exc:
        return [f"(artifact chain tidak terbaca: {exc})"]


# =============================================================================
# MODEL BUILDERS
# =============================================================================

def build_vectorizer() -> TfidfVectorizer:
    cfg = FINAL_MODEL_CONFIG["tfidf"]
    return TfidfVectorizer(
        max_features=int(cfg["max_features"]),
        ngram_range=tuple(int(x) for x in cfg["ngram_range"]),
        sublinear_tf=bool(cfg["sublinear_tf"]),
        min_df=int(cfg["min_df"]),
        max_df=float(cfg["max_df"]),
    )


def build_final_pipeline():
    """
    Pipeline final sesuai konfigurasi terkunci.

    TAHAP 5: metode ketidakseimbangan kelas dari Experiment 5 ikut
    diterapkan. Untuk metode resampling (random_over/random_under/smote)
    pipeline harus memakai `imblearn.pipeline.Pipeline`, karena pada
    pipeline sklearn biasa sampler akan ikut dipanggil saat prediksi.
    Pipeline imblearn otomatis melewati sampler pada tahap prediksi.
    """
    method = str(
        FINAL_MODEL_CONFIG["imbalance"].get("method", "none")
    ).strip().lower()

    steps: list[tuple[str, object]] = [("tfidf", build_vectorizer())]

    if method in {"none", "", None}:
        steps.append(("classifier", build_classifier()))
        return Pipeline(steps=steps)

    if method == "class_weight_balanced":
        # handled by class_weight pada estimator SVM
        steps.append(("classifier", build_classifier()))
        return Pipeline(steps=steps)

    try:
        from imblearn.over_sampling import RandomOverSampler, SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.under_sampling import RandomUnderSampler
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Metode imbalance "
            f"{method!r} membutuhkan imbalanced-learn. "
            "Pasang dengan: pip install imbalanced-learn"
        ) from exc

    if method == "random_over":
        sampler = RandomOverSampler(random_state=RANDOM_STATE)
    elif method == "random_under":
        sampler = RandomUnderSampler(random_state=RANDOM_STATE)
    elif method == "smote":
        sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    else:
        raise ValueError(
            f"Metode imbalance tidak didukung: {method!r}. "
            "Pilihan: none, class_weight_balanced, random_over, "
            "random_under, smote."
        )

    steps.append(("sampler", sampler))
    steps.append(("classifier", build_classifier()))

    return ImbPipeline(steps=steps)


def build_classifier():
    """Classifier honouring the locked multiclass strategy (OvO / OvR)."""
    cfg = FINAL_MODEL_CONFIG["svm"]
    strategy = FINAL_MODEL_CONFIG["multiclass"]["strategy"]
    class_weight = cfg.get("class_weight")

    # TAHAP 5: metode ketidakseimbangan 'class_weight_balanced' diterapkan
    # langsung pada estimator SVM.
    imbalance_method = str(
        FINAL_MODEL_CONFIG["imbalance"].get("method", "none")
    ).strip().lower()
    if imbalance_method == "class_weight_balanced":
        class_weight = "balanced"

    base_svm = SVC(
        kernel=str(cfg["kernel"]),
        C=float(cfg["C"]),
        gamma=cfg.get("gamma", "scale"),
        class_weight=class_weight,
        random_state=RANDOM_STATE,
    )
    if strategy == "ovo":
        return OneVsOneClassifier(base_svm)
    if strategy == "ovr":
        from sklearn.multiclass import OneVsRestClassifier

        return OneVsRestClassifier(base_svm)
    raise ValueError(
        f"Strategi multiclass tidak dikenal: {strategy!r}. "
        "Hanya 'ovo' dan 'ovr' yang didukung."
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    from shared.deployment_integrity import (
        verify_selection_contract, evaluation_provenance, runtime_fingerprint,
        research_fingerprint, sha256, verify_fitted_parameters,
    )
    selected, selected_run = verify_selection_contract()
    provenance = evaluation_provenance(selected_run)
    for path in [FINAL_PIPELINE_PATH, FINAL_METADATA_PATH, FINAL_EVALUATION_PATH,
                 FINAL_CLASSIFICATION_REPORT_PATH, FINAL_CONFUSION_MATRIX_PATH, FINAL_PREDICTIONS_PATH]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite final deployment artifact: {path}")

    # Gerbang environment: training di luar environment penelitian terkunci
    # menghasilkan angka yang tidak persis sama dengan artefak canonical.
    # Lihat shared/environment.py dan docs/remediation/ENVIRONMENT-REPRODUCIBILITY.md.
    from shared.environment import require_locked_environment

    print("=" * 80)
    print("FINAL MODEL TRAINING - WAYGROUND SENTIMENT ANALYSIS")
    print("=" * 80)

    print("\n[0/8] CHECKING RESEARCH ENVIRONMENT")
    environment = require_locked_environment()
    print(
        "  Python {python} · numpy {numpy} · pandas {pandas} · "
        "scikit-learn {scikit-learn}".format(**environment)
    )
    print("[OK] Environment sesuai environment penelitian terkunci.")

    print("\n[1/8] LOADING FINAL CONFIGURATION")
    print(f"Source of truth : shared/final_config.py")
    print(f"Cross-check     : {BEST_CONFIG_PATH.relative_to(PROJECT_ROOT)}")
    verify_best_config_agrees()
    final_config.validate_config(require_dataset=False)
    print("[OK] Final configuration validated.")

    tfidf_cfg = FINAL_MODEL_CONFIG["tfidf"]
    svm_cfg = FINAL_MODEL_CONFIG["svm"]
    print("\n[FINAL CONFIGURATION]")
    print(
        f"  Split             : "
        f"{int(TRAIN_TEST_SPLIT_RATIO * 100)}:{int(TEST_SIZE * 100)}"
    )
    print(f"  Random State      : {RANDOM_STATE}")
    print(f"  Stratify          : {STRATIFY}")
    print(f"  TF-IDF Features   : {tfidf_cfg['max_features']}")
    print(f"  TF-IDF N-gram     : {tuple(tfidf_cfg['ngram_range'])}")
    print(f"  Sublinear TF      : {tfidf_cfg['sublinear_tf']}")
    print(f"  min_df / max_df   : {tfidf_cfg['min_df']} / {tfidf_cfg['max_df']}")
    print(f"  SVM               : kernel={svm_cfg['kernel']}, C={svm_cfg['C']}")
    print(f"  Class Weight      : {svm_cfg['class_weight']}")
    print(f"  Imbalance         : {FINAL_MODEL_CONFIG['imbalance']['method']}")
    print(f"  Multiclass        : {FINAL_MODEL_CONFIG['multiclass']['strategy'].upper()}")

    print("\n[2/8] LOADING CANONICAL DATASET")
    print(f"Dataset: {DATASET_PATH.relative_to(PROJECT_ROOT)}")
    df = _prepare_dataset(DATASET_PATH)
    print(f"Rows       : {len(df)}")
    print(f"Text       : {TEXT_COLUMN}")
    print(f"Label      : {LABEL_COLUMN}")
    full_dist = _print_distribution("Full distribution:", df[LABEL_COLUMN])

    if len(df) != EXPECTED_DATASET_ROWS:
        raise ValueError(
            "Jumlah data tidak sesuai dengan dataset eksperimen yang dikunci. "
            f"Expected {EXPECTED_DATASET_ROWS}, actual {len(df)}."
        )
    if full_dist != EXPECTED_CLASS_DISTRIBUTION:
        raise ValueError(
            "Distribusi sentiment_label_final tidak sesuai dengan dataset "
            "eksperimen yang dikunci.\n"
            f"Expected: {EXPECTED_CLASS_DISTRIBUTION}\nActual:   {full_dist}"
        )
    print("[OK] Canonical dataset and label distribution validated.")

    print("\n[3/8] CREATING FIXED DEVELOPMENT / HOLDOUT SPLIT")
    #
    # TAHAP 5: split TIDAK lagi dibuat sendiri dengan `train_test_split`
    # per baris. Split canonical Stage 5 (berkelompok berdasarkan
    # `processed_text`) dipakai supaya:
    #   * holdout final sama dengan holdout yang dipakai eksperimen,
    #   * kebocoran `processed_text` yang dihilangkan Tahap 3 tidak
    #     kembali muncul saat evaluasi akhir.
    #
    from shared.splitting import split_parts_for_dataset

    dev_df, holdout_df, split_info_raw = split_parts_for_dataset(
        df,
        label_col=LABEL_COLUMN,
    )
    if split_info_raw["source"] != "stage5_files":
        raise ValueError("Final training requires the saved canonical split")
    if len(dev_df) != selected["split"]["development_rows"] or len(holdout_df) != selected["split"]["holdout_rows"]:
        raise ValueError("Final training split differs from selected experiment split")

    X_dev = dev_df[TEXT_COLUMN].to_numpy()
    y_dev = dev_df[LABEL_COLUMN].to_numpy()
    X_holdout = holdout_df[TEXT_COLUMN].to_numpy()
    y_holdout = holdout_df[LABEL_COLUMN].to_numpy()
    print(f"Development : {len(X_dev)}")
    print(f"Holdout     : {len(X_holdout)}")
    print(f"Metode      : {split_info_raw.get('method', split_info_raw.get('source', '-'))}")
    print(f"Sumber      : {split_info_raw.get('source', '-')} ({split_info_raw.get('path', '-')})")

    if len(dev_df) + len(holdout_df) != len(df):
        raise RuntimeError(
            "Validasi split gagal: jumlah development + holdout tidak "
            "sama dengan jumlah dataset."
        )

    dev_groups = set(dev_df["processed_text"].astype(str))
    hold_groups = set(holdout_df["processed_text"].astype(str))
    leaking = dev_groups & hold_groups
    if leaking:
        raise RuntimeError(
            f"Validasi split gagal: {len(leaking)} nilai processed_text "
            "muncul di development dan holdout (kebocoran data)."
        )

    _print_distribution("Development distribution:", y_dev)
    _print_distribution("Holdout distribution:", y_holdout)

    print("\n[4/8] BUILDING FINAL PIPELINE")
    final_pipeline = build_final_pipeline()
    strategy_label = {
        "ovo": "OneVsOneClassifier",
        "ovr": "OneVsRestClassifier",
    }.get(
        FINAL_MODEL_CONFIG["multiclass"]["strategy"],
        FINAL_MODEL_CONFIG["multiclass"]["strategy"],
    )
    imbalance_label = FINAL_MODEL_CONFIG["imbalance"]["method"]
    sampler_label = (
        "" if imbalance_label == "none" else f" -> {imbalance_label}"
    )
    print(
        f"[PIPELINE] TF-IDF{sampler_label} -> {strategy_label} -> SVM"
    )

    print("\n[5/8] TRAINING FINAL MODEL")
    print("Training data : DEVELOPMENT ONLY")
    print(f"Training rows : {len(X_dev)}")
    print("Holdout       : NOT USED FOR TRAINING")
    final_pipeline.fit(X_dev, y_dev)
    print("[OK] Final model trained.")

    print("\n[6/8] FINAL HOLDOUT EVALUATION")
    print("Diagnostic evaluation: holdout has prior exposure in invalidated attempts; not a pristine independent test.")
    y_holdout_pred = final_pipeline.predict(X_holdout)
    accuracy = accuracy_score(y_holdout, y_holdout_pred)
    precision = precision_score(
        y_holdout, y_holdout_pred, labels=EXPECTED_CLASSES, average="macro", zero_division=0
    )
    recall = recall_score(
        y_holdout, y_holdout_pred, labels=EXPECTED_CLASSES, average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        y_holdout, y_holdout_pred, labels=EXPECTED_CLASSES, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_holdout, y_holdout_pred, labels=EXPECTED_CLASSES, average="weighted", zero_division=0
    )
    y_dev_pred = final_pipeline.predict(X_dev)
    train_accuracy = accuracy_score(y_dev, y_dev_pred)
    train_macro_f1 = f1_score(
        y_dev, y_dev_pred, labels=EXPECTED_CLASSES, average="macro", zero_division=0
    )
    print("\nFINAL HOLDOUT METRICS")
    print(f"  Accuracy        : {accuracy:.4f}")
    print(f"  Precision Macro : {precision:.4f}")
    print(f"  Recall Macro    : {recall:.4f}")
    print(f"  Macro F1        : {macro_f1:.4f}")
    print(f"  Weighted F1     : {weighted_f1:.4f}")
    print("\nTRAINING DIAGNOSTIC")
    print(f"  Train Accuracy  : {train_accuracy:.4f}")
    print(f"  Train Macro F1  : {train_macro_f1:.4f}")

    print("\n[7/8] CLASSIFICATION REPORT AND CONFUSION MATRIX")
    report = classification_report(
        y_holdout, y_holdout_pred, labels=EXPECTED_CLASSES, target_names=EXPECTED_CLASSES,
        output_dict=True, zero_division=0,
    )
    print(classification_report(
        y_holdout, y_holdout_pred, labels=EXPECTED_CLASSES, target_names=EXPECTED_CLASSES,
        zero_division=0,
    ))
    cm = confusion_matrix(y_holdout, y_holdout_pred, labels=EXPECTED_CLASSES)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual_{label}" for label in EXPECTED_CLASSES],
        columns=[f"Predicted_{label}" for label in EXPECTED_CLASSES],
    )
    print("CONFUSION MATRIX")
    print(cm_df.to_string())
    if int(cm.sum()) != len(X_holdout):
        raise RuntimeError(
            f"Confusion matrix total {int(cm.sum())} != jumlah sampel holdout {len(X_holdout)}"
        )

    print("\n[8/8] SAVING FINAL ARTIFACTS")
    FINAL_PIPELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone(timedelta(hours=7)))
    timestamp = now.isoformat()
    model_version = now.strftime("%Y%m%d_%H%M%S")

    verify_fitted_parameters(final_pipeline, FINAL_MODEL_CONFIG)
    joblib.dump(final_pipeline, FINAL_PIPELINE_PATH)
    cm_df.to_csv(FINAL_CONFUSION_MATRIX_PATH, encoding="utf-8-sig")
    with FINAL_CLASSIFICATION_REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    pred_df = pd.DataFrame({
        "dataset_index": holdout_df.index.to_numpy(),
        "actual_sentiment": y_holdout,
        "predicted_sentiment": y_holdout_pred,
        "correct": y_holdout == y_holdout_pred,
        "processed_text": X_holdout,
    })
    for optional_col in ("reviewId", "content", "rating", "at"):
        if optional_col in holdout_df.columns:
            pred_df.insert(1, optional_col, holdout_df[optional_col].to_numpy())
    pred_df.to_csv(FINAL_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    exposed_ids = set(provenance["exposed_review_ids"])
    subsets = {}
    for name, mask in {
        "previously_exposed": holdout_df.reviewId.isin(exposed_ids).to_numpy(),
        "not_exposed_in_invalidated_cv": ~holdout_df.reviewId.isin(exposed_ids).to_numpy(),
        "expert_labeled": holdout_df.sentiment_label_expert.fillna("").ne("").to_numpy(),
    }.items():
        n = int(mask.sum())
        subsets[name] = {"rows": n,
            "accuracy": float(accuracy_score(y_holdout[mask], y_holdout_pred[mask])) if n else None,
            "f1_macro_three_classes": float(f1_score(y_holdout[mask], y_holdout_pred[mask], labels=EXPECTED_CLASSES, average="macro", zero_division=0)) if n else None,
            "note": "Descriptive subset; independence from historical lexicon development is unverified"}

    dataset_info = {
        "path": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "total_samples": int(len(df)),
        "development_samples": int(len(X_dev)),
        "holdout_samples": int(len(X_holdout)),
        "text_column": TEXT_COLUMN,
        "label_column": LABEL_COLUMN,
        "class_distribution": full_dist,
        "classes": EXPECTED_CLASSES,
    }
    split_info = {
        "train_size": TRAIN_TEST_SPLIT_RATIO,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratify": STRATIFY,
        "method": split_info_raw.get("method", "StratifiedGroupKFold (grup = processed_text)"),
        "source": split_info_raw.get("source", "stage5_files"),
        "path": split_info_raw.get("path", ""),
        "group_column": "processed_text",
        "holdout_used_for_selection": False,
    }
    metrics = {
        "holdout_accuracy": float(accuracy),
        "holdout_precision_macro": float(precision),
        "holdout_recall_macro": float(recall),
        "holdout_f1_macro": float(macro_f1),
        "holdout_f1_weighted": float(weighted_f1),
        "train_accuracy_diagnostic": float(train_accuracy),
        "train_f1_macro_diagnostic": float(train_macro_f1),
    }
    artifacts = {
        "pipeline": str(FINAL_PIPELINE_PATH.relative_to(PROJECT_ROOT)),
        "model": str(FINAL_PIPELINE_PATH.relative_to(PROJECT_ROOT)),
        "vectorizer": str(FINAL_PIPELINE_PATH.relative_to(PROJECT_ROOT)),
        "metadata": str(FINAL_METADATA_PATH.relative_to(PROJECT_ROOT)),
        "predictions": str(FINAL_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)),
        "evaluation": str(FINAL_EVALUATION_PATH.relative_to(PROJECT_ROOT)),
        "classification_report": str(FINAL_CLASSIFICATION_REPORT_PATH.relative_to(PROJECT_ROOT)),
        "confusion_matrix": str(FINAL_CONFUSION_MATRIX_PATH.relative_to(PROJECT_ROOT)),
    }
    versions = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }

    evaluation = {
        "status": "final",
        "model_version": model_version,
        "evaluation_timestamp": timestamp,
        "methodology": {
            "selection_metric": "f1_macro",
            "holdout_used_for_selection": False,
            "evaluation_passes_in_this_training_run": 1,
            "final_evaluation_set": "diagnostic_holdout_with_prior_exposure",
            "training_data": "development_only",
            "selection_evidence": _selection_evidence(),
        },
        "dataset": dataset_info,
        "split": split_info,
        "configuration": FINAL_MODEL_CONFIG,
        "evaluation_provenance": provenance,
        "subsets": subsets,
        "metrics": metrics,
        "holdout_metrics": {
            "accuracy": float(accuracy),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(macro_f1),
            "f1_weighted": float(weighted_f1),
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": EXPECTED_CLASSES,
        "per_class": {
            cls: {
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1-score": report[cls]["f1-score"],
                "support": report[cls]["support"],
            }
            for cls in EXPECTED_CLASSES
        },
        "versions": versions,
        "artifacts": artifacts,
    }
    with FINAL_EVALUATION_PATH.open("w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)

    metadata = {
        "status": "FINAL",
        "model_version": model_version,
        "training_timestamp": timestamp,
        "config_source": "shared/final_config.py",
        "experiment_config": str(BEST_CONFIG_PATH.relative_to(PROJECT_ROOT)),
        "dataset_source": str(DATASET_PATH.relative_to(PROJECT_ROOT)),

        # Holdout governance
        "holdout_used_for_selection": False,
        "evaluation_passes_in_this_training_run": 1,
        "bundle_schema": 1,
        "labeling": selected["labeling"],
        "evaluation_provenance": provenance,
        # Fingerprint sumber daya yang menentukan perilaku inferensi.
        "inference_fingerprint": runtime_fingerprint(),
        # Fingerprint jalur riset (pelabelan lexicon) untuk audit.
        "research_fingerprint": research_fingerprint(),

        "dataset": dataset_info,
        "split": split_info,
        "model": FINAL_MODEL_CONFIG,

        "evaluation": {
            "selection_metric": "f1_macro",
            "holdout_used_for_selection": False,
            "evaluation_passes_in_this_training_run": 1,
            "accuracy": float(accuracy),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(macro_f1),
            "f1_weighted": float(weighted_f1),
            "train_accuracy_diagnostic": float(train_accuracy),
            "train_f1_macro_diagnostic": float(train_macro_f1),
        },
        "versions": versions,
        "artifacts": artifacts,
        "canonical_inference_artifact": str(FINAL_PIPELINE_PATH.relative_to(PROJECT_ROOT)),
    }
    bound_paths = [FINAL_PIPELINE_PATH, FINAL_EVALUATION_PATH, FINAL_CLASSIFICATION_REPORT_PATH,
                   FINAL_CONFUSION_MATRIX_PATH, FINAL_PREDICTIONS_PATH, BEST_CONFIG_PATH,
                   EXPERIMENTS_DIR / "selected_configuration.json", DATASET_PATH,
                   DATASET_PATH.parent / "dataset_manifest.json",
                   DATASET_PATH.parent / "stage5_split_v2/development.csv",
                   DATASET_PATH.parent / "stage5_split_v2/holdout.csv"]
    metadata["artifact_hashes"] = {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in bound_paths}
    metadata["training_code_sha256"] = sha256(__file__)
    with FINAL_METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n[FILES SAVED]")
    for name, path in (
        ("Pipeline", FINAL_PIPELINE_PATH),
        ("Metadata", FINAL_METADATA_PATH),
        ("Evaluation", FINAL_EVALUATION_PATH),
        ("Classification Report", FINAL_CLASSIFICATION_REPORT_PATH),
        ("Confusion Matrix", FINAL_CONFUSION_MATRIX_PATH),
        ("Holdout Predictions", FINAL_PREDICTIONS_PATH),
    ):
        print(f"  {name:<20}: {path}")
    print("\n" + "=" * 80)
    print("FINAL MODEL TRAINING COMPLETE")
    print("=" * 80)
    print("\nFINAL RESULT")
    print(f"  Accuracy        : {accuracy:.4f}")
    print(f"  Precision Macro : {precision:.4f}")
    print(f"  Recall Macro    : {recall:.4f}")
    print(f"  Macro F1        : {macro_f1:.4f}")
    print("\n[IMPORTANT]")
    print(f"  Diagnostic holdout {len(X_holdout)} rows; prior exposure is recorded in metadata.")
    print("  Jangan gunakan holdout untuk tuning atau pemilihan model lagi.")
    print(f"  Artifact inference canonical: {FINAL_PIPELINE_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
