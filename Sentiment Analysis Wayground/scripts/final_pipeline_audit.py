"""
FINAL PIPELINE AUDIT - Wayground Sentiment Analysis
===================================================

Audit akhir setelah final model selesai dilatih.

Memeriksa:
1. final dataset validation report
2. best_config.json vs shared/final_config.py
3. final_pipeline.joblib dan struktur pipeline
4. TF-IDF + OneVsRestClassifier + SVM Linear
5. metadata.json
6. final evaluation artifacts
7. holdout prediction consistency
8. metrik dan confusion matrix consistency
9. konfigurasi holdout tidak digunakan untuk selection

Script ini bersifat READ-ONLY terhadap artifact model dan dataset.
Ia hanya membuat laporan audit.

Jalankan dari root project:
    python scripts/final_pipeline_audit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from shared.constants import EXPERIMENTS_DIR, AUDIT_DIR  # noqa: E402

from shared.final_config import (  # noqa: E402
    DATASET_RELATIVE_PATH,
    FINAL_CLASSIFICATION_REPORT_PATH,
    FINAL_CONFUSION_MATRIX_PATH,
    FINAL_EVALUATION_PATH,
    FINAL_METADATA_PATH,
    FINAL_PIPELINE_PATH,
    FINAL_PREDICTIONS_PATH,
    FINAL_MODEL_CONFIG,
    LABEL_COLUMN,
    TEXT_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    TRAIN_TEST_SPLIT_RATIO,
    verify_best_config_agrees,
)

REPORT_PATH = AUDIT_DIR / "final_pipeline_audit.json"
VALIDATION_REPORT_PATH = AUDIT_DIR / "final_dataset_validation.json"
DATASET_PATH = PROJECT_ROOT / DATASET_RELATIVE_PATH

EXPECTED_CLASSES = ["negatif", "netral", "positif"]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON bukan object: {path}")
    return data


def check(label: str, condition: bool, details: str = "") -> dict[str, Any]:
    return {"name": label, "status": "PASS" if condition else "FAIL", "details": details}


def main() -> int:
    print("=" * 80)
    print("FINAL PIPELINE AUDIT - WAYGROUND")
    print("=" * 80)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # 1. Dataset validation report
    # ------------------------------------------------------------------
    try:
        validation = load_json(VALIDATION_REPORT_PATH)
        passed = validation.get("status") == "PASS"
        checks.append(check("Final dataset validation", passed, str(validation.get("status"))))
        if not passed:
            errors.append("Final dataset validation berstatus FAIL.")
    except Exception as exc:
        checks.append(check("Final dataset validation", False, str(exc)))
        errors.append(f"Validation report tidak tersedia/valid: {exc}")

    # ------------------------------------------------------------------
    # 2. Configuration lock
    # ------------------------------------------------------------------
    try:
        verify_best_config_agrees()
        checks.append(check("best_config == final_config", True))
    except Exception as exc:
        checks.append(check("best_config == final_config", False, str(exc)))
        errors.append(f"Konfigurasi final tidak konsisten: {exc}")

    # ------------------------------------------------------------------
    # 3. Required artifacts
    # ------------------------------------------------------------------
    required_artifacts = {
        "final_pipeline.joblib": FINAL_PIPELINE_PATH,
        "metadata.json": FINAL_METADATA_PATH,
        "final_model_evaluation.json": FINAL_EVALUATION_PATH,
        "final_classification_report.json": FINAL_CLASSIFICATION_REPORT_PATH,
        "final_confusion_matrix.csv": FINAL_CONFUSION_MATRIX_PATH,
        "final_holdout_predictions.csv": FINAL_PREDICTIONS_PATH,
    }
    for name, path in required_artifacts.items():
        exists = path.exists()
        checks.append(check(f"Artifact: {name}", exists, str(path.relative_to(PROJECT_ROOT))))
        if not exists:
            errors.append(f"Artifact tidak ditemukan: {path}")

    # ------------------------------------------------------------------
    # 4. Model structure
    # ------------------------------------------------------------------
    pipeline = None
    if FINAL_PIPELINE_PATH.exists():
        try:
            pipeline = joblib.load(FINAL_PIPELINE_PATH)
            is_pipeline = isinstance(pipeline, Pipeline)
            checks.append(check("Model adalah sklearn Pipeline", is_pipeline, type(pipeline).__name__))
            if not is_pipeline:
                errors.append("final_pipeline.joblib bukan sklearn Pipeline.")
            else:
                step_names = list(pipeline.named_steps.keys())
                has_tfidf = "tfidf" in pipeline.named_steps
                has_classifier = "classifier" in pipeline.named_steps
                checks.append(check("Pipeline memiliki TF-IDF", has_tfidf, str(step_names)))
                checks.append(check("Pipeline memiliki classifier", has_classifier, str(step_names)))
                if not has_tfidf or not has_classifier:
                    errors.append(f"Struktur pipeline tidak sesuai: {step_names}")

                if has_tfidf:
                    vectorizer = pipeline.named_steps["tfidf"]
                    params = vectorizer.get_params()
                    expected = FINAL_MODEL_CONFIG["tfidf"]
                    tfidf_ok = (
                        params.get("max_features") == expected["max_features"]
                        and tuple(params.get("ngram_range")) == tuple(expected["ngram_range"])
                        and bool(params.get("sublinear_tf")) == bool(expected["sublinear_tf"])
                        and params.get("min_df") == expected["min_df"]
                        and float(params.get("max_df")) == float(expected["max_df"])
                    )
                    checks.append(check("TF-IDF parameters match final config", tfidf_ok))
                    if not tfidf_ok:
                        errors.append(f"TF-IDF model parameters berbeda: {params}")

                if has_classifier:
                    clf = pipeline.named_steps["classifier"]
                    # TAHAP 5: nama variabel dan label pemeriksaan sebelumnya
                    # menuliskan "One-vs-One" padahal yang dicek adalah
                    # OneVsRestClassifier (OvR). Nama dan label sekarang
                    # sesuai dengan artefak (OvR dipilih Experiment 6).
                    is_ovr = isinstance(clf, OneVsRestClassifier)
                    checks.append(check("Classifier menggunakan One-vs-Rest", is_ovr, type(clf).__name__))
                    if not is_ovr:
                        errors.append("Classifier final bukan OneVsRestClassifier.")
                    else:
                        estimator = clf.estimator
                        is_linear_svc = isinstance(estimator, SVC) and estimator.kernel == "linear"
                        checks.append(check("Base classifier adalah Linear SVM", is_linear_svc, type(estimator).__name__))
                        if not is_linear_svc:
                            errors.append("Base classifier final bukan SVC kernel linear.")
                        if is_linear_svc:
                            c_ok = float(estimator.C) == float(FINAL_MODEL_CONFIG["svm"]["C"])
                            checks.append(check("SVM C match final config", c_ok, f"C={estimator.C}"))
                            if not c_ok:
                                errors.append("Nilai C model final berbeda dari final_config.py.")
        except Exception as exc:
            checks.append(check("Load final model", False, str(exc)))
            errors.append(f"Gagal memuat model final: {exc}")

    # ------------------------------------------------------------------
    # 5. Metadata consistency
    # ------------------------------------------------------------------
    if FINAL_METADATA_PATH.exists():
        try:
            metadata = load_json(FINAL_METADATA_PATH)
            checks.append(check("Metadata JSON valid", True))
            metadata_checks = {
                "holdout_used_for_selection": metadata.get("holdout_used_for_selection") is False,
                "holdout_evaluated_once": metadata.get("holdout_evaluated_once") is True,
            }
            for name, ok in metadata_checks.items():
                checks.append(check(f"Metadata: {name}", ok, repr(metadata.get(name))))
                if not ok:
                    errors.append(f"Metadata {name} tidak sesuai.")
        except Exception as exc:
            checks.append(check("Metadata JSON valid", False, str(exc)))
            errors.append(f"Metadata tidak valid: {exc}")

    # ------------------------------------------------------------------
    # 6. Evaluation artifacts consistency
    # ------------------------------------------------------------------
    if FINAL_EVALUATION_PATH.exists():
        try:
            evaluation = load_json(FINAL_EVALUATION_PATH)
            metrics = evaluation.get("metrics", evaluation.get("holdout_metrics", {}))
            holdout_used = evaluation.get("methodology", {}).get("holdout_used_for_selection")
            if holdout_used is None:
                holdout_used = evaluation.get("holdout_used_for_selection")
            ok = holdout_used is False or holdout_used is None
            if holdout_used is None:
                warnings.append("Field holdout_used_for_selection tidak ditemukan pada evaluation JSON.")
            checks.append(check("Evaluation artifact valid", True))
            checks.append(check("Evaluation tidak memakai holdout untuk selection", ok, repr(holdout_used)))
            if not ok:
                errors.append("Evaluation menyatakan holdout dipakai untuk selection.")
        except Exception as exc:
            checks.append(check("Evaluation artifact valid", False, str(exc)))
            errors.append(f"Evaluation JSON tidak valid: {exc}")

    if FINAL_PREDICTIONS_PATH.exists() and DATASET_PATH.exists():
        try:
            pred = pd.read_csv(FINAL_PREDICTIONS_PATH)
            required_pred = {"actual_sentiment", "predicted_sentiment", "correct", "processed_text"}
            missing_pred = required_pred - set(pred.columns)
            checks.append(check("Holdout prediction columns", not missing_pred, str(sorted(missing_pred))))
            if missing_pred:
                errors.append(f"Kolom prediction hilang: {sorted(missing_pred)}")
            else:
                actual = pred["actual_sentiment"].astype(str).str.casefold()
                predicted = pred["predicted_sentiment"].astype(str).str.casefold()
                correct = pred["correct"].astype(str).str.casefold().isin({"true", "1"})
                accuracy = float(accuracy_score(actual, predicted))
                stored_correct_accuracy = float(correct.mean()) if len(correct) else 0.0
                checks.append(check("Prediction correctness consistent", abs(accuracy - stored_correct_accuracy) < 1e-12, f"computed={accuracy:.6f}, stored={stored_correct_accuracy:.6f}"))

                f1 = float(f1_score(actual, predicted, labels=EXPECTED_CLASSES, average="macro", zero_division=0))
                checks.append(check("Prediction labels are valid", set(actual) <= set(EXPECTED_CLASSES) and set(predicted) <= set(EXPECTED_CLASSES)))
                print(f"\n[HOLDOUT PREDICTIONS]")
                print(f"  Rows          : {len(pred):,}")
                print(f"  Accuracy      : {accuracy:.6f}")
                print(f"  Macro F1      : {f1:.6f}")
        except Exception as exc:
            checks.append(check("Holdout predictions readable", False, str(exc)))
            errors.append(f"Gagal memeriksa predictions: {exc}")

    if FINAL_CONFUSION_MATRIX_PATH.exists() and FINAL_PREDICTIONS_PATH.exists():
        try:
            cm = pd.read_csv(FINAL_CONFUSION_MATRIX_PATH, index_col=0)
            pred = pd.read_csv(FINAL_PREDICTIONS_PATH)
            cm_total = int(cm.to_numpy().sum())
            checks.append(check("Confusion matrix matches holdout size", cm_total == len(pred), f"cm={cm_total}, predictions={len(pred)}"))
            if cm_total != len(pred):
                errors.append("Total confusion matrix berbeda dari jumlah prediction holdout.")
        except Exception as exc:
            checks.append(check("Confusion matrix readable", False, str(exc)))
            errors.append(f"Gagal memeriksa confusion matrix: {exc}")

    # ------------------------------------------------------------------
    # 7. Methodology constants
    # ------------------------------------------------------------------
    checks.append(check("Split is 80:20", TRAIN_TEST_SPLIT_RATIO == 0.80 and TEST_SIZE == 0.20, f"{TRAIN_TEST_SPLIT_RATIO}:{TEST_SIZE}"))
    checks.append(check("Random state is 42", RANDOM_STATE == 42, str(RANDOM_STATE)))
    if not (TRAIN_TEST_SPLIT_RATIO == 0.80 and TEST_SIZE == 0.20):
        errors.append("Split final bukan 80:20.")
    if RANDOM_STATE != 42:
        errors.append("Random state final bukan 42.")

    failed = [item for item in checks if item["status"] == "FAIL"]
    status = "FAIL" if errors or failed else "PASS"

    report = {
        "status": status,
        "project": "Sentiment Analysis Wayground",
        "model_artifact": str(FINAL_PIPELINE_PATH.relative_to(PROJECT_ROOT)),
        "dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "configuration": {
            "split": "80:20",
            "random_state": RANDOM_STATE,
            "text_column": TEXT_COLUMN,
            "label_column": LABEL_COLUMN,
            "tfidf": FINAL_MODEL_CONFIG["tfidf"],
            "svm": FINAL_MODEL_CONFIG["svm"],
            "multiclass": FINAL_MODEL_CONFIG["multiclass"],
        },
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "total_checks": len(checks),
            "passed": sum(x["status"] == "PASS" for x in checks),
            "failed": sum(x["status"] == "FAIL" for x in checks),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[CHECK SUMMARY]")
    print(f"  Total : {len(checks)}")
    print(f"  PASS  : {report['summary']['passed']}")
    print(f"  FAIL  : {report['summary']['failed']}")
    for item in checks:
        mark = "✓" if item["status"] == "PASS" else "✗"
        print(f"  {mark} {item['name']}")
        if item.get("details"):
            print(f"      {item['details']}")

    for warning in warnings:
        print(f"[WARN] {warning}")

    print("\n" + "=" * 80)
    if status == "PASS":
        print("FINAL PIPELINE AUDIT: PASS")
        print("Model final siap digunakan sebagai artifact inference.")
    else:
        print("FINAL PIPELINE AUDIT: FAIL")
        for error in errors:
            print(f"  - {error}")
    print(f"Report: {REPORT_PATH}")
    print("=" * 80)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
