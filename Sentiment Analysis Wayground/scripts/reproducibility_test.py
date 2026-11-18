"""
Reproducibility Test - Final Model, Experiment Based

Konfigurasi final diambil dari Experiment 3-6.
Experiment 7 dibaca sebagai diagnostic tambahan.

Tujuan:
- memastikan 3 run menghasilkan prediksi identik;
- memastikan holdout berasal dari split yang sama;
- memastikan model yang diuji benar-benar mengikuti artifact chain;
- artifact prediksi lama hanya menjadi pemeriksaan tambahan.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from shared.constants import (  # noqa: E402
    EXPERIMENTS_DIR,
    CANONICAL_DATASET_PATH,
)

from shared.experiment_config_loader import (  # noqa: E402
    EXPECTED_CLASSES,
    load_experiment_chain,
    print_chain,
)

DATASET_PATH = CANONICAL_DATASET_PATH
PREDICTIONS_PATH = EXPERIMENTS_DIR / "final_holdout_predictions.csv"
OUTPUT_PATH = EXPERIMENTS_DIR / "reproducibility_test_final.json"

RANDOM_STATE = 42
HOLDOUT_RATIO = 0.20
NUMBER_OF_RUNS = 3


def compute_hash(values) -> str:
    normalized = np.asarray(values, dtype=str)
    payload = "\n".join(normalized.tolist()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")
    required = ["processed_text", "sentiment_label_final"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    df = df[required].copy()
    df["processed_text"] = df["processed_text"].fillna("").astype(str).str.strip()
    df["sentiment_label_final"] = (
        df["sentiment_label_final"].fillna("").astype(str).str.strip().str.lower()
    )
    df = df[(df["processed_text"] != "") & (df["sentiment_label_final"] != "")].reset_index(drop=True)

    invalid = sorted(set(df["sentiment_label_final"]) - set(EXPECTED_CLASSES))
    if invalid:
        raise ValueError(f"Label tidak valid: {invalid}")

    return df


def build_pipeline(final_config):
    tfidf = final_config["tfidf"]
    svm = final_config["svm"]

    base_svm = SVC(
        kernel=svm["kernel"],
        C=svm["C"],
        gamma=svm["gamma"],
        class_weight=svm["class_weight"],
        random_state=RANDOM_STATE,
    )

    if final_config["multiclass"] == "ovo":
        classifier = OneVsOneClassifier(base_svm)
    elif final_config["multiclass"] == "ovr":
        classifier = OneVsRestClassifier(base_svm)
    else:
        raise ValueError(f"Strategi multiclass tidak didukung: {final_config['multiclass']}")

    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=tfidf["max_features"],
                ngram_range=tuple(tfidf["ngram_range"]),
                sublinear_tf=tfidf["sublinear_tf"],
                min_df=tfidf["min_df"],
                max_df=tfidf["max_df"],
            ),
        ),
        ("classifier", classifier),
    ])


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=EXPECTED_CLASSES, average="macro", zero_division=0)
        ),
    }


def load_existing_predictions():
    if not PREDICTIONS_PATH.exists():
        return None

    existing = pd.read_csv(PREDICTIONS_PATH, encoding="utf-8-sig")
    candidates = ["predicted_sentiment", "predicted", "prediction"]
    column = next((c for c in candidates if c in existing.columns), None)

    if column is None:
        return {
            "status": "INVALID_ARTIFACT",
            "reason": f"Kolom prediksi tidak ditemukan. Columns={list(existing.columns)}",
        }

    predictions = (
        existing[column].fillna("").astype(str).str.strip().str.lower().values
    )
    return {
        "status": "LOADED",
        "rows": len(predictions),
        "hash": compute_hash(predictions),
        "predictions": predictions,
    }


def main():
    print("=" * 80)
    print("REPRODUCIBILITY TEST - EXPERIMENT BASED FINAL MODEL")
    print("=" * 80)

    chain = load_experiment_chain(include_exp7=True)
    print_chain(chain)

    final_config = chain["final"]
    df = load_dataset()

    X = df["processed_text"]
    y = df["sentiment_label_final"]

    X_dev, X_holdout, y_dev, y_holdout = train_test_split(
        X,
        y,
        test_size=HOLDOUT_RATIO,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nDataset rows : {len(df)}")
    print(f"Development  : {len(X_dev)}")
    print(f"Holdout      : {len(X_holdout)}")

    dev_hash = compute_hash(X_dev.reset_index(drop=True))
    holdout_text_hash = compute_hash(X_holdout.reset_index(drop=True))
    holdout_label_hash = compute_hash(y_holdout.reset_index(drop=True))

    print(f"Development hash   : {dev_hash}")
    print(f"Holdout text hash  : {holdout_text_hash}")
    print(f"Holdout label hash : {holdout_label_hash}")

    predictions_runs = []
    run_results = []

    for run in range(1, NUMBER_OF_RUNS + 1):
        print(f"\n[RUN {run}/{NUMBER_OF_RUNS}]")
        model = build_pipeline(final_config)
        model.fit(X_dev, y_dev)
        pred = model.predict(X_holdout)

        if len(pred) != len(X_holdout):
            raise RuntimeError(
                f"Jumlah prediksi run {run} ({len(pred)}) != holdout ({len(X_holdout)})."
            )

        m = metrics(y_holdout, pred)
        pred_hash = compute_hash(pred)

        predictions_runs.append(np.asarray(pred, dtype=str))
        run_results.append({
            "run": run,
            "prediction_count": int(len(pred)),
            "prediction_hash": pred_hash,
            **m,
        })

        print(f"  Predictions : {len(pred)}")
        print(f"  Accuracy    : {m['accuracy']:.4f}")
        print(f"  Macro F1    : {m['f1_macro']:.4f}")
        print(f"  SHA-256     : {pred_hash}")

    reference = predictions_runs[0]
    comparisons = []
    all_identical = True

    for idx, pred in enumerate(predictions_runs[1:], start=2):
        equal = bool(np.array_equal(reference, pred))
        different = int(np.sum(reference != pred))
        comparisons.append({
            "run_a": 1,
            "run_b": idx,
            "identical": equal,
            "different_predictions": different,
        })
        all_identical = all_identical and equal

    existing = load_existing_predictions()
    existing_status = "NOT_FOUND"
    existing_match = None

    if existing is not None:
        existing_status = existing["status"]
        if existing_status == "LOADED":
            existing_match = bool(
                existing["rows"] == len(reference)
                and existing["hash"] == compute_hash(reference)
            )

    verdict = "PASS" if all_identical else "FAIL"

    print("\n" + "=" * 80)
    print("REPRODUCIBILITY RESULT")
    print("=" * 80)
    print(f"Run-to-run consistency : {verdict}")
    print(f"All runs identical     : {all_identical}")
    print(f"Existing artifact      : {existing_status}")
    print(f"Existing artifact match: {existing_match}")

    payload = {
        "experiment": "reproducibility_test_final",
        "verdict": verdict,
        "configuration_source": {
            "exp3": str(chain["exp3"]["path"]),
            "exp4": str(chain["exp4"]["path"]),
            "exp5": str(chain["exp5"]["path"]),
            "exp6": str(chain["exp6"]["path"]),
            "exp7": str(chain["exp7"]["path"]) if chain["exp7"] else None,
        },
        "final_configuration": final_config,
        "random_state": RANDOM_STATE,
        "holdout_ratio": HOLDOUT_RATIO,
        "number_of_runs": NUMBER_OF_RUNS,
        "dataset": {
            "total_rows": int(len(df)),
            "development_rows": int(len(X_dev)),
            "holdout_rows": int(len(X_holdout)),
            "development_text_hash": dev_hash,
            "holdout_text_hash": holdout_text_hash,
            "holdout_label_hash": holdout_label_hash,
        },
        "runs": run_results,
        "run_consistency": {
            "all_runs_identical": all_identical,
            "pairwise_comparison": comparisons,
        },
        "existing_artifact": {
            "status": existing_status,
            "match_with_run_1": existing_match,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[SAVED] {OUTPUT_PATH}")
    return verdict == "PASS"


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
