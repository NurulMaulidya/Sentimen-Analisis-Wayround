"""
Experiment 7 - Regularization / Generalization Analysis

Konfigurasi dasar DIAMBIL dari Experiment 3, 4, 5, dan 6.
Parameter yang diuji hanya C.

Experiment 7 bersifat diagnostik:
- winner C dicatat dari hasil Experiment 7;
- perubahan final configuration tetap harus mengikuti aturan selection utama;
- holdout tidak disentuh.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
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
OUTPUT_PATH = EXPERIMENTS_DIR / "regularization_comparison_final.json"
CSV_OUTPUT_PATH = EXPERIMENTS_DIR / "regularization_comparison_final.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Search space Experiment 7.
# C adalah parameter yang memang sedang diuji, jadi tidak boleh diambil
# dari hasil Experiment 7 yang sedang dibuat.
C_VALUES = (
    0.01, 0.10, 0.50, 1.00, 1.50, 2.00
)


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


def build_pipeline(tfidf_config: dict[str, Any], svm_config: dict[str, Any], c_value: float):
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=tfidf_config["max_features"],
                ngram_range=tuple(tfidf_config["ngram_range"]),
                sublinear_tf=tfidf_config["sublinear_tf"],
                min_df=tfidf_config["min_df"],
                max_df=tfidf_config["max_df"],
            ),
        ),
        (
            "svm",
            SVC(
                kernel=svm_config["kernel"],
                C=float(c_value),
                gamma=svm_config["gamma"],
                class_weight=svm_config["class_weight"],
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def train_metrics(model, X, y):
    pred = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(
            f1_score(y, pred, labels=EXPECTED_CLASSES, average="macro", zero_division=0)
        ),
    }


def main():
    print("=" * 80)
    print("EXPERIMENT 7 - REGULARIZATION USING EXPERIMENT CHAIN")
    print("=" * 80)

    chain = load_experiment_chain(include_exp7=False)
    print_chain(chain)

    final_config = chain["final"]
    df = load_dataset()

    X = df["processed_text"]
    y = df["sentiment_label_final"]

    X_dev, X_holdout, y_dev, y_holdout = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nDevelopment : {len(X_dev)}")
    print(f"Holdout     : {len(X_holdout)}")
    print("Holdout     : RESERVED, NOT USED FOR SELECTION")

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
    }

    results = []

    for index, c_value in enumerate(C_VALUES, start=1):
        print(f"\n[{index}/{len(C_VALUES)}] C={c_value}")

        model = build_pipeline(final_config["tfidf"], final_config["svm"], c_value)
        model.fit(X_dev, y_dev)
        train = train_metrics(model, X_dev, y_dev)

        cv_result = cross_validate(
            build_pipeline(final_config["tfidf"], final_config["svm"], c_value),
            X_dev,
            y_dev,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
        )

        cv_f1 = float(np.mean(cv_result["test_macro_f1"]))
        cv_std = float(np.std(cv_result["test_macro_f1"], ddof=1))
        gap = float(train["macro_f1"] - cv_f1)

        row = {
            "C": float(c_value),
            "kernel": final_config["svm"]["kernel"],
            "gamma": final_config["svm"]["gamma"],
            "class_weight": final_config["svm"]["class_weight"],
            "train_accuracy": train["accuracy"],
            "train_macro_f1": train["macro_f1"],
            "cv_accuracy_mean": float(np.mean(cv_result["test_accuracy"])),
            "cv_macro_f1_mean": cv_f1,
            "cv_macro_f1_std": cv_std,
            "train_cv_gap": gap,
            "train_cv_gap_pp": gap * 100,
            "tfidf": final_config["tfidf"],
            "multiclass": final_config["multiclass"],
            "imbalance": final_config["imbalance"],
        }
        results.append(row)

        print(f"  Train Macro F1 : {train['macro_f1']:.4f}")
        print(f"  CV Macro F1    : {cv_f1:.4f} +/- {cv_std:.4f}")
        print(f"  Train-CV Gap   : {gap * 100:.2f} pp")

    ranked = sorted(
        results,
        key=lambda r: (
            -r["cv_macro_f1_mean"],
            r["cv_macro_f1_std"],
            r["C"],
        ),
    )
    best = ranked[0]

    exp4_c = final_config["svm"]["C"]
    locked_row = next(
        (r for r in results if np.isclose(r["C"], exp4_c, rtol=0, atol=1e-12)),
        None,
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT 7 RESULT")
    print("=" * 80)
    print(f"Best C by Exp7 CV        : {best['C']}")
    print(f"Best CV Macro F1         : {best['cv_macro_f1_mean']:.4f}")
    print(f"Best CV Std              : {best['cv_macro_f1_std']:.4f}")
    print(f"Exp4 locked C            : {exp4_c}")

    if locked_row is not None:
        print(f"Exp4 C CV Macro F1       : {locked_row['cv_macro_f1_mean']:.4f}")
        print(
            f"Difference Exp7 winner  : "
            f"{best['cv_macro_f1_mean'] - locked_row['cv_macro_f1_mean']:+.4f}"
        )

    print("\nHoldout tidak digunakan untuk ranking/pemilihan.")

    payload = {
        "experiment": "experiment_7_regularization_final",
        "purpose": [
            "Evaluate effect of SVM C",
            "Use Experiment 3-6 as locked base configuration",
            "Diagnose train-CV generalization gap",
            "Keep holdout isolated",
        ],
        "configuration_source": {
            "exp3": str(chain["exp3"]["path"]),
            "exp4": str(chain["exp4"]["path"]),
            "exp5": str(chain["exp5"]["path"]),
            "exp6": str(chain["exp6"]["path"]),
        },
        "dataset": {
            "total_samples": int(len(df)),
            "development_samples": int(len(X_dev)),
            "holdout_samples": int(len(X_holdout)),
            "holdout_used_for_selection": False,
        },
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
        },
        "cv": {
            "folds": CV_FOLDS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
        },
        "locked_configuration": final_config,
        "c_candidates": list(C_VALUES),
        "selection_rule": {
            "primary": "CV Macro F1 mean descending",
            "secondary": "CV Macro F1 standard deviation ascending",
            "tertiary": "lower C",
            "holdout": "reserved for final evaluation",
        },
        "best_by_exp7": best,
        "comparison_with_exp4": {
            "exp4_C": exp4_c,
            "winner_is_same_as_exp4": bool(np.isclose(best["C"], exp4_c, rtol=0, atol=1e-12)),
            "note": (
                "Experiment 7 is diagnostic. Final model configuration should only "
                "change if the stated model-selection policy explicitly permits it."
            ),
        },
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(results).to_csv(CSV_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\n[SAVED]")
    print(f"  JSON : {OUTPUT_PATH}")
    print(f"  CSV  : {CSV_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
