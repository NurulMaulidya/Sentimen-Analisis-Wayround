"""
Experiment 5 - Class Imbalance Handling
Wayground Sentiment Analysis

Artifact chain:
Experiment 3 (TF-IDF) -> Experiment 4 (SVM) -> Experiment 5 (Imbalance)

Experiment 5 does NOT hardcode TF-IDF/SVM as its source configuration.
It reads the actual selected parameters from Experiment 3 and Experiment 4
artifacts, then compares imbalance-handling strategies using 5-fold CV.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.svm import SVC

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
except ImportError as exc:
    raise ImportError(
        "imbalanced-learn belum terpasang. Jalankan: "
        "pip install imbalanced-learn"
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from shared.constants import EXPERIMENTS_DIR  # noqa: E402

from scripts.experiments.common import (
    RANDOM_STATE,
    CV_FOLDS,
    load_canonical_dataset,
    make_fixed_split,
    make_grouped_cv,
    save_json,
)

OUTPUT_DIR = EXPERIMENTS_DIR
RESULTS_PATH = OUTPUT_DIR / "experiment_5_class_imbalance_results.json"

# Experiment 3 is the ONLY source of TF-IDF parameters.
EXP3_PATH = OUTPUT_DIR / "experiment_3_tfidf_results.json"

# Experiment 4 is the ONLY source of SVM parameters.
EXP4_PATH = OUTPUT_DIR / "experiment_4_svm_results.json"

METHODS = (
    "none",
    "class_weight_balanced",
    "random_over",
    "random_under",
    "smote",
)


def load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"{name} artifact tidak ditemukan:\n{path}\n"
            "Jalankan eksperimen sebelumnya terlebih dahulu."
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{name} harus berupa object JSON.")

    return data


def extract_tfidf(exp3: dict[str, Any]) -> dict[str, Any]:
    """
    Ambil konfigurasi TF-IDF dari hasil Experiment 3.
    Tidak menggunakan shared.final_config dan tidak meng-hardcode
    parameter hasil eksperimen.
    """
    candidates: list[dict[str, Any]] = []

    for key in ("locked_tfidf", "best", "best_configuration"):
        value = exp3.get(key)
        if isinstance(value, dict):
            if all(k in value for k in (
                "max_features", "ngram_range",
                "sublinear_tf", "min_df", "max_df"
            )):
                candidates.append(value)

            for nested_key in ("tfidf", "best_tfidf", "configuration"):
                nested = value.get(nested_key)
                if isinstance(nested, dict):
                    candidates.append(nested)

    # Some Experiment 3 versions store the selected TF-IDF directly
    # under "best".
    for candidate in candidates:
        required = {
            "max_features",
            "ngram_range",
            "sublinear_tf",
            "min_df",
            "max_df",
        }
        if required.issubset(candidate):
            ngram = candidate["ngram_range"]
            if isinstance(ngram, list):
                ngram = tuple(ngram)

            return {
                "max_features": int(candidate["max_features"]),
                "ngram_range": tuple(ngram),
                "sublinear_tf": bool(candidate["sublinear_tf"]),
                "min_df": int(candidate["min_df"]),
                "max_df": float(candidate["max_df"]),
            }

    raise ValueError(
        "Konfigurasi TF-IDF dari Experiment 3 tidak ditemukan."
    )


def extract_svm(exp4: dict[str, Any]) -> dict[str, Any]:
    """
    Ambil konfigurasi SVM terpilih dari hasil Experiment 4.
    Tidak menggunakan shared.final_config dan tidak meng-hardcode
    parameter hasil eksperimen.
    """
    candidates: list[dict[str, Any]] = []

    for key in ("locked_svm", "best", "best_configuration"):
        value = exp4.get(key)
        if isinstance(value, dict):
            if "kernel" in value and "C" in value:
                candidates.append(value)

            for nested_key in ("svm", "best_svm", "configuration"):
                nested = value.get(nested_key)
                if isinstance(nested, dict):
                    candidates.append(nested)

    for candidate in candidates:
        if "kernel" in candidate and "C" in candidate:
            return {
                "kernel": str(candidate["kernel"]).lower(),
                "C": float(candidate["C"]),
                "gamma": candidate.get("gamma", "scale"),
            }

    raise ValueError(
        "Konfigurasi SVM dari Experiment 4 tidak ditemukan."
    )


def validate_artifact_chain(
    tfidf: dict[str, Any],
    svm: dict[str, Any],
) -> None:
    errors: list[str] = []

    required_tfidf = {
        "max_features",
        "ngram_range",
        "sublinear_tf",
        "min_df",
        "max_df",
    }
    required_svm = {"kernel", "C", "gamma"}

    missing_tfidf = required_tfidf.difference(tfidf)
    missing_svm = required_svm.difference(svm)

    if missing_tfidf:
        errors.append(
            f"TF-IDF dari Experiment 3 missing: {sorted(missing_tfidf)}"
        )

    if missing_svm:
        errors.append(
            f"SVM dari Experiment 4 missing: {sorted(missing_svm)}"
        )

    if tfidf.get("max_features", 0) <= 0:
        errors.append("TF-IDF max_features harus > 0.")

    if tfidf.get("min_df", 0) <= 0:
        errors.append("TF-IDF min_df harus > 0.")

    if not 0 < float(tfidf.get("max_df", 0)) <= 1:
        errors.append("TF-IDF max_df harus berada pada (0, 1].")

    if svm.get("kernel") not in {"linear", "rbf", "poly", "sigmoid"}:
        errors.append(f"Kernel SVM tidak valid: {svm.get('kernel')!r}")

    if float(svm.get("C", 0)) <= 0:
        errors.append("SVM C harus > 0.")

    if errors:
        raise RuntimeError(
            "\n[ARTIFACT CHAIN ERROR]\n"
            + "\n".join(f"  - {error}" for error in errors)
            + "\nExperiment 5 dihentikan."
        )


def build_estimator(
    tfidf: dict[str, Any],
    svm: dict[str, Any],
    method: str,
):
    vectorizer = TfidfVectorizer(
        max_features=tfidf["max_features"],
        ngram_range=tuple(tfidf["ngram_range"]),
        sublinear_tf=tfidf["sublinear_tf"],
        min_df=tfidf["min_df"],
        max_df=tfidf["max_df"],
    )

    svm_params: dict[str, Any] = {
        "kernel": svm["kernel"],
        "C": svm["C"],
        "class_weight": (
            "balanced"
            if method == "class_weight_balanced"
            else None
        ),
    }

    # gamma is irrelevant for linear SVM.
    if svm["kernel"] != "linear":
        svm_params["gamma"] = svm["gamma"]

    classifier = SVC(**svm_params)

    if method in {"none", "class_weight_balanced"}:
        return SklearnPipeline(
            [
                ("tfidf", vectorizer),
                ("svm", classifier),
            ]
        )

    if method == "random_over":
        sampler = RandomOverSampler(random_state=RANDOM_STATE)
    elif method == "random_under":
        sampler = RandomUnderSampler(random_state=RANDOM_STATE)
    elif method == "smote":
        # SMOTE cannot operate directly on sparse TF-IDF reliably.
        # Use SMOTE only if the vectorizer output is accepted by the
        # installed imbalanced-learn version.
        sampler = SMOTE(
            random_state=RANDOM_STATE,
            k_neighbors=3,
        )
    else:
        raise ValueError(f"Metode tidak dikenal: {method}")

    return ImbPipeline(
        [
            ("tfidf", vectorizer),
            ("sampler", sampler),
            ("svm", classifier),
        ]
    )


def evaluate_method(
    X_train,
    y_train,
    tfidf: dict[str, Any],
    svm: dict[str, Any],
    method: str,
    cv,
    groups_train=None,
) -> dict[str, Any]:
    start = time.perf_counter()

    estimator = build_estimator(
        tfidf=tfidf,
        svm=svm,
        method=method,
    )

    scores = cross_val_score(
        estimator,
        X_train,
        y_train,
        cv=cv,
        groups=groups_train,
        scoring="f1_macro",
        n_jobs=1,
        error_score="raise",
    )

    return {
        "method": method,
        "cv_macro_f1_mean": float(np.mean(scores)),
        "cv_macro_f1_std": float(np.std(scores, ddof=1)),
        "fold_scores": [float(score) for score in scores],
        "runtime_seconds": round(
            time.perf_counter() - start, 4
        ),
    }


def main() -> dict[str, Any]:
    print("=" * 80)
    print("EXPERIMENT 5 - CLASS IMBALANCE HANDLING")
    print("=" * 80)

    df, text_column, label_column = load_canonical_dataset()

    X_train, X_holdout, y_train, y_holdout, groups_train = make_fixed_split(
        df,
        text_column=text_column,
        label_column=label_column,
        return_groups=True,
    )

    # Holdout exists only for the final evaluation stage.
    # It is never passed to cross_val_score.
    print("\n[FIXED SPLIT]")
    print(f"Train        : {len(X_train)}")
    print(f"Holdout      : {len(X_holdout)}")
    print(f"Random State : {RANDOM_STATE}")
    print("Stratify     : True")
    print("Holdout used for selection: NO")

    # ------------------------------------------------------------
    # Artifact chain
    # ------------------------------------------------------------
    exp3 = load_json(EXP3_PATH, "Experiment 3")
    exp4 = load_json(EXP4_PATH, "Experiment 4")

    tfidf = extract_tfidf(exp3)
    svm = extract_svm(exp4)

    validate_artifact_chain(tfidf, svm)

    print("\n" + "=" * 70)
    print("TF-IDF FROM EXPERIMENT 3")
    print("=" * 70)
    for key, value in tfidf.items():
        print(f"{key:16}: {value}")
    print(f"Source           : {EXP3_PATH}")

    print("\n" + "=" * 70)
    print("SVM FROM EXPERIMENT 4")
    print("=" * 70)
    for key, value in svm.items():
        print(f"{key:16}: {value}")
    print(f"Source           : {EXP4_PATH}")

    print("\n[ARTIFACT CHAIN]")
    print("TF-IDF source    : Experiment 3 artifact")
    print("SVM source       : Experiment 4 artifact")
    print("final_config.py  : NOT USED")

    if CV_FOLDS != 5:
        raise ValueError(
            f"Experiment 5 membutuhkan 5-fold CV. Actual: {CV_FOLDS}"
        )

    # TAHAP 5: CV berkelompok (grup = processed_text).
    cv = make_grouped_cv(CV_FOLDS)

    results: list[dict[str, Any]] = []

    print("\n" + "=" * 70)
    print("CLASS IMBALANCE METHODS")
    print("=" * 70)

    for index, method in enumerate(METHODS, 1):
        print(f"\n[{index}/{len(METHODS)}] {method}")

        result = evaluate_method(
            X_train=X_train,
            y_train=y_train,
            tfidf=tfidf,
            svm=svm,
            method=method,
            cv=cv,
            groups_train=groups_train,
        )

        results.append(result)

        print(
            f"CV Macro F1: {result['cv_macro_f1_mean']:.4f} "
            f"+/- {result['cv_macro_f1_std']:.4f}"
        )

    ranked = sorted(
        results,
        key=lambda result: (
            -result["cv_macro_f1_mean"],
            result["cv_macro_f1_std"],
        ),
    )

    winner = ranked[0]

    print("\n" + "=" * 80)
    print("EXPERIMENT 5 RESULT")
    print("=" * 80)

    for rank, result in enumerate(ranked, 1):
        print(
            f"{rank}. {result['method']:24} "
            f"F1={result['cv_macro_f1_mean']:.4f} "
            f"STD={result['cv_macro_f1_std']:.4f}"
        )

    print("\n[LOCKED IMBALANCE]")
    print(f"Method : {winner['method']}")
    print(f"CV F1  : {winner['cv_macro_f1_mean']:.4f}")
    print(f"CV STD : {winner['cv_macro_f1_std']:.4f}")

    payload = {
        "experiment": "experiment_5_class_imbalance",
        "version": "artifact_chain_exp3_exp4",
        "dataset": {
            "rows_total": int(len(df)),
            "rows_development_train": int(len(X_train)),
            "rows_holdout": int(len(X_holdout)),
            "text_column": text_column,
            "label_column": label_column,
        },
        "fixed_split": {
            "train_size": int(len(X_train)),
            "holdout_size": int(len(X_holdout)),
            "train_ratio": 0.8,
            "holdout_ratio": 0.2,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "holdout_used_for_selection": False,
        },
        "methodology": {
            "cv": "StratifiedGroupKFold (grup = processed_text)",
            "cv_folds": CV_FOLDS,
            "random_state": RANDOM_STATE,
            "metric": "f1_macro",
            "selection_rule": (
                "highest CV Macro F1, then lower CV standard deviation"
            ),
            "holdout_used_for_selection": False,
            "tfidf_inside_cv_pipeline": True,
            "resampling_inside_cv_pipeline": True,
            "tfidf_parameters_source": str(
                EXP3_PATH.relative_to(PROJECT_ROOT)
            ).replace("\\", "/"),
            "svm_parameters_source": str(
                EXP4_PATH.relative_to(PROJECT_ROOT)
            ).replace("\\", "/"),
            "final_config_dependency": False,
        },
        "artifact_chain": {
            "experiment_3_source": str(
                EXP3_PATH.relative_to(PROJECT_ROOT)
            ).replace("\\", "/"),
            "experiment_4_source": str(
                EXP4_PATH.relative_to(PROJECT_ROOT)
            ).replace("\\", "/"),
            "experiment_3_tfidf": tfidf,
            "experiment_4_svm": svm,
            "artifact_validation": "PASS",
        },
        "results": results,
        "ranking": [
            {
                "rank": rank,
                "method": result["method"],
                "cv_macro_f1_mean": result["cv_macro_f1_mean"],
                "cv_macro_f1_std": result["cv_macro_f1_std"],
            }
            for rank, result in enumerate(ranked, 1)
        ],
        "best": winner,
        "locked_imbalance": {
            "method": winner["method"],
            "cv_macro_f1_mean": winner["cv_macro_f1_mean"],
            "cv_macro_f1_std": winner["cv_macro_f1_std"],
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(payload, RESULTS_PATH)

    print(f"\n[OUTPUT] {RESULTS_PATH}")
    print("[DONE] Experiment 5 selesai.")
    print("[INFO] Holdout tidak disentuh untuk seleksi.")

    return payload


if __name__ == "__main__":
    main()
