"""
Experiment 6 - One-vs-One vs One-vs-Rest
FINAL / REPRODUCIBLE VERSION

Experiment 6 compares OvO vs OvR while inheriting the selected configuration
from Experiment 3 (TF-IDF), Experiment 4 (SVM), and Experiment 5 (class imbalance).

IMPORTANT:
- Do NOT import shared.final_config.py before all experiments are complete.
- Experiment 3/4/5 artifacts are the source of truth for this experiment.
- No TF-IDF, SVM, or imbalance parameter is hardcoded as a "final" value.
- Holdout 20% is never used for model selection.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from shared.constants import (  # noqa: E402
    EXPERIMENTS_DIR,
    CANONICAL_DATASET_PATH,
)

try:
    from scripts.experiments.common import (
        PROJECT_ROOT as COMMON_PROJECT_ROOT,
        RANDOM_STATE,
        CV_FOLDS,
        load_canonical_dataset,
        make_fixed_split,
        make_grouped_cv,
        save_json,
    )
    PROJECT_ROOT = COMMON_PROJECT_ROOT
except ImportError:
    from .common import (
        PROJECT_ROOT as COMMON_PROJECT_ROOT,
        RANDOM_STATE,
        CV_FOLDS,
        load_canonical_dataset,
        make_fixed_split,
        make_grouped_cv,
        save_json,
    )
    PROJECT_ROOT = COMMON_PROJECT_ROOT

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False


OUTPUT_DIR = EXPERIMENTS_DIR
EXP3_PATH = OUTPUT_DIR / "experiment_3_tfidf_results.json"
EXP4_PATH = OUTPUT_DIR / "experiment_4_svm_results.json"
EXP5_PATH = OUTPUT_DIR / "experiment_5_class_imbalance_results.json"

JSON_PATH = OUTPUT_DIR / "experiment_6_ovo_ovr_results.json"
CSV_PATH = OUTPUT_DIR / "experiment_6_ovo_ovr_results.csv"

CLASSES = ["negatif", "netral", "positif"]


def load_json(path: Path, name: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{name} tidak ditemukan: {path}\n"
            "Jalankan eksperimen sebelumnya terlebih dahulu."
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{name} harus berupa object JSON.")
    return data


def normalize_ngram(value):
    if isinstance(value, (list, tuple)):
        return tuple(int(x) for x in value)
    raise ValueError(f"ngram_range tidak valid: {value!r}")


def extract_tfidf(exp3: dict) -> dict:
    """
    Extract TF-IDF selected by Experiment 3.

    Supports the artifact structures used by previous versions, but never
    falls back to a hardcoded TF-IDF configuration.
    """
    candidates = []

    for key in ("locked_tfidf", "best_configuration", "best", "locked"):
        value = exp3.get(key)
        if isinstance(value, dict):
            candidates.append(value)
            for nested_key in (
                "tfidf",
                "best_tfidf",
                "configuration",
                "best_configuration",
            ):
                nested = value.get(nested_key)
                if isinstance(nested, dict):
                    candidates.append(nested)

    # Also support an explicit artifact_chain.tfidf structure.
    chain = exp3.get("artifact_chain")
    if isinstance(chain, dict) and isinstance(chain.get("tfidf"), dict):
        candidates.append(chain["tfidf"])

    required = {
        "max_features",
        "ngram_range",
        "sublinear_tf",
        "min_df",
        "max_df",
    }

    for c in candidates:
        if required.issubset(c):
            return {
                "max_features": int(c["max_features"]),
                "ngram_range": normalize_ngram(c["ngram_range"]),
                "sublinear_tf": bool(c["sublinear_tf"]),
                "min_df": c["min_df"],
                "max_df": c["max_df"],
            }

    raise ValueError(
        "TF-IDF hasil Experiment 3 tidak ditemukan atau tidak lengkap. "
        "Periksa experiment_3_tfidf_results_v4.json."
    )


def extract_svm(exp4: dict) -> dict:
    """
    Extract SVM selected by Experiment 4.

    No SVM value is hardcoded as the final configuration.
    """
    candidates = []

    for key in ("locked_svm", "best_configuration", "best", "locked"):
        value = exp4.get(key)
        if isinstance(value, dict):
            candidates.append(value)
            for nested_key in (
                "svm",
                "best_svm",
                "configuration",
                "best_configuration",
            ):
                nested = value.get(nested_key)
                if isinstance(nested, dict):
                    candidates.append(nested)

    chain = exp4.get("artifact_chain")
    if isinstance(chain, dict) and isinstance(chain.get("svm"), dict):
        candidates.append(chain["svm"])

    for c in candidates:
        if "kernel" in c and "C" in c:
            return {
                "kernel": str(c["kernel"]).lower(),
                "C": float(c["C"]),
                "gamma": c.get("gamma", "scale"),
            }

    raise ValueError(
        "SVM hasil Experiment 4 tidak ditemukan atau tidak lengkap. "
        "Periksa experiment_4_svm_results_v3.json."
    )


def extract_imbalance(exp5: dict) -> dict:
    """
    Extract the selected imbalance method from Experiment 5.
    """
    candidates = []

    for key in ("locked_imbalance", "best", "best_configuration", "locked"):
        value = exp5.get(key)
        if isinstance(value, dict):
            candidates.append(value)

    chain = exp5.get("artifact_chain")
    if isinstance(chain, dict) and isinstance(chain.get("imbalance"), dict):
        candidates.append(chain["imbalance"])

    for c in candidates:
        if "method" in c:
            result = {
                "method": str(c["method"]).lower(),
                "cv_macro_f1_mean": c.get("cv_macro_f1_mean"),
                "cv_macro_f1_std": c.get("cv_macro_f1_std"),
            }
            if result["cv_macro_f1_mean"] is not None:
                result["cv_macro_f1_mean"] = float(result["cv_macro_f1_mean"])
            if result["cv_macro_f1_std"] is not None:
                result["cv_macro_f1_std"] = float(result["cv_macro_f1_std"])
            return result

    raise ValueError(
        "Metode class imbalance hasil Experiment 5 tidak ditemukan atau tidak lengkap. "
        "Periksa experiment_5_class_imbalance_results.json."
    )


def validate_artifact_chain(tfidf: dict, svm: dict, imbalance: dict):
    """
    Validate completeness only.

    Deliberately does NOT compare against hardcoded expected values.
    The previous experiment artifacts are the source of truth.
    """
    errors = []

    tfidf_required = {
        "max_features",
        "ngram_range",
        "sublinear_tf",
        "min_df",
        "max_df",
    }
    missing_tfidf = tfidf_required - set(tfidf)
    if missing_tfidf:
        errors.append(f"TF-IDF missing keys: {sorted(missing_tfidf)}")

    for key in ("kernel", "C", "gamma"):
        if key not in svm:
            errors.append(f"SVM missing key: {key}")

    method = imbalance.get("method")
    valid_methods = {
        "none",
        "class_weight_balanced",
        "random_over",
        "random_under",
        "smote",
    }
    if method not in valid_methods:
        errors.append(
            f"Imbalance method tidak dikenali: {method!r}. "
            f"Valid: {sorted(valid_methods)}"
        )

    if errors:
        raise RuntimeError(
            "[ARTIFACT CHAIN ERROR]\n"
            + "\n".join(f"  - {item}" for item in errors)
        )


def build_pipeline(tfidf: dict, svm: dict, imbalance_method: str, strategy: str):
    """
    Build the exact Experiment 6 pipeline from prior experiment artifacts.

    TF-IDF comes from Exp3.
    SVM hyperparameters come from Exp4.
    Imbalance method comes from Exp5.
    Only multiclass strategy changes between OvO and OvR.
    """
    vectorizer = TfidfVectorizer(
        max_features=tfidf["max_features"],
        ngram_range=tfidf["ngram_range"],
        sublinear_tf=tfidf["sublinear_tf"],
        min_df=tfidf["min_df"],
        max_df=tfidf["max_df"],
    )

    class_weight = (
        "balanced" if imbalance_method == "class_weight_balanced" else None
    )

    base_svm = SVC(
        kernel=svm["kernel"],
        C=svm["C"],
        gamma=svm["gamma"],
        class_weight=class_weight,
        probability=False,
    )

    if strategy == "ovo":
        classifier = OneVsOneClassifier(base_svm)
    elif strategy == "ovr":
        classifier = OneVsRestClassifier(base_svm)
    else:
        raise ValueError("Strategy harus 'ovo' atau 'ovr'.")

    steps = [("tfidf", vectorizer)]

    # Class-weight balancing is a parameter of SVC, so no sampler is needed.
    if imbalance_method == "none" or imbalance_method == "class_weight_balanced":
        steps.append(("classifier", classifier))
        return SklearnPipeline(steps)

    if not IMBLEARN_AVAILABLE:
        raise ImportError(
            "imbalanced-learn diperlukan karena Experiment 5 memilih "
            f"metode '{imbalance_method}'. Install dengan: "
            "pip install imbalanced-learn"
        )

    if imbalance_method == "random_over":
        sampler = RandomOverSampler(random_state=RANDOM_STATE)
    elif imbalance_method == "random_under":
        sampler = RandomUnderSampler(random_state=RANDOM_STATE)
    elif imbalance_method == "smote":
        sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    else:
        raise ValueError(f"Metode imbalance tidak didukung: {imbalance_method}")

    return ImbPipeline(
        steps=[
            ("tfidf", vectorizer),
            ("sampler", sampler),
            ("classifier", classifier),
        ]
    )


def evaluate_strategy(X, y, tfidf, svm, imbalance_method, strategy, groups=None):
    # TAHAP 5: CV berkelompok (grup = processed_text).
    cv = make_grouped_cv(CV_FOLDS)

    folds = []
    oof_true = []
    oof_pred = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), 1):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        model = build_pipeline(
            tfidf=tfidf,
            svm=svm,
            imbalance_method=imbalance_method,
            strategy=strategy,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_val)

        result = {
            "fold": fold,
            "accuracy": float(accuracy_score(y_val, pred)),
            "precision_macro": float(
                precision_score(
                    y_val, pred, labels=CLASSES, average="macro", zero_division=0
                )
            ),
            "recall_macro": float(
                recall_score(
                    y_val, pred, labels=CLASSES, average="macro", zero_division=0
                )
            ),
            "f1_macro": float(
                f1_score(
                    y_val, pred, labels=CLASSES, average="macro", zero_division=0
                )
            ),
            "f1_weighted": float(
                f1_score(
                    y_val, pred, labels=CLASSES, average="weighted", zero_division=0
                )
            ),
        }

        folds.append(result)
        oof_true.extend(y_val.tolist())
        oof_pred.extend(pred.tolist())

        print(
            f"  Fold {fold}: Accuracy={result['accuracy']:.4f} | "
            f"Macro F1={result['f1_macro']:.4f}"
        )

    fold_df = pd.DataFrame(folds)
    cm = confusion_matrix(oof_true, oof_pred, labels=CLASSES)
    report = classification_report(
        oof_true,
        oof_pred,
        labels=CLASSES,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    return {
        "strategy": strategy,
        "cv_folds": CV_FOLDS,
        "fold_results": folds,
        "mean_accuracy": float(fold_df["accuracy"].mean()),
        "std_accuracy": float(fold_df["accuracy"].std(ddof=1)),
        "mean_precision_macro": float(fold_df["precision_macro"].mean()),
        "mean_recall_macro": float(fold_df["recall_macro"].mean()),
        "mean_f1_macro": float(fold_df["f1_macro"].mean()),
        "std_f1_macro": float(fold_df["f1_macro"].std(ddof=1)),
        "mean_f1_weighted": float(fold_df["f1_weighted"].mean()),
        "oof_confusion_matrix": cm.tolist(),
        "oof_classification_report": report,
    }


def main():
    print("=" * 80)
    print("EXPERIMENT 6 - OvO vs OvR - ARTIFACT CHAIN")
    print("=" * 80)

    # Load the canonical dataset only. No final_config dependency.
    df, text_column, label_column = load_canonical_dataset()

    X_train, X_holdout, y_train, y_holdout, groups_train = make_fixed_split(
        df,
        text_column=text_column,
        label_column=label_column,
        return_groups=True,
    )

    # Load previous experiment artifacts.
    exp3 = load_json(EXP3_PATH, "Experiment 3")
    exp4 = load_json(EXP4_PATH, "Experiment 4")
    exp5 = load_json(EXP5_PATH, "Experiment 5")

    tfidf = extract_tfidf(exp3)
    svm = extract_svm(exp4)
    imbalance = extract_imbalance(exp5)

    validate_artifact_chain(tfidf, svm, imbalance)

    print("\n" + "=" * 70)
    print("LOCKED ARTIFACT CHAIN FROM PREVIOUS EXPERIMENTS")
    print("=" * 70)
    print(f"Exp3 TF-IDF : {tfidf}")
    print(f"Exp3 source : {EXP3_PATH}")
    print("Artifact check: PASS")

    print(f"\nExp4 SVM    : {svm}")
    print(f"Exp4 source : {EXP4_PATH}")
    print("Artifact check: PASS")

    print(f"\nExp5 method : {imbalance['method']}")
    print(f"Exp5 source : {EXP5_PATH}")
    print("Artifact check: PASS")

    print("\n[EXPERIMENT 6 SCOPE]")
    print("Only multiclass strategy changes: OvO vs OvR")
    print("TF-IDF, SVM hyperparameters, imbalance method, split, and CV remain fixed.")

    print("\n[DATA]")
    print(f"Rows total      : {len(df)}")
    print(f"Development     : {len(X_train)}")
    print(f"Holdout         : {len(X_holdout)}")
    print("Holdout used for selection: NO")

    results = []

    for strategy in ("ovo", "ovr"):
        print("\n" + "-" * 80)
        print(f"RUNNING {strategy.upper()}")
        print("-" * 80)

        result = evaluate_strategy(
            X_train,
            y_train,
            tfidf,
            svm,
            imbalance["method"],
            strategy,
            groups=groups_train,
        )
        results.append(result)

        print(
            f"{strategy.upper()} Mean Macro F1: "
            f"{result['mean_f1_macro']:.4f} +/- {result['std_f1_macro']:.4f}"
        )

    ranked = sorted(
        results,
        key=lambda r: (-r["mean_f1_macro"], r["std_f1_macro"]),
    )
    winner = ranked[0]

    print("\n" + "=" * 80)
    print("EXPERIMENT 6 RESULT")
    print("=" * 80)

    for rank, result in enumerate(ranked, 1):
        print(
            f"{rank}. {result['strategy'].upper():4} "
            f"Macro F1={result['mean_f1_macro']:.4f} "
            f"STD={result['std_f1_macro']:.4f}"
        )

    print(f"\n[SELECTED MULTICLASS] {winner['strategy'].upper()}")
    print(f"CV Macro F1 : {winner['mean_f1_macro']:.4f}")
    print(f"CV STD      : {winner['std_f1_macro']:.4f}")

    summary = {
        "experiment": "experiment_6_ovo_ovr",
        "version": "FINAL",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "configuration_source": {
            "final_config_dependency": False,
            "tfidf_source": str(EXP3_PATH),
            "svm_source": str(EXP4_PATH),
            "imbalance_source": str(EXP5_PATH),
            "final_config_is_locked_after_all_experiments": True,
        },
        "dataset": {
            "path": str(
                CANONICAL_DATASET_PATH
            ),
            "rows_total": int(len(df)),
            "rows_development_train": int(len(X_train)),
            "rows_holdout": int(len(X_holdout)),
            "text_column": text_column,
            "label_column": label_column,
            "classes": CLASSES,
            "class_distribution": {
                str(k): int(v) for k, v in df[label_column].value_counts().items()
            },
        },
        "methodology": {
            "cv": "StratifiedGroupKFold (grup = processed_text)",
            "cv_folds": CV_FOLDS,
            "random_state": RANDOM_STATE,
            "selection_metric": "mean_cv_f1_macro",
            "selection_tiebreaker": "lower_cv_f1_macro_std",
            "holdout_used_for_selection": False,
            "tfidf_inside_cv_pipeline": True,
            "imbalance_method": imbalance["method"],
            "scope": "Only One-vs-One vs One-vs-Rest",
            "multiclass_base_estimator": "SVC",
            "ovo_implementation": "OneVsOneClassifier(SVC)",
            "ovr_implementation": "OneVsRestClassifier(SVC)",
        },
        "artifact_chain": {
            "experiment_3_source": str(EXP3_PATH),
            "experiment_4_source": str(EXP4_PATH),
            "experiment_5_source": str(EXP5_PATH),
            "artifact_validation": "PASS",
            "tfidf": tfidf,
            "svm": svm,
            "imbalance": imbalance,
        },
        "strategies": results,
        "winner_by_cv_f1_macro": winner["strategy"],
        "locked_multiclass": {
            "strategy": winner["strategy"],
            "mean_f1_macro": winner["mean_f1_macro"],
            "std_f1_macro": winner["std_f1_macro"],
        },
        "holdout_note": (
            "CV digunakan untuk memilih strategi multiclass. "
            "Holdout tetap tidak disentuh dan wajib digunakan satu kali "
            "pada evaluasi final setelah seluruh konfigurasi terkunci."
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(summary, JSON_PATH)

    csv_rows = []
    for result in results:
        csv_rows.append(
            {
                "strategy": result["strategy"],
                "cv_folds": result["cv_folds"],
                "mean_accuracy": result["mean_accuracy"],
                "std_accuracy": result["std_accuracy"],
                "mean_precision_macro": result["mean_precision_macro"],
                "mean_recall_macro": result["mean_recall_macro"],
                "mean_f1_macro": result["mean_f1_macro"],
                "std_f1_macro": result["std_f1_macro"],
                "mean_f1_weighted": result["mean_f1_weighted"],
                "selected_winner": result["strategy"] == winner["strategy"],
            }
        )

    pd.DataFrame(csv_rows).to_csv(
        CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\n[OUTPUT JSON] {JSON_PATH}")
    print(f"[OUTPUT CSV ] {CSV_PATH}")
    print("\n[DONE] Experiment 6 selesai. Holdout tidak digunakan untuk seleksi.")


if __name__ == "__main__":
    main()
