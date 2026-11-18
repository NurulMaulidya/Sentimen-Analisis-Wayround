"""
Learning Curve Analysis - Final Locked Model

Tujuan:
- Menguji pengaruh ukuran data training terhadap performa model.
- Membandingkan Train Macro F1 dan 5-Fold CV Macro F1.
- Menggunakan konfigurasi FINAL yang sudah dikunci dari eksperimen 3-6.
- Tidak menggunakan holdout set untuk pemilihan/evaluasi learning curve.
- Semua ukuran training merupakan subset dari DEVELOPMENT SET 80%.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from shared.constants import (  # noqa: E402
    EXPERIMENTS_DIR,
    CANONICAL_DATASET_PATH,
)
DATASET_PATH = CANONICAL_DATASET_PATH
OUTPUT_PATH = EXPERIMENTS_DIR / "learning_curve_final.json"
TEXT_COLUMN = "processed_text"
LABEL_COLUMN = "sentiment_label_final"
EXPECTED_CLASSES = ["negatif", "netral", "positif"]
RANDOM_STATE = 42
CV_FOLDS = 5
HOLDOUT_RATIO = 0.20
TRAIN_SIZES = [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]

# FINAL LOCKED CONFIGURATION
#
# TAHAP 5: nilai di bawah sebelumnya ditulis manual di skrip ini dan sudah
# basi (sublinear_tf=False, min_df=2, multiclass "OvO"). Sekarang dibaca
# dari shared/final_config.py - satu-satunya sumber konfigurasi final.
from shared.final_config import FINAL_MODEL_CONFIG as _LOCKED_CONFIG

TFIDF_CONFIG = {
    "max_features": _LOCKED_CONFIG["tfidf"]["max_features"],
    "ngram_range": tuple(_LOCKED_CONFIG["tfidf"]["ngram_range"]),
    "sublinear_tf": bool(_LOCKED_CONFIG["tfidf"]["sublinear_tf"]),
    "min_df": _LOCKED_CONFIG["tfidf"]["min_df"],
    "max_df": _LOCKED_CONFIG["tfidf"]["max_df"],
}
SVM_CONFIG = {
    "kernel": _LOCKED_CONFIG["svm"]["kernel"],
    "C": _LOCKED_CONFIG["svm"]["C"],
    "gamma": _LOCKED_CONFIG["svm"]["gamma"],
    "class_weight": _LOCKED_CONFIG["svm"].get("class_weight"),
}
MULTICLASS_STRATEGY = _LOCKED_CONFIG["multiclass"]["strategy"].upper()
IMBALANCE_METHOD = _LOCKED_CONFIG["imbalance"]["method"]


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        header = f.readline()
    return "," if header.count(",") >= header.count(";") else ";"


def load_dataset() -> tuple[pd.Series, pd.Series]:
    delimiter = detect_delimiter(DATASET_PATH)
    df = pd.read_csv(DATASET_PATH, sep=delimiter, encoding="utf-8-sig")
    required = [TEXT_COLUMN, LABEL_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")
    df = df[required].copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str).str.strip()
    df[LABEL_COLUMN] = (
        df[LABEL_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    df = df[
        (df[TEXT_COLUMN] != "")
        & (df[LABEL_COLUMN].isin(EXPECTED_CLASSES))
    ].reset_index(drop=True)
    if df.empty:
        raise ValueError("Dataset kosong setelah validasi.")
    return df[TEXT_COLUMN], df[LABEL_COLUMN]


def build_pipeline() -> Pipeline:
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=TFIDF_CONFIG["max_features"],
                ngram_range=TFIDF_CONFIG["ngram_range"],
                sublinear_tf=TFIDF_CONFIG["sublinear_tf"],
                min_df=TFIDF_CONFIG["min_df"],
                max_df=TFIDF_CONFIG["max_df"],
            ),
        ),
        (
            "svm",
            SVC(
                kernel=SVM_CONFIG["kernel"],
                C=SVM_CONFIG["C"],
                gamma=SVM_CONFIG["gamma"],
                class_weight=SVM_CONFIG["class_weight"],
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def stratified_subset(
    X_dev: pd.Series,
    y_dev: pd.Series,
    ratio: float,
) -> tuple[pd.Series, pd.Series]:
    if ratio >= 1.0:
        return X_dev.copy(), y_dev.copy()
    X_subset, _, y_subset, _ = train_test_split(
        X_dev,
        y_dev,
        train_size=ratio,
        random_state=RANDOM_STATE,
        stratify=y_dev,
    )
    return X_subset.reset_index(drop=True), y_subset.reset_index(drop=True)


def main() -> list[dict]:
    print("=" * 70)
    print("LEARNING CURVE ANALYSIS - FINAL LOCKED MODEL")
    print("=" * 70)

    print("\n[FINAL CONFIGURATION]")
    print(
        f"  TF-IDF: max_features={TFIDF_CONFIG['max_features']}, "
        f"ngram={TFIDF_CONFIG['ngram_range']}, "
        f"sublinear={TFIDF_CONFIG['sublinear_tf']}, "
        f"min_df={TFIDF_CONFIG['min_df']}, "
        f"max_df={TFIDF_CONFIG['max_df']}"
    )
    print(
        f"  SVM: kernel={SVM_CONFIG['kernel']}, "
        f"C={SVM_CONFIG['C']}, "
        f"gamma={SVM_CONFIG['gamma']}, "
        f"class_weight={SVM_CONFIG['class_weight']}"
    )
    print(f"  Multiclass: {MULTICLASS_STRATEGY}")
    print(f"  Imbalance: {IMBALANCE_METHOD}")
    print(f"  Random state: {RANDOM_STATE}")
    print(f"  CV folds: {CV_FOLDS}")

    X, y = load_dataset()
    print(f"\n[DATASET] {len(X)} samples")

    # FIXED 80:20 SPLIT.
    # Holdout is isolated and never used by the learning curve.
    X_dev, X_holdout, y_dev, y_holdout = train_test_split(
        X,
        y,
        test_size=HOLDOUT_RATIO,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_dev = X_dev.reset_index(drop=True)
    y_dev = y_dev.reset_index(drop=True)
    X_holdout = X_holdout.reset_index(drop=True)
    y_holdout = y_holdout.reset_index(drop=True)

    print(f"[DEVELOPMENT] {len(X_dev)} samples")
    print(f"[HOLDOUT] {len(X_holdout)} samples (NOT USED)")
    print("\n[DEVELOPMENT LABEL DISTRIBUTION]")
    print(y_dev.value_counts().reindex(EXPECTED_CLASSES, fill_value=0).to_string())

    print("\n[TRAINING SIZES]")
    for ratio in TRAIN_SIZES:
        print(f"  {int(ratio * 100)}% of development = {int(round(len(X_dev) * ratio))} samples")

    print("\n" + "-" * 70)

    results = []

    for ratio in TRAIN_SIZES:
        X_subset, y_subset = stratified_subset(X_dev, y_dev, ratio)
        train_size = len(X_subset)

        print(f"\n[Testing {int(ratio * 100)}% of DEVELOPMENT data]")
        print(f"  Training samples: {train_size}")

        pipeline = build_pipeline()
        pipeline.fit(X_subset, y_subset)

        y_train_pred = pipeline.predict(X_subset)
        train_f1 = f1_score(
            y_subset,
            y_train_pred,
            labels=EXPECTED_CLASSES,
            average="macro",
            zero_division=0,
        )

        class_counts = y_subset.value_counts()
        min_class_count = int(class_counts.min())

        if min_class_count < CV_FOLDS:
            raise ValueError(
                f"Data pada ukuran {int(ratio * 100)}% tidak cukup untuk "
                f"{CV_FOLDS}-fold stratified CV. Minimum class count={min_class_count}."
            )

        cv = StratifiedKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        cv_scores = cross_val_score(
            build_pipeline(),
            X_subset,
            y_subset,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
        )

        cv_f1_mean = float(cv_scores.mean())
        cv_f1_std = float(cv_scores.std())
        gap = float(train_f1 - cv_f1_mean)

        print(f"  Train Macro F1: {train_f1:.4f}")
        print(f"  CV Macro F1:    {cv_f1_mean:.4f} +/- {cv_f1_std:.4f}")
        print(f"  Train-CV Gap:   {gap:.4f} ({gap * 100:.2f} pp)")

        results.append({
            "train_ratio_of_development": float(ratio),
            "train_percentage_of_development": int(ratio * 100),
            "train_samples": int(train_size),
            "train_f1_macro": train_f1,
            "cv_f1_mean": cv_f1_mean,
            "cv_f1_std": cv_f1_std,
            "train_cv_gap": gap,
            "train_cv_gap_pp": gap * 100,
            "min_class_count": min_class_count,
            "cv_folds": CV_FOLDS,
        })

    print("\n" + "=" * 70)
    print("LEARNING CURVE RESULTS")
    print("=" * 70)
    print(
        f"{'Dev %':>7} | {'Samples':>8} | {'Train F1':>10} | "
        f"{'CV F1':>12} | {'Gap pp':>8}"
    )
    print("-" * 70)

    for r in results:
        print(
            f"{r['train_percentage_of_development']:>6}% | "
            f"{r['train_samples']:>8} | "
            f"{r['train_f1_macro']:>10.4f} | "
            f"{r['cv_f1_mean']:.4f} +/- {r['cv_f1_std']:.4f} | "
            f"{r['train_cv_gap_pp']:>7.2f}"
        )

    # Analysis
    print("\n[ANALYSIS]")

    final_result = results[-1]
    final_gap_pp = final_result["train_cv_gap_pp"]
    final_cv = final_result["cv_f1_mean"]

    if final_gap_pp > 10:
        print(
            f"  [WARN] Train-CV gap pada 100% development = "
            f"{final_gap_pp:.2f} pp."
        )
        print(
            "         Ini adalah generalization gap diagnostik, "
            "bukan persentase 'overfitting'."
        )
    elif final_gap_pp > 5:
        print(
            f"  [INFO] Train-CV gap pada 100% development = "
            f"{final_gap_pp:.2f} pp."
        )
    else:
        print(
            f"  [OK] Train-CV gap pada 100% development = "
            f"{final_gap_pp:.2f} pp."
        )

    if final_cv >= 0.70:
        print(f"  [OK] CV Macro F1 akhir = {final_cv:.4f} ({final_cv * 100:.2f}%).")
    else:
        print(f"  [WARN] CV Macro F1 akhir = {final_cv:.4f} (< 70%).")

    # Compare 60% vs 100% of DEVELOPMENT data.
    result_60 = next(
        r for r in results
        if r["train_percentage_of_development"] == 60
    )
    improvement_60_to_100 = final_cv - result_60["cv_f1_mean"]

    print(
        f"  [INFO] CV F1 60% -> 100% development: "
        f"{result_60['cv_f1_mean']:.4f} -> {final_cv:.4f}"
    )
    print(
        f"         Perubahan: {improvement_60_to_100 * 100:+.2f} "
        f"percentage points."
    )

    # Simple learning-curve interpretation.
    cv_values = [r["cv_f1_mean"] for r in results]
    if cv_values[-1] > cv_values[-2]:
        print(
            "  [INFO] Performa CV masih meningkat pada ukuran data terbesar."
        )
    else:
        print(
            "  [INFO] Performa CV tidak meningkat pada ukuran data terbesar; "
            "tambahan data belum menunjukkan peningkatan yang konsisten."
        )

    output_data = {
        "experiment": "learning_curve_final",
        "description": (
            "Learning curve menggunakan final locked model. "
            "Seluruh ukuran training merupakan subset dari development set "
            "80%; holdout 20% diisolasi dan tidak digunakan."
        ),
        "dataset": {
            "total_samples": int(len(X)),
            "development_samples": int(len(X_dev)),
            "holdout_samples": int(len(X_holdout)),
            "holdout_used": False,
        },
        "configuration": {
            "tfidf": {
                "max_features": TFIDF_CONFIG["max_features"],
                "ngram_range": list(TFIDF_CONFIG["ngram_range"]),
                "sublinear_tf": TFIDF_CONFIG["sublinear_tf"],
                "min_df": TFIDF_CONFIG["min_df"],
                "max_df": TFIDF_CONFIG["max_df"],
            },
            "svm": SVM_CONFIG,
            "multiclass_strategy": MULTICLASS_STRATEGY,
            "imbalance_method": IMBALANCE_METHOD,
            "random_state": RANDOM_STATE,
            "cv_folds": CV_FOLDS,
            "holdout_ratio": HOLDOUT_RATIO,
        },
        "train_sizes": TRAIN_SIZES,
        "results": results,
        "final_analysis": {
            "final_train_f1_macro": final_result["train_f1_macro"],
            "final_cv_f1_macro": final_result["cv_f1_mean"],
            "final_cv_std": final_result["cv_f1_std"],
            "final_train_cv_gap_pp": final_gap_pp,
            "cv_improvement_60_to_100_pp": improvement_60_to_100 * 100,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n[Saved] {OUTPUT_PATH}")
    return results


if __name__ == "__main__":
    main()
