"""
Out-of-Fold Generalization Analysis - Experiment Based

Konfigurasi TIDAK ditulis manual.
Parameter model diambil dari artefak Experiment 3, 4, 5, dan 6.

Holdout hanya dievaluasi setelah konfigurasi final terbentuk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from shared.constants import (  # noqa: E402
    EXPERIMENTS_DIR,
    CANONICAL_DATASET_PATH,
    OUTPUT_DIR,
)

from shared.experiment_config_loader import (  # noqa: E402
    EXPECTED_CLASSES,
    load_experiment_chain,
    print_chain,
)
from shared.splitting import (  # noqa: E402
    build_groups,
    split_parts_for_dataset,
)

DATASET_PATH = CANONICAL_DATASET_PATH

# Output diagnostik TIDAK ditulis ke output_v4/experiments/, supaya folder
# itu tetap hanya berisi 7 eksperimen canonical + artefak evaluasi final.
# Prinsip: "Jangan menimpa hasil eksperimen lama."
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
OUTPUT_PATH = DIAGNOSTICS_DIR / "oof_generalization_final.json"
OUTPUT_CSV_PATH = DIAGNOSTICS_DIR / "oof_generalization_final_summary.csv"

RANDOM_STATE = 42
CV_FOLDS = 5
TEST_SIZE = 0.20


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")
    required = ["processed_text", "sentiment_label_final"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    # reviewId dipakai untuk menyelaraskan baris dengan split canonical
    # Stage 5 (bukan split ulang per baris).
    keep = list(required)
    if "reviewId" in df.columns:
        keep.append("reviewId")

    df = df[keep].copy()
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
    strategy = final_config["multiclass"]

    vectorizer = TfidfVectorizer(
        max_features=tfidf["max_features"],
        ngram_range=tuple(tfidf["ngram_range"]),
        sublinear_tf=tfidf["sublinear_tf"],
        min_df=tfidf["min_df"],
        max_df=tfidf["max_df"],
    )

    base_svm = SVC(
        kernel=svm["kernel"],
        C=svm["C"],
        gamma=svm["gamma"],
        class_weight=svm["class_weight"],
        random_state=RANDOM_STATE,
    )

    if strategy == "ovo":
        classifier = OneVsOneClassifier(base_svm)
    elif strategy == "ovr":
        classifier = OneVsRestClassifier(base_svm)
    else:
        raise ValueError(f"Strategi multiclass tidak didukung: {strategy}")

    return Pipeline([
        ("tfidf", vectorizer),
        ("svm", classifier),
    ])


def macro_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=EXPECTED_CLASSES, average="macro", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(y_true, y_pred, labels=EXPECTED_CLASSES, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, labels=EXPECTED_CLASSES, average="macro", zero_division=0)
        ),
    }


def main():
    print("=" * 80)
    print("OUT-OF-FOLD GENERALIZATION ANALYSIS - EXPERIMENT BASED")
    print("=" * 80)

    chain = load_experiment_chain(include_exp7=True)
    print_chain(chain)

    final_config = chain["final"]
    df = load_dataset()

    print("\n[DATASET]")
    print(f"  Total rows : {len(df)}")
    print(f"  Distribution: {df['sentiment_label_final'].value_counts().to_dict()}")

    X = df["processed_text"]
    y = df["sentiment_label_final"]

    # Split CANONICAL (berkelompok, dari Stage 5), bukan train_test_split
    # per baris.
    #
    # TEMUAN LEAKAGE YANG DIPERBAIKI:
    # Versi lama memakai train_test_split() per baris di atas SELURUH
    # dataset. Akibatnya:
    #   * "holdout" pada skrip ini BUKAN holdout canonical Stage 5,
    #   * baris dengan processed_text identik bisa muncul di development
    #     DAN holdout script (kebocoran yang sengaja dihilangkan Tahap 3),
    #   * holdout canonical ikut terbawa sebagai bagian "development".
    development, holdout, split_info = split_parts_for_dataset(df)
    assert len(development) + len(holdout) == len(df), (
        "Split canonical tidak mencakup seluruh dataset canonical."
    )

    X_dev = development["processed_text"]
    y_dev = development["sentiment_label_final"]
    X_holdout = holdout["processed_text"]
    y_holdout = holdout["sentiment_label_final"]

    dev_groups = set(build_groups(development))
    hold_groups = set(build_groups(holdout))
    if dev_groups & hold_groups:
        raise RuntimeError(
            f"Split canonical bocor: {len(dev_groups & hold_groups)} grup "
            "processed_text muncul di development dan holdout."
        )

    print(f"  Split source  : {split_info.get('source')}")
    print(f"  Development : {len(X_dev)}")
    print(f"  Holdout     : {len(X_holdout)}")
    print("  Holdout     : NOT USED FOR SELECTION")

    # 1. Full-development training
    print("\n[1] FULL DEVELOPMENT TRAINING")
    full_pipeline = build_pipeline(final_config)
    full_pipeline.fit(X_dev, y_dev)
    y_dev_pred = full_pipeline.predict(X_dev)
    train_metrics = macro_metrics(y_dev, y_dev_pred)

    # 2. True OOF
    print("\n[2] TRUE OUT-OF-FOLD PREDICTIONS")

    # CV BERKELOMPOK: grup = processed_text.
    # StratifiedKFold per baris (versi lama) membiarkan teks identik muncul
    # di fold latih dan fold validasi, sehingga OOF terlalu optimistis.
    groups_dev = build_groups(development).to_numpy()

    cv = StratifiedGroupKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_predictions = np.empty(len(X_dev), dtype=object)
    fold_details = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        cv.split(X_dev, y_dev, groups=groups_dev), start=1
    ):
        model = build_pipeline(final_config)
        model.fit(X_dev.iloc[train_idx], y_dev.iloc[train_idx])
        pred = model.predict(X_dev.iloc[val_idx])
        oof_predictions[val_idx] = pred

        metrics = macro_metrics(y_dev.iloc[val_idx], pred)
        train_groups = set(groups_dev[train_idx])
        val_groups = set(groups_dev[val_idx])
        fold_details.append({
            "fold": fold_idx,
            "train_size": int(len(train_idx)),
            "validation_size": int(len(val_idx)),
            "group_overlap": int(len(train_groups & val_groups)),
            **metrics,
        })
        print(
            f"  Fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}, "
            f"F1={metrics['f1_macro']:.4f}, "
            f"grup tumpang tindih={len(train_groups & val_groups)}"
        )

    if np.any(pd.isna(oof_predictions)):
        raise RuntimeError("OOF prediction tidak lengkap.")

    oof_metrics = macro_metrics(y_dev, oof_predictions)
    fold_f1 = [x["f1_macro"] for x in fold_details]
    cv_f1_mean = float(np.mean(fold_f1))
    cv_f1_std = float(np.std(fold_f1, ddof=0))

    # 3. Holdout
    print("\n[3] FINAL HOLDOUT EVALUATION")
    y_holdout_pred = full_pipeline.predict(X_holdout)
    holdout_metrics = macro_metrics(y_holdout, y_holdout_pred)

    # 4. Gaps
    train_f1 = train_metrics["f1_macro"]
    oof_f1 = oof_metrics["f1_macro"]
    holdout_f1 = holdout_metrics["f1_macro"]

    gaps = {
        "train_oof": train_f1 - oof_f1,
        "train_holdout": train_f1 - holdout_f1,
        "cv_holdout": cv_f1_mean - holdout_f1,
        "oof_holdout": oof_f1 - holdout_f1,
    }

    print("\n[METRICS]")
    print(f"  Train Macro F1   : {train_f1:.4f}")
    print(f"  OOF Macro F1     : {oof_f1:.4f}")
    print(f"  CV Macro F1      : {cv_f1_mean:.4f} +/- {cv_f1_std:.4f}")
    print(f"  Holdout Macro F1 : {holdout_f1:.4f}")

    print("\n[GAPS]")
    for name, value in gaps.items():
        print(f"  {name:15s}: {value:+.4f} ({value * 100:+.2f} pp)")

    results = {
        "experiment": "oof_generalization_final",
        "configuration_source": {
            "type": "experiment_artifacts",
            "exp3": str(chain["exp3"]["path"]),
            "exp4": str(chain["exp4"]["path"]),
            "exp5": str(chain["exp5"]["path"]),
            "exp6": str(chain["exp6"]["path"]),
            "exp7": str(chain["exp7"]["path"]) if chain["exp7"] else None,
        },
        "configuration": final_config,
        "methodology": {
            "random_state": RANDOM_STATE,
            "cv_folds": CV_FOLDS,
            "holdout_ratio": TEST_SIZE,
            "holdout_used_for_selection": False,
            "oof_is_true_out_of_fold": True,
            "tfidf_fit_scope": "train_fold_only",
            "holdout_fit_scope": "full_development_only",
            "split_source": split_info.get("source"),
            "split_method": "canonical Stage 5 grouped split (grup = processed_text)",
            "cv_method": "StratifiedGroupKFold (grup = processed_text)",
            "group_overlap_development_holdout": 0,
            "group_overlap_total_cv_folds": int(
                sum(f.get("group_overlap", 0) for f in fold_details)
            ),
        },
        "dataset": {
            "total_rows": int(len(df)),
            "development_rows": int(len(X_dev)),
            "holdout_rows": int(len(X_holdout)),
            "class_distribution": df["sentiment_label_final"].value_counts().to_dict(),
        },
        "train": train_metrics,
        "oof": {**oof_metrics, "folds": fold_details},
        "cv": {
            "macro_f1_mean": cv_f1_mean,
            "macro_f1_std": cv_f1_std,
        },
        "holdout": holdout_metrics,
        "generalization_gaps": gaps,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = pd.DataFrame([
        {"metric": "Train Macro F1", "value": train_f1},
        {"metric": "OOF Macro F1", "value": oof_f1},
        {"metric": "CV Macro F1 Mean", "value": cv_f1_mean},
        {"metric": "CV Macro F1 Std", "value": cv_f1_std},
        {"metric": "Holdout Macro F1", "value": holdout_f1},
        {"metric": "Train-OOF Gap", "value": gaps["train_oof"]},
        {"metric": "Train-Holdout Gap", "value": gaps["train_holdout"]},
        {"metric": "CV-Holdout Gap", "value": gaps["cv_holdout"]},
        {"metric": "OOF-Holdout Gap", "value": gaps["oof_holdout"]},
    ])
    summary.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print("\n[SAVED]")
    print(f"  JSON : {OUTPUT_PATH}")
    print(f"  CSV  : {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
