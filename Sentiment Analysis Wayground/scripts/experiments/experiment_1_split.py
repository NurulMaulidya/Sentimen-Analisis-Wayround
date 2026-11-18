from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from shared.constants import EXPERIMENTS_DIR  # noqa: E402
from shared.splitting import split_parts_for_dataset, grouped_stratified_split, build_groups
from scripts.experiments.common import make_grouped_cv

try:
    from scripts.experiments.common import (
        CV_FOLDS,
        RANDOM_STATE,
        OUTPUT_DIR,
        detect_label_column,
        detect_text_column,
        load_dataset,
        normalize_labels,
        print_header,
        save_json,
    )
except ImportError:
    from common import (
        CV_FOLDS,
        RANDOM_STATE,
        OUTPUT_DIR,
        detect_label_column,
        detect_text_column,
        load_dataset,
        normalize_labels,
        print_header,
        save_json,
    )


# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------

SPLIT_RATIOS = (0.70, 0.80, 0.90)

OUTPUT_FILE = OUTPUT_DIR / "experiment_1_split_results.json"

# No TF-IDF/SVM values are hardcoded here.
# They must come from a previous artifact supplied through --previous-config.
DEFAULT_PREVIOUS_CONFIG = (
    PROJECT_ROOT / "scripts" / "experiments" / "baseline_config.json"
)


# ---------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------

def load_json(path: Path, name: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{name} tidak ditemukan:\n{path}\n\n"
            "Experiment 1 membutuhkan artifact konfigurasi sebelumnya. "
            "Gunakan --previous-config <path-json>."
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{name} harus berupa object JSON: {path}")

    return data


def normalize_ngram(value: Any) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])

    raise ValueError(f"ngram_range tidak valid: {value!r}")


def collect_candidates(data: dict, keys: tuple[str, ...]) -> list[dict]:
    candidates: list[dict] = []

    for key in keys:
        value = data.get(key)

        if isinstance(value, dict):
            candidates.append(value)

            for nested_key in (
                "configuration",
                "tfidf",
                "svm",
                "best_tfidf",
                "best_svm",
                "locked_tfidf",
                "locked_svm",
            ):
                nested = value.get(nested_key)

                if isinstance(nested, dict):
                    candidates.append(nested)

    return candidates


def extract_tfidf(config: dict) -> dict:
    required = {
        "max_features",
        "ngram_range",
        "sublinear_tf",
        "min_df",
        "max_df",
    }

    candidates = collect_candidates(
        config,
        (
            "tfidf",
            "locked_tfidf",
            "best_tfidf",
            "best_configuration",
            "configuration",
            "best",
        ),
    )

    # Also support a configuration object with nested "model".
    model = config.get("model")
    if isinstance(model, dict):
        candidates.extend(
            collect_candidates(
                model,
                ("tfidf", "configuration", "best"),
            )
        )

    for candidate in candidates:
        if required.issubset(candidate):
            return {
                "max_features": int(candidate["max_features"]),
                "ngram_range": normalize_ngram(candidate["ngram_range"]),
                "sublinear_tf": bool(candidate["sublinear_tf"]),
                "min_df": candidate["min_df"],
                "max_df": candidate["max_df"],
            }

    raise ValueError(
        "Parameter TF-IDF tidak ditemukan di artifact sebelumnya.\n"
        "Required: max_features, ngram_range, sublinear_tf, min_df, max_df."
    )


def extract_svm(config: dict) -> dict:
    candidates = collect_candidates(
        config,
        (
            "svm",
            "locked_svm",
            "best_svm",
            "best_configuration",
            "configuration",
            "best",
        ),
    )

    model = config.get("model")
    if isinstance(model, dict):
        candidates.extend(
            collect_candidates(
                model,
                ("svm", "configuration", "best"),
            )
        )

    for candidate in candidates:
        if "kernel" in candidate and "C" in candidate:
            return {
                "kernel": str(candidate["kernel"]).lower(),
                "C": float(candidate["C"]),
                "gamma": candidate.get("gamma", "scale"),
                "class_weight": candidate.get("class_weight", None),
            }

    raise ValueError(
        "Parameter SVM tidak ditemukan di artifact sebelumnya.\n"
        "Required: kernel dan C."
    )


def validate_previous_config(tfidf: dict, svm: dict) -> None:
    if tfidf["max_features"] <= 0:
        raise ValueError("TF-IDF max_features harus > 0.")

    if tfidf["min_df"] <= 0:
        raise ValueError("TF-IDF min_df harus > 0.")

    if svm["kernel"] not in {"linear", "rbf", "poly", "sigmoid"}:
        raise ValueError(f"Kernel SVM tidak valid: {svm['kernel']}")

    if svm["C"] <= 0:
        raise ValueError(f"SVM C harus > 0: {svm['C']}")

    if svm.get("class_weight") not in (None, "balanced"):
        raise ValueError(
            f"class_weight SVM tidak didukung: {svm.get('class_weight')}"
        )


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

def build_pipeline(tfidf_cfg: dict, svm_cfg: dict) -> Pipeline:
    vectorizer = TfidfVectorizer(
        max_features=tfidf_cfg["max_features"],
        ngram_range=tfidf_cfg["ngram_range"],
        sublinear_tf=tfidf_cfg["sublinear_tf"],
        min_df=tfidf_cfg["min_df"],
        max_df=tfidf_cfg["max_df"],
    )

    classifier = SVC(
        kernel=svm_cfg["kernel"],
        C=svm_cfg["C"],
        gamma=svm_cfg.get("gamma", "scale"),
        class_weight=svm_cfg.get("class_weight"),
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        [
            ("tfidf", vectorizer),
            ("svm", classifier),
        ]
    )


# ---------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------

def run_split_experiment(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    tfidf_cfg: dict,
    svm_cfg: dict,
    previous_config_path: Path,
) -> dict:

    print_header("EXPERIMENT 1: SPLIT RATIO SENSITIVITY")

    working = df.copy()

    working[text_column] = (
        working[text_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    working[label_column] = normalize_labels(working[label_column])

    working = working[
        (working[text_column] != "")
        & working[label_column].notna()
    ].copy()

    if working.empty:
        raise ValueError("Dataset kosong setelah validasi.")

    # Ratio sensitivity is a diagnostic INSIDE development; canonical holdout
    # IDs must never become training/CV examples when the ratio is varied.
    working, canonical_holdout, _ = split_parts_for_dataset(working, label_col=label_column)
    X = working[text_column]
    y = working[label_column]

    class_counts = y.value_counts()

    if class_counts.min() < CV_FOLDS:
        raise ValueError(
            "Jumlah data kelas terkecil tidak cukup "
            f"untuk {CV_FOLDS}-Fold CV."
        )

    print(f"[DATA] Total valid : {len(working):,}")
    print("[DATA] Distribution:")

    for label, count in class_counts.items():
        print(f"       {label:<8}: {count:,}")

    print()
    print("[LOCKED FROM PREVIOUS ARTIFACT]")
    print(f"TF-IDF : {tfidf_cfg}")
    print(f"SVM    : {svm_cfg}")
    print(f"Source : {previous_config_path}")
    print()

    results = []

    for train_ratio in SPLIT_RATIOS:
        test_ratio = 1.0 - train_ratio

        print_header(
            f"SPLIT {train_ratio:.0%}:{test_ratio:.0%}"
        )

        n_samples = len(X)
        n_train = int(n_samples * train_ratio)
        n_unused = n_samples - n_train

        if n_train <= 0 or n_unused <= 0:
            raise ValueError(
                f"Split {train_ratio:.0%}:{test_ratio:.0%} tidak valid: "
                f"train={n_train}, holdout={n_unused}."
            )

        subset, unused, _ = grouped_stratified_split(working, label_col=label_column, test_size=test_ratio)
        X_train, y_train = subset[text_column], subset[label_column]
        X_unused, y_unused = unused[text_column], unused[label_column]

        print(f"[TRAIN] {len(X_train):,} data")
        print(
            f"[HOLDOUT] {len(X_unused):,} data "
            "(tidak dievaluasi)"
        )

        cv = make_grouped_cv(CV_FOLDS)

        pipeline = build_pipeline(tfidf_cfg, svm_cfg)

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
            groups=build_groups(subset).to_numpy(),
            n_jobs=1,
        )

        mean_f1 = float(scores.mean())
        std_f1 = float(scores.std(ddof=1))

        result = {
            "train_ratio": train_ratio,
            "test_ratio": test_ratio,
            "train_samples": int(len(X_train)),
            "unused_holdout_samples": int(len(X_unused)),
            "cv_f1_macro_mean": mean_f1,
            "cv_f1_macro_std": std_f1,
            "cv_scores": [float(score) for score in scores],
            "test_set_used": False,
        }

        results.append(result)

        print(f"CV Macro F1 : {mean_f1:.4f}")
        print(f"CV Std      : {std_f1:.4f}")

    # Selection is based only on CV Macro F1.
    # The holdout remains untouched.
    selected = max(
        results,
        key=lambda item: item["cv_f1_macro_mean"],
    )

    selected_train_ratio = selected["train_ratio"]
    selected_test_ratio = selected["test_ratio"]

    ranked_results = sorted(
        results,
        key=lambda item: item["cv_f1_macro_mean"],
        reverse=True,
    )

    for rank, item in enumerate(ranked_results, start=1):
        item["rank"] = rank
        item["selected"] = (
            item["train_ratio"] == selected_train_ratio
        )

    print()
    print("=" * 70)
    print("PEMILIHAN SPLIT TERBAIK")
    print("=" * 70)
    print(
        f"[BEST] Split        : "
        f"{selected_train_ratio:.0%}:{selected_test_ratio:.0%}"
    )
    print(
        f"[BEST] CV Macro F1  : "
        f"{selected['cv_f1_macro_mean']:.4f}"
    )
    print(
        f"[BEST] CV Std       : "
        f"{selected['cv_f1_macro_std']:.4f}"
    )
    print("[INFO] Dasar pemilihan: CV Macro F1 tertinggi")
    print("[INFO] Holdout digunakan untuk pemilihan: False")

    metadata = {
        "experiment": "experiment_1_split_ratio_sensitivity",
        "version": "FINAL_ARTIFACT_CHAIN",
        "dataset": {
            "total_samples": int(len(working)),
            "text_column": text_column,
            "label_column": label_column,
            "class_distribution": {
                str(label): int(count)
                for label, count in class_counts.items()
            },
        },
        "configuration": {
            "split_ratios_tested": ["70:30", "80:20", "90:10"],
            "final_split": "80%:20% (predefined grouped canonical split)",
            "diagnostic_best_ratio": f"{selected_train_ratio:.0%}:{selected_test_ratio:.0%}",
            "scope": "ratio sensitivity within canonical development only",
            "canonical_holdout_rows_excluded": len(canonical_holdout),
            "cv_method": "StratifiedGroupKFold",
            "random_state": RANDOM_STATE,
            "cv_folds": CV_FOLDS,
            "scoring": "f1_macro",
            "tfidf_source": str(previous_config_path),
            "svm_source": str(previous_config_path),
            "tfidf": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in tfidf_cfg.items()
            },
            "svm": svm_cfg,
        },
        "selection_method": "highest_cv_macro_f1",
        "test_set_used_for_selection": False,
        "results": results,
        "selected_result": selected,
        "artifact_chain": {
            "previous_configuration_source": str(previous_config_path),
            "parameters_taken_from_previous_experiment": False,
            "role": "predeclared diagnostic baseline; not final model selection",
            "tfidf_hardcoded": False,
            "svm_hardcoded": False,
        },
    }

    save_json(metadata, OUTPUT_FILE)

    print_header("EXPERIMENT 1 SELESAI")
    print(
        f"Diagnostic best ratio within development : "
        f"{selected_train_ratio:.0%}:{selected_test_ratio:.0%}"
    )
    print(
        f"CV Macro F1 : "
        f"{selected['cv_f1_macro_mean']:.4f}"
    )
    print(f"Output      : {OUTPUT_FILE}")

    return metadata


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 1 - Split ratio sensitivity "
            "using parameters from a previous experiment artifact."
        )
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path dataset CSV. Default: canonical dataset.",
    )

    parser.add_argument(
        "--previous-config",
        type=str,
        default=str(DEFAULT_PREVIOUS_CONFIG),
        help=(
            "Path JSON artifact experiment sebelumnya yang berisi "
            "parameter TF-IDF dan SVM."
        ),
    )

    return parser


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    print_header("EXPERIMENT 1")

    previous_config_path = Path(args.previous_config)

    if not previous_config_path.is_absolute():
        previous_config_path = PROJECT_ROOT / previous_config_path

    previous_config = load_json(
        previous_config_path,
        "Previous experiment configuration",
    )

    tfidf_cfg = extract_tfidf(previous_config)
    svm_cfg = extract_svm(previous_config)

    validate_previous_config(tfidf_cfg, svm_cfg)

    df, dataset_path = load_dataset(args.data)

    print(f"[DATA] File: {dataset_path}")
    print(f"[CONFIG] Previous artifact: {previous_config_path}")

    text_column = detect_text_column(df)
    label_column = detect_label_column(df)

    print(f"[DATA] Text column : {text_column}")
    print(f"[DATA] Label column: {label_column}")

    run_split_experiment(
        df=df,
        text_column=text_column,
        label_column=label_column,
        tfidf_cfg=tfidf_cfg,
        svm_cfg=svm_cfg,
        previous_config_path=previous_config_path,
    )


if __name__ == "__main__":
    main()
