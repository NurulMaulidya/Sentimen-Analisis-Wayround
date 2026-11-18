"""
Experiment 4 - SVM Kernel & Hyperparameter Optimization

Wayground Sentiment Analysis

Tujuan:
- Mengoptimasi kernel, C, dan gamma SVM.
- TF-IDF dikunci dari Experiment 3.
- Pemilihan konfigurasi berdasarkan Stratified 5-Fold CV Macro F1.
- Holdout tidak digunakan untuk pemilihan konfigurasi.

Kriteria pemilihan:
1. CV Macro F1 tertinggi.
2. Jika tie, CV STD terendah.
3. Jika masih tie, konfigurasi SVM yang lebih sederhana.
4. Jika masih tie, C yang lebih kecil.

Search space:
- Kernel: linear, rbf, poly, sigmoid
- C: [0.01, 0.1, 0.5, 1.0, 1.5, 2.0]
- Gamma non-linear: ["scale", "auto", 0.01, 0.1, 1.0]

Total:
- Linear: 6
- RBF: 6 x 5 = 30
- Poly: 6 x 5 = 30
- Sigmoid: 6 x 5 = 30
- Total = 128 konfigurasi
"""

from __future__ import annotations

import json
import sys
import time
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
from shared.constants import CANONICAL_DATASET_PATH, EXPERIMENTS_DIR  # noqa: E402

try:
    from .common import (
        CV_FOLDS,
        GROUP_COLUMN,
        RANDOM_STATE,
        make_grouped_cv,
        print_header,
    )
except ImportError:
    from common import (
        CV_FOLDS,
        GROUP_COLUMN,
        RANDOM_STATE,
        make_grouped_cv,
        print_header,
    )


# ============================================================
# PATHS
# ============================================================

# TAHAP 5: dataset canonical mengikuti shared.constants sehingga ikut
# mengikuti env WAYGROUND_PIPELINE_OUTPUT_DIR. Sebelumnya path ditulis
# manual ke `data/pipeline_output` sehingga Experiment 4 berjalan di atas
# dataset LAMA (3.060 baris) sementara Experiment 3 memakai dataset baru
# (3.082 baris) - rantai artefak jadi tidak konsisten.
DATASET_PATH = CANONICAL_DATASET_PATH

# Only consume the canonical artifact; version suffixes do not prove freshness.
EXP3_RESULT_CANDIDATES = [
    EXPERIMENTS_DIR / "experiment_3_tfidf_results.json",
]
EXP3_RESULT_PATH = EXPERIMENTS_DIR / "experiment_3_tfidf_results.json"
# shared.final_config hanya digunakan setelah seluruh eksperimen selesai
# untuk mengunci konfigurasi final.
# (Path dataset & artefak Exp3 ditetapkan di blok PATHS di atas.)

OUTPUT_FILENAME = "experiment_4_svm_results.json"
OUTPUT_CSV = "experiment_4_svm_results.csv"


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

# Lebih ringkas daripada eksperimen sebelumnya:
# mencakup rentang regularisasi yang luas tanpa terlalu granular.
C_VALUES = (
    0.01,
    0.1,
    0.5,
    1.0,
    1.5,
    2.0,
    5.0,
    10.0,
)

SVM_KERNELS = (
    "linear",
    "rbf",
    "poly",
    "sigmoid",
)

# Gamma tidak relevan untuk linear kernel.
GAMMA_VALUES = (
    "scale",
    "auto",
    0.01,
    0.1,
    1.0,
)

# Prioritas kompleksitas model untuk tie yang benar-benar identik.
# Linear paling sederhana; RBF diprioritaskan sebelum sigmoid/poly
# karena RBF adalah kandidat non-linear utama dalam penelitian ini.
KERNEL_COMPLEXITY = {
    "linear": 0,
    "rbf": 1,
    "sigmoid": 2,
    "poly": 3,
}


# ============================================================
# HELPERS
# ============================================================

def detect_delimiter(path: Path) -> str:
    """Deteksi delimiter CSV sederhana berdasarkan header."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        header = f.readline()

    comma_count = header.count(",")
    semicolon_count = header.count(";")

    if comma_count == 0 and semicolon_count == 0:
        raise ValueError(f"Delimiter CSV tidak dikenali: {path}")

    return "," if comma_count >= semicolon_count else ";"


def load_canonical_dataset() -> tuple[pd.DataFrame, str, str]:
    """Load dan validasi dataset canonical."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATASET_PATH}"
        )

    delimiter = detect_delimiter(DATASET_PATH)
    df = pd.read_csv(
        DATASET_PATH,
        sep=delimiter,
        encoding="utf-8-sig",
        low_memory=False,
    )

    text_col = "processed_text"
    label_col = "sentiment_label_final"

    missing = {text_col, label_col}.difference(df.columns)
    if missing:
        raise ValueError(
            f"Kolom dataset tidak lengkap. Missing: {sorted(missing)}"
        )

    df = df[[c for c in (text_col, label_col, "reviewId") if c in df.columns]].copy()

    df[text_col] = (
        df[text_col]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df[label_col] = (
        df[label_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = df[
        (df[text_col] != "")
        & (df[label_col] != "")
    ].reset_index(drop=True)

    expected_classes = {"negatif", "netral", "positif"}
    actual_classes = set(df[label_col].unique())

    if actual_classes != expected_classes:
        raise ValueError(
            "Kelas dataset tidak sesuai.\n"
            f"Expected: {sorted(expected_classes)}\n"
            f"Actual: {sorted(actual_classes)}"
        )

    return df, text_col, label_col


def load_locked_tfidf() -> dict[str, Any]:
    """Load konfigurasi TF-IDF yang sudah dikunci dari Experiment 3."""
    # The canonical filename follows WAYGROUND_OUTPUT_DIR. Legacy suffixed
    # files must not silently replace a missing result from the current run.
    exp3_path = None
    for candidate in EXP3_RESULT_CANDIDATES:
        if candidate.exists():
            exp3_path = candidate
            break

    if exp3_path is None:
        kandidat = "\n".join(f"    - {p}" for p in EXP3_RESULT_CANDIDATES)
        raise FileNotFoundError(
            f"Hasil Experiment 3 tidak ditemukan.\n"
            f"Output dir : {EXPERIMENTS_DIR}\n"
            f"Kandidat:\n{kandidat}\n"
            "Jalankan experiment_3_tfidf.py terlebih dahulu."
        )

    print(f"  Artefak Exp3 : {exp3_path.name}")

    with exp3_path.open("r", encoding="utf-8") as f:
        exp3 = json.load(f)

    best = exp3.get("best") or exp3.get("locked_tfidf") or exp3.get("best_configuration")

    if not isinstance(best, dict):
        raise ValueError(
            "Artefak Experiment 3 tidak memiliki object konfigurasi "
            "TF-IDF ('best' / 'locked_tfidf' / 'best_configuration')."
        )

    required = {
        "max_features",
        "ngram_range",
        "sublinear_tf",
        "min_df",
        "max_df",
    }
    missing = required.difference(best)

    if missing:
        raise ValueError(
            "Konfigurasi TF-IDF Experiment 3 tidak lengkap. "
            f"Missing: {sorted(missing)}"
        )

    return {
        "max_features": int(best["max_features"]),
        "ngram_range": list(best["ngram_range"]),
        "sublinear_tf": bool(best["sublinear_tf"]),
        "min_df": int(best["min_df"]),
        "max_df": float(best["max_df"]),
        "actual_vocabulary_size": best.get(
            "actual_vocabulary_size_on_full_development",
            best.get("actual_vocabulary_size"),
        ),
    }


def build_pipeline(
    tfidf_config: dict[str, Any],
    kernel: str,
    C: float,
    gamma: str | float = "scale",
) -> Pipeline:
    """Pipeline TF-IDF -> SVM."""
    tfidf = TfidfVectorizer(
        max_features=tfidf_config["max_features"],
        ngram_range=tuple(tfidf_config["ngram_range"]),
        sublinear_tf=tfidf_config["sublinear_tf"],
        min_df=tfidf_config["min_df"],
        max_df=tfidf_config["max_df"],
    )

    svm_kwargs: dict[str, Any] = {
        "kernel": kernel,
        "C": float(C),
    }

    if kernel != "linear":
        svm_kwargs["gamma"] = gamma

    return Pipeline(
        steps=[
            ("tfidf", tfidf),
            ("svm", SVC(**svm_kwargs)),
        ]
    )


def calculate_total_experiments() -> int:
    """Hitung total konfigurasi yang benar-benar dijalankan."""
    total = 0

    for kernel in SVM_KERNELS:
        if kernel == "linear":
            total += len(C_VALUES)
        else:
            total += len(C_VALUES) * len(GAMMA_VALUES)

    return total


def select_best_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Seleksi konfigurasi:
    1. Macro F1 tertinggi.
    2. STD terendah.
    3. Kernel lebih sederhana.
    4. C lebih kecil.

    Holdout tidak digunakan.
    """
    if not results:
        raise ValueError("Tidak ada hasil eksperimen.")

    ranked = sorted(
        results,
        key=lambda r: (
            -r["cv_f1_mean"],
            r["cv_f1_std"],
            KERNEL_COMPLEXITY[r["kernel"]],
            r["C"],
            str(r["gamma"]),
        ),
    )

    return ranked[0]


def get_exact_metric_ties(
    results: list[dict[str, Any]],
    best: dict[str, Any],
    tolerance: float = 1e-12,
) -> list[dict[str, Any]]:
    """Ambil konfigurasi dengan F1 dan STD identik secara numerik."""
    return [
        r
        for r in results
        if abs(r["cv_f1_mean"] - best["cv_f1_mean"]) <= tolerance
        and abs(r["cv_f1_std"] - best["cv_f1_std"]) <= tolerance
    ]


def summarize_near_best(
    results: list[dict[str, Any]],
    best: dict[str, Any],
    tolerance: float = 0.001,
) -> list[dict[str, Any]]:
    """
    Ambil konfigurasi yang sangat dekat dengan best.
    Ini untuk analisis, bukan tie-breaker.
    """
    return [
        r
        for r in results
        if (
            best["cv_f1_mean"] - r["cv_f1_mean"]
        ) <= tolerance
    ]


# ============================================================
# MAIN
# ============================================================

def main() -> dict[str, Any]:
    start_time = time.perf_counter()

    try:
        print_header("EXPERIMENT 4: SVM OPTIMIZATION")
    except Exception:
        print("\n" + "=" * 70)
        print("EXPERIMENT 4: SVM OPTIMIZATION")
        print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD DATASET
    # --------------------------------------------------------

    df, text_col, label_col = load_canonical_dataset()

    print("\n[DATASET]")
    print(f"Path        : {DATASET_PATH}")
    print(f"Rows        : {len(df)}")
    print(f"Text column : {text_col}")
    print(f"Label column: {label_col}")

    print("\n[LABEL DISTRIBUTION]")
    print(df[label_col].value_counts().sort_index())

    # --------------------------------------------------------
    # 2. FIXED DEVELOPMENT / HOLDOUT SPLIT
    # --------------------------------------------------------
    #
    # TAHAP 5: sebelumnya memakai `train_test_split` per baris. Akibatnya
    # holdout eksperimen berbeda dari holdout canonical Stage 5 dan
    # kebocoran `processed_text` yang dihilangkan Tahap 3 kembali muncul.
    # Sekarang split diambil dari artefak Stage 5 (berkelompok).

    from shared.splitting import build_groups, split_parts_for_dataset

    development, holdout, split_info = split_parts_for_dataset(
        df,
        group_col=GROUP_COLUMN,
        label_col=label_col,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    X_train = development[text_col].astype(str)
    y_train = development[label_col].astype(str)
    X_holdout = holdout[text_col].astype(str)
    y_holdout = holdout[label_col].astype(str)
    groups_train = build_groups(development, GROUP_COLUMN).to_numpy()

    print("\n[FIXED SPLIT]")
    print(f"Development : {len(X_train)}")
    print(f"Holdout     : {len(X_holdout)}")
    print("Train ratio : 0.80")
    print("Test ratio  : 0.20")
    print(f"Random State: {RANDOM_STATE}")
    print("Stratify    : True")
    print("Holdout used for selection: False")
    print(
        "Metode      : "
        f"{split_info.get('method', split_info.get('source', '-'))}"
    )
    print(
        "Sumber      : "
        f"{split_info.get('source', '-')} ({split_info.get('path', '-')})"
    )

    # Holdout sengaja tidak digunakan selama optimasi.
    del X_holdout, y_holdout

    # --------------------------------------------------------
    # 3. LOAD LOCKED TF-IDF
    # --------------------------------------------------------

    tfidf_config = load_locked_tfidf()

    print("\n[LOCKED TF-IDF FROM EXPERIMENT 3]")
    print(f"  max_features : {tfidf_config['max_features']}")
    print(f"  ngram_range  : {tfidf_config['ngram_range']}")
    print(f"  sublinear_tf : {tfidf_config['sublinear_tf']}")
    print(f"  min_df       : {tfidf_config['min_df']}")
    print(f"  max_df       : {tfidf_config['max_df']}")

    if tfidf_config.get("actual_vocabulary_size") is not None:
        print(
            "  actual_vocab : "
            f"{tfidf_config['actual_vocabulary_size']}"
        )

    # --------------------------------------------------------
    # 4. CV SETUP
    # --------------------------------------------------------

    if CV_FOLDS != 5:
        raise ValueError(
            f"Experiment 4 membutuhkan 5-fold CV. Actual: {CV_FOLDS}"
        )

    cv = make_grouped_cv(CV_FOLDS)

    # --------------------------------------------------------
    # 5. SEARCH SPACE
    # --------------------------------------------------------

    total_experiments = calculate_total_experiments()
    expected_experiments = 128

    if total_experiments != expected_experiments:
        raise RuntimeError(
            "Search space tidak sesuai konfigurasi yang direncanakan.\n"
            f"Expected: {expected_experiments}\n"
            f"Actual: {total_experiments}"
        )

    print("\n[SEARCH SPACE]")
    print(f"  Kernels      : {list(SVM_KERNELS)}")
    print(f"  C values     : {list(C_VALUES)}")
    print(f"  Gamma values : {list(GAMMA_VALUES)}")
    print(f"  Total        : {total_experiments}")
    print(f"  CV Folds     : {CV_FOLDS}")
    print("  Metric       : Macro F1")

    print("\n[SELECTION RULE]")
    print("  1. Highest CV Macro F1")
    print("  2. Lowest CV STD")
    print("  3. Simpler kernel for exact metric tie")
    print("  4. Lower C for remaining exact tie")
    print("  Holdout is NOT used for selection")

    # --------------------------------------------------------
    # 6. RUN EXPERIMENTS
    # --------------------------------------------------------

    results: list[dict[str, Any]] = []
    experiment_number = 0
    experiment_start = time.perf_counter()

    for kernel in SVM_KERNELS:
        gamma_list = (
            ("scale",)
            if kernel == "linear"
            else GAMMA_VALUES
        )

        for C in C_VALUES:
            for gamma in gamma_list:
                experiment_number += 1
                config_start = time.perf_counter()

                pipeline = build_pipeline(
                    tfidf_config=tfidf_config,
                    kernel=kernel,
                    C=C,
                    gamma=gamma,
                )

                scores = cross_val_score(
                    pipeline,
                    X_train,
                    y_train,
                    cv=cv,
                    groups=groups_train,
                    scoring="f1_macro",
                    n_jobs=1,
                )

                elapsed = time.perf_counter() - config_start

                result = {
                    "experiment_number": experiment_number,
                    "kernel": kernel,
                    "C": float(C),
                    "gamma": gamma if kernel != "linear" else "scale",
                    "cv_f1_mean": float(scores.mean()),
                    "cv_f1_std": float(scores.std(ddof=1)),
                    "cv_f1_scores": [
                        float(score) for score in scores
                    ],
                    "runtime_seconds": round(elapsed, 4),
                }

                results.append(result)

                print(
                    f"[{experiment_number:3d}/{total_experiments}] "
                    f"{kernel:<8} "
                    f"C={C:<7g} "
                    f"gamma={str(gamma):<6} "
                    f"F1={scores.mean():.4f} "
                    f"+/- {scores.std(ddof=1):.4f}"
                )

    experiment_elapsed = time.perf_counter() - experiment_start

    if len(results) != total_experiments:
        raise RuntimeError(
            "Jumlah hasil tidak sesuai.\n"
            f"Expected: {total_experiments}\n"
            f"Actual: {len(results)}"
        )

    # --------------------------------------------------------
    # 7. RANK RESULTS
    # --------------------------------------------------------

    ranked_results = sorted(
        results,
        key=lambda r: (
            -r["cv_f1_mean"],
            r["cv_f1_std"],
            KERNEL_COMPLEXITY[r["kernel"]],
            r["C"],
            str(r["gamma"]),
        ),
    )

    for rank, result in enumerate(ranked_results, start=1):
        result["rank"] = rank

    best = select_best_result(results)
    exact_ties = get_exact_metric_ties(results, best)
    near_best = summarize_near_best(results, best, tolerance=0.001)

    # --------------------------------------------------------
    # 8. TOP 10
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP 10 SVM CONFIGURATIONS")
    print("=" * 70)

    for result in ranked_results[:10]:
        print(
            f"{result['rank']:2d}. "
            f"{result['kernel']:<8} "
            f"C={result['C']:<7g} "
            f"gamma={str(result['gamma']):<6} "
            f"F1={result['cv_f1_mean']:.4f} "
            f"+/- {result['cv_f1_std']:.4f}"
        )

    # --------------------------------------------------------
    # 9. TIE / NEAR-BEST ANALYSIS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TIE & GENERALIZATION-CANDIDATE ANALYSIS")
    print("=" * 70)

    print(
        f"Exact metric ties with best: {len(exact_ties)} "
        "(F1 and STD identical)"
    )

    if len(exact_ties) > 1:
        print(
            "\n[INFO] Beberapa konfigurasi memiliki Macro F1 dan "
            "STD yang identik."
        )
        print(
            "Pemilihan dilakukan berdasarkan kesederhanaan kernel "
            "kemudian nilai C yang lebih kecil."
        )

    print(
        f"\nConfigurations within 0.001 F1 of best: "
        f"{len(near_best)}"
    )

    print(
        "\n[NOTE] Near-best bukan tie. "
        "Perbedaan kecil digunakan untuk analisis stabilitas kandidat, "
        "bukan untuk mengganti kriteria utama."
    )

    # --------------------------------------------------------
    # 10. BEST CONFIGURATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BEST SVM CONFIGURATION")
    print("=" * 70)

    print(f"Kernel      : {best['kernel']}")
    print(f"C           : {best['C']}")
    print(f"Gamma       : {best['gamma']}")
    print(f"CV Macro F1 : {best['cv_f1_mean']:.4f}")
    print(f"CV Std      : {best['cv_f1_std']:.4f}")
    print("Selection   : Macro F1 -> STD -> simplicity -> lower C")

    # --------------------------------------------------------
    # 11. OUTPUT
    # --------------------------------------------------------

    output_dir = EXPERIMENTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "experiment": "experiment_4_svm",
        "version": "grouped_search_128_configs_stage3",
        "configuration_source": {
            "final_config_dependency": False,
            "tfidf_source": str(EXP3_RESULT_PATH),
            "final_config_is_locked_after_all_experiments": True,
        },
        "description": (
            "Optimasi SVM menggunakan TF-IDF yang telah dikunci "
            "dari Experiment 3. Pemilihan berdasarkan Stratified "
            "5-Fold Cross-Validation Macro F1. Holdout tidak "
            "digunakan untuk pemilihan."
        ),
        "methodology": {
            "dataset": str(
                DATASET_PATH.relative_to(PROJECT_ROOT)
            ).replace("\\", "/"),
            "dataset_rows": int(len(df)),
            "development_rows": int(len(X_train)),
            "holdout_rows": int(
                len(df) - len(X_train)
            ),
            "cv_method": "StratifiedGroupKFold (grup = processed_text)",
            "cv_folds": int(CV_FOLDS),
            "cv_shuffle": True,
            "random_state": int(RANDOM_STATE),
            "scoring": "f1_macro",
            "holdout_used_for_selection": False,
            "tfidf_locked_from_experiment_3": True,
        },
        "tfidf_locked": tfidf_config,
        "search_space": {
            "kernels": list(SVM_KERNELS),
            "C_values": [float(c) for c in C_VALUES],
            "gamma_values": list(GAMMA_VALUES),
            "total_configurations": total_experiments,
            "configuration_breakdown": {
                "linear": len(C_VALUES),
                "rbf": len(C_VALUES) * len(GAMMA_VALUES),
                "poly": len(C_VALUES) * len(GAMMA_VALUES),
                "sigmoid": len(C_VALUES) * len(GAMMA_VALUES),
            },
        },
        "selection_rule": {
            "primary": "Highest CV Macro F1",
            "secondary": "Lowest CV standard deviation",
            "tertiary": "Simpler kernel for exact metric tie",
            "quaternary": "Lower C for remaining exact tie",
        },
        "tie_analysis": {
            "number_of_exact_metric_ties": len(exact_ties),
            "exact_metric_ties": exact_ties,
            "number_of_near_best_within_0_001_f1": len(near_best),
            "near_best_configurations": near_best,
        },
        "best": best,
        "top_10": ranked_results[:10],
        "results": ranked_results,
        "runtime_seconds": round(experiment_elapsed, 4),
        "total_runtime_seconds": round(
            time.perf_counter() - start_time,
            4,
        ),
    }

    output_path = output_dir / OUTPUT_FILENAME
    csv_path = output_dir / OUTPUT_CSV

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")

    pd.DataFrame(ranked_results).to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nOutput JSON: {output_path}")
    print(f"Output CSV : {csv_path}")

    return output


if __name__ == "__main__":
    main()
