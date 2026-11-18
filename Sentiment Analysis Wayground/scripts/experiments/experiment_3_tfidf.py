from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
try:
    from .common import (
        CV_FOLDS,
        FINAL_TEST_RATIO,
        FINAL_TRAIN_RATIO,
        RANDOM_STATE,
        make_grouped_cv,
        print_header,
    )
except ImportError:
    from common import (
        CV_FOLDS,
        FINAL_TEST_RATIO,
        FINAL_TRAIN_RATIO,
        RANDOM_STATE,
        make_grouped_cv,
        print_header,
    )
# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from shared.constants import (  # noqa: E402
    CANONICAL_DATASET_PATH,
    EXPERIMENTS_DIR,
)
from shared.experiment_contract import snapshot_sizes
# Experiment 3 harus dapat berjalan SEBELUM final_config.py dikunci.
# Karena itu metadata dataset final yang dibutuhkan eksperimen didefinisikan
# sebagai baseline lokal, bukan dibaca dari shared.final_config.py.
#
# Setelah seluruh eksperimen selesai, konfigurasi terbaik ditetapkan secara
# eksplisit di shared/final_config.py melalui lock_final_config.py.

DATASET_PATH = CANONICAL_DATASET_PATH
DATASET_RELATIVE_PATH = DATASET_PATH.relative_to(PROJECT_ROOT)

# Kontrak dataset (Tahap 5). Nilai diambil dari shared/final_config.py yang
# diperbarui oleh scripts/refresh_dataset_contract.py, sehingga tidak ada
# angka hardcode yang bisa basi.
EXPECTED_DATASET_ROWS, EXPECTED_DEVELOPMENT_ROWS, EXPECTED_HOLDOUT_ROWS = snapshot_sizes(DATASET_PATH)
EXPECTED_TEXT_COLUMN = "processed_text"
EXPECTED_LABEL_COLUMN = "sentiment_label_final"
OUTPUT_FILENAME = "experiment_3_tfidf_results.json"
OUTPUT_CSV_FILENAME = "experiment_3_tfidf_results.csv"
# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================
# Expanded search space:
# 5 max_features x 3 ngram x 2 sublinear x 4 min_df = 120
MAX_FEATURES = (1000, 2000, 3000, 5000, 8000)
NGRAM_RANGES = (
    (1, 1),
    (1, 2),
    (1, 3),
)
SUBLINEAR_OPTIONS = (
    True,
    False,
)
MIN_DF_OPTIONS = (
    1,
    2,
    3,
    5,
)
MAX_DF = 0.90
BASELINE_SVM_CONFIG = {
    "kernel": "linear",
    "C": 10.0,
    "class_weight": None,
}
# ============================================================
# HELPERS
# ============================================================
def detect_delimiter(path: Path) -> str:
    """Deteksi delimiter CSV berdasarkan header."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        header = f.readline()
    comma_count = header.count(",")
    semicolon_count = header.count(";")
    if comma_count == 0 and semicolon_count == 0:
        raise ValueError(
            f"Dataset tidak memiliki delimiter CSV yang dikenali: {path}"
        )
    return "," if comma_count >= semicolon_count else ";"
def load_canonical_dataset() -> tuple[pd.DataFrame, str, str]:
    """
    Load dataset final canonical dan validasi schema.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset canonical tidak ditemukan:\n{DATASET_PATH}"
        )
    delimiter = detect_delimiter(DATASET_PATH)
    df = pd.read_csv(
        DATASET_PATH,
        sep=delimiter,
        encoding="utf-8-sig",
        low_memory=False,
    )
    required = {EXPECTED_TEXT_COLUMN, EXPECTED_LABEL_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Schema dataset canonical tidak sesuai.\n"
            f"Missing: {sorted(missing)}\n"
            f"Columns: {list(df.columns)}"
        )
    df = df[[c for c in (EXPECTED_TEXT_COLUMN, EXPECTED_LABEL_COLUMN, "reviewId") if c in df]].copy()
    df[EXPECTED_TEXT_COLUMN] = (
        df[EXPECTED_TEXT_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    df[EXPECTED_LABEL_COLUMN] = (
        df[EXPECTED_LABEL_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    before = len(df)
    df = df[
        (df[EXPECTED_TEXT_COLUMN] != "")
        & (df[EXPECTED_LABEL_COLUMN] != "")
    ].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[WARN] Dropped empty text/label rows: {dropped}")
    classes = sorted(df[EXPECTED_LABEL_COLUMN].unique().tolist())
    expected_classes = {"negatif", "netral", "positif"}
    if set(classes) != expected_classes:
        raise ValueError(
            "Kelas dataset tidak sesuai.\n"
            f"Expected: {sorted(expected_classes)}\n"
            f"Actual: {classes}"
        )
    if len(df) != EXPECTED_DATASET_ROWS:
        raise ValueError(
            "Jumlah baris dataset canonical tidak sesuai dengan konfigurasi final.\n"
            f"Expected: {EXPECTED_DATASET_ROWS}\n"
            f"Actual: {len(df)}"
        )
    return df, EXPECTED_TEXT_COLUMN, EXPECTED_LABEL_COLUMN
def make_fixed_split(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
):
    """
    Membuat split final 80:20 yang SAMA dengan split canonical Stage 5.

    TAHAP 5: sebelumnya memakai `train_test_split` per baris, sehingga
    holdout eksperimen berbeda dari holdout canonical dan kebocoran
    `processed_text` yang sengaja dihilangkan Tahap 3 kembali muncul.
    Sekarang split diambil dari artefak Stage 5 (berkelompok).
    """
    from shared.splitting import build_groups, split_parts_for_dataset

    development, holdout, info = split_parts_for_dataset(
        df,
        label_col=label_column,
    )

    X_train = development[text_column].astype(str)
    y_train = development[label_column].astype(str)
    X_holdout = holdout[text_column].astype(str)
    y_holdout = holdout[label_column].astype(str)
    groups_train = build_groups(development, "processed_text").to_numpy()

    print(f"\n[FIXED SPLIT]")
    print(f"Development : {len(X_train)}")
    print(f"Holdout     : {len(X_holdout)}")
    print(f"Train ratio : {FINAL_TRAIN_RATIO:.0%}")
    print(f"Test ratio  : {FINAL_TEST_RATIO:.0%}")
    print(f"Random State: {RANDOM_STATE}")
    print("Stratify    : True")
    print("Holdout used for selection: False")
    print(f"Metode      : {info.get('method', info.get('source', '-'))}")
    print(f"Sumber      : {info.get('source', '-')} ({info.get('path', '-')})")

    return X_train, X_holdout, y_train, y_holdout, groups_train
def build_pipeline(
    max_features: int,
    ngram_range: tuple[int, int],
    sublinear_tf: bool,
    min_df: int,
) -> Pipeline:
    """
    Pipeline TF-IDF -> Linear SVM.
    TF-IDF di-fit ulang pada setiap fold CV sehingga validation fold
    tidak ikut memengaruhi vocabulary/IDF fold tersebut.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    sublinear_tf=sublinear_tf,
                    min_df=min_df,
                    max_df=MAX_DF,
                ),
            ),
            (
                "svm",
                SVC(
                    kernel=BASELINE_SVM_CONFIG["kernel"],
                    C=BASELINE_SVM_CONFIG["C"],
                    class_weight=BASELINE_SVM_CONFIG["class_weight"],
                ),
            ),
        ]
    )
def select_best_result(results: list[dict]) -> dict:
    """
    Ranking:
        1. CV Macro F1 tertinggi
        2. CV STD terendah
        3. max_features terkecil
        4. min_df terbesar
        5. ngram paling sederhana
        6. sublinear False
    Tie-breaker tambahan hanya dipakai jika metrik utama benar-benar sama.
    """
    if not results:
        raise ValueError("Tidak ada hasil eksperimen.")
    ranked = sorted(
        results,
        key=lambda result: (
            -result["cv_f1_mean"],
            result["cv_f1_std"],
            result["max_features"],
            result["min_df"],
            result["ngram_range"][1],
            result["sublinear_tf"],
        ),
    )
    return ranked[0]
def get_tied_best_results(results: list[dict], best: dict) -> list[dict]:
    """
    Ambil konfigurasi yang memiliki F1 dan STD sama dengan best
    sampai toleransi numerik kecil.
    """
    return [
        result
        for result in results
        if abs(result["cv_f1_mean"] - best["cv_f1_mean"]) < 1e-12
        and abs(result["cv_f1_std"] - best["cv_f1_std"]) < 1e-12
    ]
def measure_vocabulary_size(
    X_train: pd.Series,
    config: dict,
) -> int:
    """
    Mengukur vocabulary aktual setelah vectorizer di-fit pada seluruh
    development set.
    Penting:
    - max_features adalah batas maksimum jumlah fitur, bukan jumlah
      vocabulary yang pasti terbentuk.
    - Pengukuran ini hanya metadata/audit setelah CV selesai dan tidak
      digunakan untuk memilih konfigurasi.
    """
    vectorizer = TfidfVectorizer(
        max_features=config["max_features"],
        ngram_range=tuple(config["ngram_range"]),
        sublinear_tf=config["sublinear_tf"],
        min_df=config["min_df"],
        max_df=config["max_df"],
    )
    vectorizer.fit(X_train)
    return len(vectorizer.vocabulary_)
def enrich_vocabulary_sizes(
    X_train: pd.Series,
    results: list[dict],
) -> None:
    """
    Menambahkan vocabulary aktual ke setiap hasil eksperimen.
    Nilai ini dihitung setelah CV sehingga tidak memengaruhi selection rule.
    """
    for result in results:
        result["actual_vocabulary_size"] = measure_vocabulary_size(
            X_train,
            result,
        )
def summarize_max_features_tie(tied_best: list[dict]) -> list[dict]:
    """Ringkasan tie khusus max_features pada konfigurasi terbaik."""
    return [
        {
            "max_features": int(r["max_features"]),
            "actual_vocabulary_size": int(r["actual_vocabulary_size"]),
            "cv_f1_mean": float(r["cv_f1_mean"]),
            "cv_f1_std": float(r["cv_f1_std"]),
            "ngram_range": list(r["ngram_range"]),
            "min_df": int(r["min_df"]),
            "max_df": float(r["max_df"]),
            "sublinear_tf": bool(r["sublinear_tf"]),
        }
        for r in sorted(tied_best, key=lambda x: x["max_features"])
    ]
def get_max_features_tie_group(
    results: list[dict],
    best: dict,
    tolerance: float = 1e-10,
) -> list[dict]:
    """
    Mencari tie antar max_features secara ketat.
    Semua parameter TF-IDF selain max_features serta CV F1/STD
    harus identik dalam toleransi numerik.
    """
    group = []
    for r in results:
        same_params = (
            tuple(r["ngram_range"]) == tuple(best["ngram_range"])
            and bool(r["sublinear_tf"]) == bool(best["sublinear_tf"])
            and int(r["min_df"]) == int(best["min_df"])
            and abs(float(r["max_df"]) - float(best["max_df"])) <= tolerance
        )
        same_metrics = (
            abs(float(r["cv_f1_mean"]) - float(best["cv_f1_mean"])) <= tolerance
            and abs(float(r["cv_f1_std"]) - float(best["cv_f1_std"])) <= tolerance
        )
        if same_params and same_metrics:
            group.append(r)
    return sorted(group, key=lambda x: int(x["max_features"]))
def audit_vocabulary_per_cv_fold(
    X_train: pd.Series,
    y_train: pd.Series,
    tie_group: list[dict],
    cv,
    groups_train=None,
) -> list[dict]:
    """
    Mengukur actual vocabulary pada training portion setiap CV fold
    untuk konfigurasi max_features yang benar-benar tie.
    Audit ini hanya diagnostik dan tidak digunakan untuk memilih model.
    """
    audit = []
    for r in tie_group:
        fold_sizes = []
        for train_idx, _ in cv.split(X_train, y_train, groups_train):
            vectorizer = TfidfVectorizer(
                max_features=int(r["max_features"]),
                ngram_range=tuple(r["ngram_range"]),
                sublinear_tf=bool(r["sublinear_tf"]),
                min_df=int(r["min_df"]),
                max_df=float(r["max_df"]),
            )
            vectorizer.fit(X_train.iloc[train_idx])
            fold_sizes.append(len(vectorizer.vocabulary_))
        audit.append({
            "max_features": int(r["max_features"]),
            "fold_vocabulary_sizes": fold_sizes,
            "mean_vocabulary_size": float(sum(fold_sizes) / len(fold_sizes)),
            "min_vocabulary_size": int(min(fold_sizes)),
            "max_vocabulary_size": int(max(fold_sizes)),
            "all_folds_same": len(set(fold_sizes)) == 1,
        })
    return audit
def interpret_max_features_tie(tie_summary: list[dict]) -> str:
    """Membuat interpretasi otomatis untuk tie max_features."""
    if len(tie_summary) <= 1:
        return "Tidak terdapat tie antar max_features pada konfigurasi terbaik."
    vocab = {int(x["actual_vocabulary_size"]) for x in tie_summary}
    if len(vocab) == 1:
        return (
            "Actual vocabulary sama pada seluruh max_features yang tie. "
            "Setelah min_df dan max_df diterapkan, jumlah fitur aktual "
            "tidak mencapai batas max_features yang lebih besar. Karena "
            "representasi TF-IDF tetap sama, Macro F1 dan CV STD identik. "
            "max_features terkecil dipilih sebagai tie-breaker efisiensi."
        )
    reached = any(
        int(x["actual_vocabulary_size"]) >= int(x["max_features"])
        for x in tie_summary
    )
    if reached:
        return (
            "Actual vocabulary berbeda antar max_features yang tie dan "
            "sebagian konfigurasi mencapai batas max_features. Artinya "
            "kapasitas tambahan memungkinkan fitur tambahan, tetapi fitur "
            "tersebut tidak meningkatkan Macro F1. max_features terkecil "
            "dipilih sebagai tie-breaker efisiensi."
        )
    return (
        "Actual vocabulary berbeda antar max_features yang tie, tetapi "
        "konfigurasi dengan batas lebih besar belum mencapai kapasitasnya. "
        "Perbedaan batas maksimum tersebut tidak memberi keuntungan "
        "performa, sehingga max_features terkecil dipilih sebagai "
        "tie-breaker efisiensi."
    )
# ============================================================
# MAIN
# ============================================================
def main() -> dict:
    start_time = time.perf_counter()
    print_header("EXPERIMENT 3: TF-IDF OPTIMIZATION")
    # --------------------------------------------------------
    # 1. LOAD CANONICAL DATASET
    # --------------------------------------------------------
    df, text_column, label_column = load_canonical_dataset()
    print("\n[DATASET]")
    print(f"Path        : {DATASET_PATH}")
    print(f"Rows        : {len(df)}")
    print(f"Text column : {text_column}")
    print(f"Label column: {label_column}")
    print(f"Classes     : {sorted(df[label_column].unique().tolist())}")
    print("\n[LABEL DISTRIBUTION]")
    print(df[label_column].value_counts().sort_index())
    # --------------------------------------------------------
    # 2. FIXED DEVELOPMENT / HOLDOUT SPLIT
    # --------------------------------------------------------
    X_train, X_holdout, y_train, y_holdout, groups_train = make_fixed_split(
        df,
        text_column=text_column,
        label_column=label_column,
    )
    print("\n[FIXED SPLIT]")
    print(f"Development : {len(X_train)}")
    print(f"Holdout     : {len(X_holdout)}")
    print("Train ratio : 0.80")
    print("Test ratio  : 0.20")
    print(f"Random State: {RANDOM_STATE}")
    print("Stratify    : True")
    print("Holdout used for selection: False")
    if (
        len(X_train) != EXPECTED_DEVELOPMENT_ROWS
        or len(X_holdout) != EXPECTED_HOLDOUT_ROWS
    ):
        raise ValueError(
            "Ukuran fixed split tidak sesuai konfigurasi final.\n"
            f"Expected: train={EXPECTED_DEVELOPMENT_ROWS}, "
            f"holdout={EXPECTED_HOLDOUT_ROWS}\n"
            f"Actual: train={len(X_train)}, holdout={len(X_holdout)}"
        )
    # Holdout sengaja tidak digunakan lagi pada Experiment 3.
    del X_holdout, y_holdout
    # --------------------------------------------------------
    # 3. CROSS VALIDATION
    # --------------------------------------------------------
    if CV_FOLDS != 5:
        raise ValueError(
            f"Experiment 3 membutuhkan 5-fold CV. Actual: {CV_FOLDS}"
        )
    # TAHAP 5: CV berkelompok (grup = processed_text) supaya baris dengan
    # teks identik tidak berada di fold latih dan fold validasi sekaligus.
    cv = make_grouped_cv(CV_FOLDS)
    # --------------------------------------------------------
    # 4. SEARCH SPACE
    # --------------------------------------------------------
    total_configurations = (
        len(MAX_FEATURES)
        * len(NGRAM_RANGES)
        * len(SUBLINEAR_OPTIONS)
        * len(MIN_DF_OPTIONS)
    )
    expected_configurations = 120
    if total_configurations != expected_configurations:
        raise RuntimeError(
            "Search space Experiment 3 tidak sesuai konfigurasi final.\n"
            f"Expected: {expected_configurations}\n"
            f"Actual: {total_configurations}"
        )
    print("\n[SEARCH SPACE - EXPANDED]")
    print(f"Max Features : {list(MAX_FEATURES)}")
    print(f"N-gram       : {[list(x) for x in NGRAM_RANGES]}")
    print(f"Sublinear TF : {list(SUBLINEAR_OPTIONS)}")
    print(f"Min DF       : {list(MIN_DF_OPTIONS)}")
    print(f"Max DF       : {MAX_DF}")
    print(f"Configurations: {total_configurations}")
    print(f"CV Folds     : {CV_FOLDS}")
    print("Metric       : Macro F1")
    # --------------------------------------------------------
    # 5. RUN EXPERIMENT
    # --------------------------------------------------------
    results: list[dict] = []
    experiment_start = time.perf_counter()
    for max_features in MAX_FEATURES:
        for ngram_range in NGRAM_RANGES:
            for sublinear_tf in SUBLINEAR_OPTIONS:
                for min_df in MIN_DF_OPTIONS:
                    config_start = time.perf_counter()
                    pipeline = build_pipeline(
                        max_features=max_features,
                        ngram_range=ngram_range,
                        sublinear_tf=sublinear_tf,
                        min_df=min_df,
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
                        "max_features": int(max_features),
                        "ngram_range": list(ngram_range),
                        "sublinear_tf": bool(sublinear_tf),
                        "min_df": int(min_df),
                        "max_df": float(MAX_DF),
                        "cv_f1_mean": float(scores.mean()),
                        "cv_f1_std": float(scores.std(ddof=1)),
                        "cv_f1_scores": [float(score) for score in scores],
                        "runtime_seconds": round(elapsed, 4),
                    }
                    results.append(result)
                    print(
                        f"features={max_features:4d} | "
                        f"ngram={ngram_range} | "
                        f"sublinear={sublinear_tf!s:5s} | "
                        f"min_df={min_df} | "
                        f"F1={scores.mean():.4f} | "
                        f"STD={scores.std(ddof=1):.4f}"
                    )
    experiment_elapsed = time.perf_counter() - experiment_start
    if len(results) != total_configurations:
        raise RuntimeError(
            "Jumlah hasil tidak sesuai.\n"
            f"Expected: {total_configurations}\n"
            f"Actual: {len(results)}"
        )
    # --------------------------------------------------------
    # 6. SELECT BEST
    # --------------------------------------------------------
    best = select_best_result(results)
    # Actual vocabulary dihitung setelah CV sebagai metadata/audit.
    # Tidak digunakan untuk memilih konfigurasi.
    enrich_vocabulary_sizes(X_train, results)
    # Tie khusus max_features: parameter TF-IDF lain dan performa
    # CV harus identik. Ini menjelaskan tie yang terlihat pada TOP 5.
    tie_max_features = get_max_features_tie_group(
        results=results,
        best=best,
        tolerance=1e-10,
    )
    tie_max_features_summary = summarize_max_features_tie(
        tie_max_features
    )
    # Audit vocabulary pada setiap training fold untuk konfigurasi tie.
    tie_vocabulary_audit = audit_vocabulary_per_cv_fold(
        X_train=X_train,
        y_train=y_train,
        tie_group=tie_max_features,
        cv=cv,
        groups_train=groups_train,
    )
    tie_interpretation = interpret_max_features_tie(
        tie_max_features_summary
    )
    ranked_results = sorted(
        results,
        key=lambda result: (
            -result["cv_f1_mean"],
            result["cv_f1_std"],
            result["max_features"],
            result["min_df"],
            result["ngram_range"][1],
            result["sublinear_tf"],
        ),
    )
    for rank, result in enumerate(ranked_results, start=1):
        result["rank"] = rank
    actual_vocabulary_size = int(best["actual_vocabulary_size"])
    # --------------------------------------------------------
    # 7. LOCKED CONFIGURATION
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("LOCKED TF-IDF")
    print("=" * 70)
    print(f"Max Features        : {best['max_features']}")
    print(f"N-gram              : {tuple(best['ngram_range'])}")
    print(f"Sublinear TF        : {best['sublinear_tf']}")
    print(f"Min DF              : {best['min_df']}")
    print(f"Max DF              : {best['max_df']}")
    print(f"Actual Vocabulary   : {actual_vocabulary_size}")
    print("\n[PERFORMANCE]")
    print(f"Best CV Macro F1    : {best['cv_f1_mean']:.4f}")
    print(f"CV Std              : {best['cv_f1_std']:.4f}")
    print(f"Tied Max-Features   : {len(tie_max_features)}")
    if len(tie_max_features) > 1:
        print(
            "\n[INFO] Terdapat tie antar max_features dengan parameter "
            "TF-IDF lain, CV Macro F1, dan CV STD identik."
        )
        print(
            "[INFO] max_features terkecil dipilih sebagai tie-breaker "
            "efisiensi. Actual vocabulary hanya digunakan sebagai audit."
        )
        print("\n[TIE ANALYSIS - MAX_FEATURES]")
        print(
            "max_features | actual_vocab | CV Macro F1 | CV STD"
        )
        print("-" * 60)
        for item in tie_max_features_summary:
            print(
                f"{item['max_features']:12d} | "
                f"{item['actual_vocabulary_size']:12d} | "
                f"{item['cv_f1_mean']:.4f}      | "
                f"{item['cv_f1_std']:.4f}"
            )
        print("\n[TIE VOCABULARY AUDIT - PER CV FOLD]")
        print(
            "max_features | fold_vocabularies          | mean | min | max"
        )
        print("-" * 78)
        for item in tie_vocabulary_audit:
            print(
                f"{item['max_features']:12d} | "
                f"{str(item['fold_vocabulary_sizes']):27s} | "
                f"{item['mean_vocabulary_size']:6.1f} | "
                f"{item['min_vocabulary_size']:3d} | "
                f"{item['max_vocabulary_size']:3d}"
            )
        print("\n[INTERPRETATION]")
        print(tie_interpretation)
    # --------------------------------------------------------
    # 8. FINAL OUTPUT
    # --------------------------------------------------------
    output = {
        "experiment": "experiment_3_tfidf",
        "version": "expanded_120_configs_vocab_tie_audit_v4",
        "description": (
            "Optimasi TF-IDF menggunakan Pipeline pada development set "
            "dengan Stratified 5-Fold Cross-Validation Macro F1. "
            "Holdout tidak digunakan untuk pemilihan konfigurasi. "
            "Search space terdiri dari 120 konfigurasi. Vocabulary aktual dihitung sebagai metadata audit untuk menjelaskan tie antar max_features."
        ),
        "methodology": {
            "purpose": (
                "Optimasi konfigurasi TF-IDF sebelum optimasi SVM "
                "pada Experiment 4."
            ),
            "dataset": {
                "path": str(
                    DATASET_PATH.relative_to(PROJECT_ROOT)
                ).replace("\\\\\\\\\\\\\\\\", "/"),
                "filename": DATASET_PATH.name,
                "rows": int(len(df)),
                "text_column": text_column,
                "label_column": label_column,
            },
            "split": {
                "train_rows": int(len(X_train)),
                "test_rows": EXPECTED_HOLDOUT_ROWS,
                "train_ratio": 0.80,
                "test_ratio": 0.20,
                "random_state": RANDOM_STATE,
                "stratify": True,
            },
            "cross_validation": {
                "method": "StratifiedGroupKFold (grup = processed_text)",
                "folds": CV_FOLDS,
                "shuffle": True,
                "random_state": RANDOM_STATE,
                "scoring": "f1_macro",
            },
            "holdout_test_used_for_selection": False,
            "pipeline_prevents_tfidf_leakage": True,
            "vocabulary_size_note": (
                "actual_vocabulary_size dihitung setelah CV pada seluruh "
                "development set dan tidak digunakan untuk pemilihan model."
            ),
        },
        "fixed_svm_baseline": {
            "kernel": BASELINE_SVM_CONFIG["kernel"],
            "C": BASELINE_SVM_CONFIG["C"],
            "class_weight": BASELINE_SVM_CONFIG["class_weight"],
        },
        "configuration_source": {
            "final_config_dependency": False,
            "source_of_truth": (
                "experiment_result_until_final_config_is_locked"
            ),
            "dataset_metadata_is_baseline_only": True,
        },
        "search_space": {
            "max_features": [int(x) for x in MAX_FEATURES],
            "ngram_ranges": [list(x) for x in NGRAM_RANGES],
            "sublinear_tf": [bool(x) for x in SUBLINEAR_OPTIONS],
            "min_df": [int(x) for x in MIN_DF_OPTIONS],
            "max_df": float(MAX_DF),
            "total_configurations": total_configurations,
        },
        "selection_rule": {
            "primary": "Highest CV Macro F1",
            "secondary": "Lowest CV standard deviation",
            "tertiary": (
                "Lowest max_features for exact CV metric tie, after "
                "checking actual vocabulary size"
            ),
            "additional_tiebreakers": [
                "Lowest max_features",
                "Highest min_df",
                "Simpler ngram range",
                "sublinear_tf=False",
            ],
        },
        "tie_analysis": {
            "number_of_exact_best_configs": len(tie_max_features),
            "best_configuration_selected_by_efficiency_tiebreak": (
                len(tie_max_features) > 1
            ),
            "max_features_tie_summary": tie_max_features_summary,
            "vocabulary_audit_per_cv_fold": tie_vocabulary_audit,
            "interpretation": (
                "max_features merupakan batas maksimum vocabulary. Jika "
                "actual_vocabulary_size sama pada beberapa max_features "
                "yang memiliki CV Macro F1 dan CV STD identik, maka nilai "
                "max_features yang lebih besar belum memberikan fitur "
                "tambahan pada konfigurasi tersebut."
            ),
        },
        "best": {
            **best,
            "actual_vocabulary_size_on_full_development": (
                actual_vocabulary_size
            ),
        },
        "best_configuration": {
            **best,
            "actual_vocabulary_size_on_full_development": (
                actual_vocabulary_size
            ),
        },
        "locked_tfidf": {
            "max_features": best["max_features"],
            "ngram_range": best["ngram_range"],
            "sublinear_tf": best["sublinear_tf"],
            "min_df": best["min_df"],
            "max_df": best["max_df"],
            "actual_vocabulary_size": actual_vocabulary_size,
        },
        "results": ranked_results,
        "runtime": {
            "experiment_seconds": round(experiment_elapsed, 4),
            "total_seconds": round(
                time.perf_counter() - start_time,
                4,
            ),
        },
    }
    # --------------------------------------------------------
    # 9. SAVE JSON + CSV
    # --------------------------------------------------------
    output_dir = EXPERIMENTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME
    csv_path = output_dir / OUTPUT_CSV_FILENAME
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
    # CSV memudahkan audit dan analisis ranking eksperimen.
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
