"""
LOCK FINAL CONFIGURATION - WAYGROUND SENTIMENT ANALYSIS

Fungsi script ini:
1. Mengambil konfigurasi final HANYA dari shared/final_config.py.
2. Memastikan konfigurasi tersebut cocok dengan hasil eksperimen yang tersedia.
3. Memastikan dataset canonical sesuai kontrak final.
4. Menulis output/experiments/best_config.json sebagai snapshot konfigurasi yang
   sudah DIKUNCI.
5. TIDAK melakukan pemilihan parameter.
6. TIDAK melakukan training.
7. TIDAK menggunakan holdout untuk memilih konfigurasi.

PRINSIP ARSITEKTUR:

    Experiment scripts
          ↓
    hasil eksperimen
          ↓
    shared/final_config.py
          ↓
    lock_final_config.py
          ↓
    best_config.json
          ↓
    train_final_model.py

Jadi hanya shared/final_config.py yang menjadi sumber konfigurasi final.
Script ini hanya bertindak sebagai "pengunci + validator". Manusia memang
sudah cukup membuat bug konfigurasi, jadi jangan beri kesempatan pada script
ini untuk memilih konfigurasi kedua kalinya.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from shared.constants import (  # noqa: E402
    CANONICAL_DATASET_PATH,
    EXPERIMENTS_DIR,
    FINAL_MODEL_DIR,
)


# ---------------------------------------------------------------------------
# SINGLE SOURCE OF TRUTH
# ---------------------------------------------------------------------------

from shared.final_config import (  # noqa: E402
    CV_FOLDS,
    EXPECTED_CLASS_DISTRIBUTION,
    EXPECTED_DATASET_ROWS,
    EXPECTED_DEVELOPMENT_ROWS,
    EXPECTED_EXPERT_VALIDATED_ROWS,
    EXPECTED_HOLDOUT_ROWS,
    FINAL_MODEL_CONFIG,
    LABEL_COLUMN,
    LABELING_METHOD,
    LEXICON_BASE_METHOD,
    NEUTRAL_THRESHOLD,
    RANDOM_STATE,
    SELECTION_METRIC,
    SENTIMENT_CLASSES,
    STRATIFY,
    TEST_SIZE,
    TEXT_COLUMN,
    TRAIN_TEST_SPLIT_RATIO,
)


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

OUTPUT_DIR = EXPERIMENTS_DIR
OUTPUT_PATH = OUTPUT_DIR / "best_config.json"

# TAHAP 5: dataset canonical mengikuti shared.constants sehingga ikut
# env WAYGROUND_PIPELINE_OUTPUT_DIR (data/pipeline_output_v2).
DATASET_PATH = CANONICAL_DATASET_PATH

EXP3_PATH = OUTPUT_DIR / "experiment_3_tfidf_results.json"
EXP4_PATH = OUTPUT_DIR / "experiment_4_svm_results.json"
EXP5_PATH = OUTPUT_DIR / "experiment_5_class_imbalance_results.json"
EXP6_PATH = OUTPUT_DIR / "experiment_6_ovo_ovr_results.json"
EXP7_PATH = OUTPUT_DIR / "experiment_7_regularization_gap_constrained.json"


# ---------------------------------------------------------------------------
# GENERIC HELPERS
# ---------------------------------------------------------------------------

def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} tidak ditemukan:\n  {path}"
        )


def load_json(path: Path, label: str = "Artefak eksperimen") -> dict[str, Any]:
    require_file(path, label)

    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Format JSON tidak valid: {path}")

    return data


def assert_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(
            f"Validasi gagal: {name}\n"
            f"  Expected: {expected!r}\n"
            f"  Actual  : {actual!r}"
        )


def assert_close(
    name: str,
    actual: float,
    expected: float,
    tol: float = 1e-9,
) -> None:
    if abs(float(actual) - float(expected)) > tol:
        raise ValueError(
            f"Validasi gagal: {name}\n"
            f"  Expected: {expected!r}\n"
            f"  Actual  : {actual!r}"
        )


def normalize_ngram(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return [int(value[0]), int(value[1])]

    raise ValueError(f"ngram_range tidak valid: {value!r}")


def normalize_strategy(value: Any) -> str:
    raw = str(value).strip().lower().replace("_", "-")

    mapping = {
        "ovo": "ovo",
        "one-vs-one": "ovo",
        "onevsone": "ovo",
        "ovr": "ovr",
        "one-vs-rest": "ovr",
        "onevsrest": "ovr",
    }

    if raw not in mapping:
        raise ValueError(
            f"Strategi multiclass tidak dikenali: {value!r}"
        )

    return mapping[raw]


def score(row: dict[str, Any]) -> tuple[float | None, float | None]:
    """Membaca skor CV tanpa menggunakan skor tersebut untuk selection."""

    mean_keys = [
        "cv_macro_f1_mean",
        "cv_f1_mean",
        "cv_macro_f1",
        "cv_mean_f1_macro",
        "mean_cv_macro_f1",
        "mean_f1_macro",
        "macro_f1_mean",
        "mean",
        "score",
    ]

    std_keys = [
        "cv_macro_f1_std",
        "cv_f1_std",
        "cv_macro_f1_std_dev",
        "cv_std_f1_macro",
        "std_cv_macro_f1",
        "std_f1_macro",
        "macro_f1_std",
        "std",
    ]

    mean = None
    std = None

    for key in mean_keys:
        value = row.get(key)
        if value is not None:
            try:
                mean = float(value)
                break
            except (TypeError, ValueError):
                pass

    for key in std_keys:
        value = row.get(key)
        if value is not None:
            try:
                std = float(value)
                break
            except (TypeError, ValueError):
                pass

    return mean, std


def find_dicts(obj: Any) -> list[dict[str, Any]]:
    """Ambil seluruh dictionary yang berada di dalam JSON secara rekursif."""

    found: list[dict[str, Any]] = []

    if isinstance(obj, dict):
        found.append(obj)

        for value in obj.values():
            found.extend(find_dicts(value))

    elif isinstance(obj, list):
        for value in obj:
            found.extend(find_dicts(value))

    return found


def get_result_rows(obj: Any) -> list[dict[str, Any]]:
    """Cari list hasil eksperimen tanpa memilih berdasarkan skor."""

    candidates: list[list[dict[str, Any]]] = []

    def walk(value: Any) -> None:
        if isinstance(value, list):
            if value and all(isinstance(x, dict) for x in value):
                candidates.append(value)

            for item in value:
                walk(item)

        elif isinstance(value, dict):
            for item in value.values():
                walk(item)

    walk(obj)

    # Prioritaskan list yang tampak seperti result grid.
    for rows in candidates:
        if any(
            any(
                key in row
                for key in (
                    "cv_macro_f1",
                    "cv_macro_f1_mean",
                    "cv_f1_mean",
                    "score",
                )
            )
            for row in rows
        ):
            return rows

    return candidates[0] if candidates else []


# ---------------------------------------------------------------------------
# DATASET VALIDATION
# ---------------------------------------------------------------------------

def validate_dataset() -> tuple[int, dict[str, int], int]:
    require_file(DATASET_PATH, "Canonical dataset")

    with DATASET_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        sample = f.read(8192)
        delimiter = (
            ","
            if sample.count(",") >= sample.count(";")
            else ";"
        )

        f.seek(0)
        reader = csv.DictReader(f, delimiter=delimiter)

        if (
            not reader.fieldnames
            or TEXT_COLUMN not in reader.fieldnames
            or LABEL_COLUMN not in reader.fieldnames
        ):
            raise ValueError(
                f"Dataset harus memiliki kolom "
                f"{TEXT_COLUMN!r} dan {LABEL_COLUMN!r}. "
                f"Header aktual: {reader.fieldnames}"
            )

        rows = list(reader)

    labels = [
        str(row.get(LABEL_COLUMN, "")).strip().lower()
        for row in rows
    ]

    if any(not str(row.get(TEXT_COLUMN, "")).strip() for row in rows):
        raise ValueError(
            "Ditemukan processed_text kosong pada dataset canonical."
        )

    if any(not label for label in labels):
        raise ValueError(
            "Ditemukan label kosong pada dataset canonical."
        )

    counts = {
        label: labels.count(label)
        for label in sorted(set(labels))
    }

    expected_labels = {
        str(x).lower()
        for x in SENTIMENT_CLASSES
    }

    assert_equal(
        "dataset.classes",
        set(counts),
        expected_labels,
    )

    assert_equal(
        "dataset_rows",
        len(rows),
        EXPECTED_DATASET_ROWS,
    )

    expected_counts = {
        str(k).lower(): int(v)
        for k, v in EXPECTED_CLASS_DISTRIBUTION.items()
    }

    assert_equal(
        "class_distribution",
        counts,
        expected_counts,
    )

    expert_count = sum(
        1
        for row in rows
        if str(row.get("sentiment_label_expert", "")).strip()
    )

    assert_equal(
        "expert_validated_rows",
        expert_count,
        EXPECTED_EXPERT_VALIDATED_ROWS,
    )

    print(f"[OK] Dataset delimiter       = {delimiter!r}")
    print(f"[OK] Dataset rows            = {len(rows)}")
    print(f"[OK] Class distribution      = {counts}")
    print(f"[OK] Expert validated rows  = {expert_count}")

    return len(rows), counts, expert_count


# ---------------------------------------------------------------------------
# LOCKED CONFIGURATION VALIDATION
# ---------------------------------------------------------------------------

def get_locked_tfidf() -> dict[str, Any]:
    tfidf = FINAL_MODEL_CONFIG.get("tfidf")

    if not isinstance(tfidf, dict):
        raise ValueError(
            "FINAL_MODEL_CONFIG['tfidf'] tidak valid."
        )

    required = {
        "max_features",
        "ngram_range",
        "sublinear_tf",
        "min_df",
        "max_df",
    }

    missing = required - set(tfidf)
    if missing:
        raise ValueError(
            f"TF-IDF pada final_config.py kekurangan parameter: "
            f"{sorted(missing)}"
        )

    return {
        "max_features": int(tfidf["max_features"]),
        "ngram_range": normalize_ngram(tfidf["ngram_range"]),
        "sublinear_tf": bool(tfidf["sublinear_tf"]),
        "min_df": int(tfidf["min_df"]),
        "max_df": float(tfidf["max_df"]),
    }


def get_locked_svm() -> dict[str, Any]:
    svm = FINAL_MODEL_CONFIG.get("svm")

    if not isinstance(svm, dict):
        raise ValueError(
            "FINAL_MODEL_CONFIG['svm'] tidak valid."
        )

    required = {
        "kernel",
        "C",
        "gamma",
        "class_weight",
    }

    missing = required - set(svm)
    if missing:
        raise ValueError(
            f"SVM pada final_config.py kekurangan parameter: "
            f"{sorted(missing)}"
        )

    return {
        "kernel": str(svm["kernel"]),
        "C": float(svm["C"]),
        "gamma": svm["gamma"],
        "class_weight": svm["class_weight"],
    }


def get_locked_imbalance() -> str:
    imbalance = FINAL_MODEL_CONFIG.get("imbalance")

    if not isinstance(imbalance, dict):
        raise ValueError(
            "FINAL_MODEL_CONFIG['imbalance'] tidak valid."
        )

    method = imbalance.get("method")
    if method is None:
        raise ValueError(
            "Parameter imbalance.method tidak ditemukan."
        )

    return str(method).strip().lower()


def get_locked_multiclass() -> str:
    multiclass = FINAL_MODEL_CONFIG.get("multiclass")

    if not isinstance(multiclass, dict):
        raise ValueError(
            "FINAL_MODEL_CONFIG['multiclass'] tidak valid."
        )

    strategy = multiclass.get("strategy")
    if strategy is None:
        raise ValueError(
            "Parameter multiclass.strategy tidak ditemukan."
        )

    return normalize_strategy(strategy)


# ---------------------------------------------------------------------------
# EXPERIMENT MATCHING
# ---------------------------------------------------------------------------

def match_tfidf(row: dict[str, Any], locked: dict[str, Any]) -> bool:
    try:
        return (
            int(row["max_features"])
            == int(locked["max_features"])
            and normalize_ngram(row["ngram_range"])
            == locked["ngram_range"]
            and bool(row["sublinear_tf"])
            == bool(locked["sublinear_tf"])
            and int(row["min_df"])
            == int(locked["min_df"])
            and abs(
                float(row["max_df"])
                - float(locked["max_df"])
            ) <= 1e-9
        )
    except (KeyError, TypeError, ValueError):
        return False


def match_svm(row: dict[str, Any], locked: dict[str, Any]) -> bool:
    try:
        return (
            str(row["kernel"]).strip().lower()
            == str(locked["kernel"]).strip().lower()
            and abs(
                float(row["C"])
                - float(locked["C"])
            ) <= 1e-9
            and str(row["gamma"]).strip().lower()
            == str(locked["gamma"]).strip().lower()
        )
    except (KeyError, TypeError, ValueError):
        return False


def find_matching_row(
    experiment: dict[str, Any],
    matcher,
    locked: dict[str, Any] | str,
) -> tuple[dict[str, Any], float | None, float | None]:
    rows = get_result_rows(experiment)

    matches = [
        row
        for row in rows
        if matcher(row, locked)
    ]

    if not matches:
        raise ValueError(
            "Konfigurasi yang dikunci di shared/final_config.py "
            "tidak ditemukan di artefak eksperimen."
        )

    if len(matches) > 1:
        raise ValueError(
            "Ditemukan lebih dari satu hasil eksperimen dengan "
            "konfigurasi final yang sama. Artefak eksperimen perlu "
            "dibersihkan agar sumber hasil tidak ambigu."
        )

    mean, std = score(matches[0])

    return matches[0], mean, std


def validate_exp3(
    exp3: dict[str, Any],
    locked_tfidf: dict[str, Any],
) -> tuple[float | None, float | None]:
    row, mean, std = find_matching_row(
        exp3,
        match_tfidf,
        locked_tfidf,
    )

    print(
        "[OK] Experiment 3 cocok dengan final_config.py:"
        f" TF-IDF={locked_tfidf}"
    )

    if mean is not None:
        print(
            f"     CV Macro F1 = {mean:.4f}"
            + (
                f" +/- {std:.4f}"
                if std is not None
                else ""
            )
        )

    return mean, std


def validate_exp4(
    exp4: dict[str, Any],
    locked_svm: dict[str, Any],
) -> tuple[float | None, float | None]:
    row, mean, std = find_matching_row(
        exp4,
        match_svm,
        locked_svm,
    )

    print(
        "[OK] Experiment 4 cocok dengan final_config.py:"
        f" SVM={locked_svm}"
    )

    if mean is not None:
        print(
            f"     CV Macro F1 = {mean:.4f}"
            + (
                f" +/- {std:.4f}"
                if std is not None
                else ""
            )
        )

    return mean, std


def validate_exp5(
    exp5: dict[str, Any],
    locked_method: str,
) -> tuple[float | None, float | None]:
    rows = get_result_rows(exp5)

    matches = [
        row
        for row in rows
        if str(row.get("method", "")).strip().lower()
        == locked_method
    ]

    # Beberapa artefak menyimpan locked method di level atas.
    locked = exp5.get("locked_imbalance")
    if isinstance(locked, dict):
        method = str(
            locked.get("method", "")
        ).strip().lower()

        if method == locked_method:
            mean, std = score(locked)
            print(
                "[OK] Experiment 5 locked method cocok:"
                f" {locked_method}"
            )
            return mean, std

    if len(matches) != 1:
        raise ValueError(
            "Experiment 5 tidak memiliki tepat satu hasil "
            f"untuk imbalance method={locked_method!r}."
        )

    mean, std = score(matches[0])

    print(
        "[OK] Experiment 5 cocok dengan final_config.py:"
        f" imbalance={locked_method}"
    )

    return mean, std


def validate_exp6(
    exp6: dict[str, Any],
    locked_strategy: str,
) -> tuple[float | None, float | None]:
    rows = get_result_rows(exp6)

    def row_strategy(row: dict[str, Any]) -> str | None:
        value = row.get(
            "strategy",
            row.get(
                "method",
                row.get("name"),
            ),
        )

        if value is None:
            return None

        try:
            return normalize_strategy(value)
        except ValueError:
            return None

    matches = [
        row
        for row in rows
        if row_strategy(row) == locked_strategy
    ]

    locked = exp6.get("locked_multiclass")
    if not isinstance(locked, dict):
        locked = exp6.get("best")

    if isinstance(locked, dict):
        value = locked.get(
            "strategy",
            locked.get(
                "method",
                locked.get("name"),
            ),
        )

        if value is not None:
            try:
                if normalize_strategy(value) == locked_strategy:
                    mean, std = score(locked)
                    print(
                        "[OK] Experiment 6 locked strategy cocok:"
                        f" {locked_strategy.upper()}"
                    )
                    return mean, std
            except ValueError:
                pass

    if len(matches) != 1:
        raise ValueError(
            "Experiment 6 tidak memiliki tepat satu hasil "
            f"untuk multiclass={locked_strategy!r}."
        )

    mean, std = score(matches[0])

    print(
        "[OK] Experiment 6 cocok dengan final_config.py:"
        f" multiclass={locked_strategy.upper()}"
    )

    return mean, std


def validate_exp7(exp7: dict[str, Any]) -> dict[str, Any]:
    """
    Experiment 7 tetap dicatat sebagai diagnostic.
    Hasil Exp7 TIDAK boleh mengubah parameter final.

    TAHAP 5: struktur artefak Exp7 diubah menjadi:
      * `best` = kandidat dengan CV Macro F1 tertinggi (diagnostik),
      * `recommended_gap_compliant_candidate` = kandidat yang sekaligus
        lolos ambang gap 3% DAN F1-nya >= 90% F1 terbaik,
      * `diagnostic_only` = True, `changes_final_configuration` = False.
    Skrip ini TIDAK memakai nilai C mana pun untuk mengganti konfigurasi.
    """

    result: dict[str, Any] = {
        "available": True,
        "role": "diagnostic_only",
        "diagnostic_only": bool(exp7.get("diagnostic_only", True)),
        "changes_final_configuration": bool(
            exp7.get("changes_final_configuration", False)
        ),
        "selection_status": exp7.get("selection_status"),
    }

    best = exp7.get("best")
    if isinstance(best, dict):
        mean, std = score(best)
        if best.get("C") is not None:
            try:
                result["best_C_by_cv_macro_f1"] = float(best["C"])
            except (TypeError, ValueError):
                pass
        if mean is not None:
            result["best_cv_macro_f1"] = mean
        if std is not None:
            result["best_cv_std"] = std
        if "train_cv_gap" in best:
            try:
                result["best_train_cv_gap"] = float(best["train_cv_gap"])
            except (TypeError, ValueError):
                pass

    recommended = exp7.get("recommended_gap_compliant_candidate")
    if isinstance(recommended, dict):
        try:
            result["gap_compliant_C"] = float(recommended.get("C"))
        except (TypeError, ValueError):
            pass
        mean, _ = score(recommended)
        if mean is not None:
            result["gap_compliant_cv_macro_f1"] = mean
        if "gap_percentage_points" in recommended:
            try:
                result["gap_compliant_gap_pp"] = float(
                    recommended["gap_percentage_points"]
                )
            except (TypeError, ValueError):
                pass

    print(
        "[OK] Experiment 7 dibaca sebagai diagnostic_only "
        "(tidak mengubah konfigurasi final)."
    )

    return result


# ---------------------------------------------------------------------------
# FINAL CONFIG SNAPSHOT
# ---------------------------------------------------------------------------

def build_locked_config(
    dataset_rows: int,
    class_distribution: dict[str, int],
    expert_count: int,
    exp_scores: dict[str, tuple[float | None, float | None]],
    exp7_diag: dict[str, Any],
) -> dict[str, Any]:

    tfidf = get_locked_tfidf()
    svm = get_locked_svm()
    imbalance = get_locked_imbalance()
    multiclass = get_locked_multiclass()

    return {
        "config_version": "locked-from-final-config",
        "locked_at": datetime.now().astimezone().isoformat(),

        "source_of_truth": {
            "file": "shared/final_config.py",
            "role": "single_source_of_truth",
            "selection_performed_here": False,
        },

        "project": {
            "name": "Wayground Sentiment Analysis",
            "method": "TF-IDF + SVM",
        },

        "dataset": {
            "path": str(DATASET_PATH),
            "text_column": TEXT_COLUMN,
            "label_column": LABEL_COLUMN,
            "rows": dataset_rows,
            "classes": [
                str(x).lower()
                for x in SENTIMENT_CLASSES
            ],
            "class_distribution": class_distribution,
            "expert_validated_rows": expert_count,
        },

        "split": {
            "train_size": TRAIN_TEST_SPLIT_RATIO,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratify": STRATIFY,
            "cv_folds": CV_FOLDS,
            "train_rows": EXPECTED_DEVELOPMENT_ROWS,
            "holdout_rows": EXPECTED_HOLDOUT_ROWS,
        },

        "labeling": {
            "method": LABELING_METHOD,
            "base_method": LEXICON_BASE_METHOD,
            "neutral_threshold": NEUTRAL_THRESHOLD,
            "final_label_column": LABEL_COLUMN,
        },

        "model": {
            "tfidf": tfidf,
            "svm": svm,
            "imbalance": {
                "method": imbalance,
            },
            "multiclass": {
                "strategy": multiclass,
                "implementation": (
                    "OneVsOneClassifier"
                    if multiclass == "ovo"
                    else "OneVsRestClassifier"
                ),
            },
        },

        "evaluation": {
            "selection_metric": SELECTION_METRIC,
            "selection_rule": (
                "Selection dilakukan sebelum locking, "
                "di luar lock_final_config.py"
            ),
            "average": "macro",
            "holdout_test_used_for_selection": False,
            "final_holdout_size": EXPECTED_HOLDOUT_ROWS,
            "cv_macro_f1": (
                exp_scores["exp6"][0]
            ),
            "cv_std": (
                exp_scores["exp6"][1]
            ),
        },

        "experiment_verification": {
            "experiment_3_tfidf": {
                "source": str(EXP3_PATH),
                "configuration": tfidf,
                "cv_macro_f1": exp_scores["exp3"][0],
                "cv_std": exp_scores["exp3"][1],
            },
            "experiment_4_svm": {
                "source": str(EXP4_PATH),
                "configuration": svm,
                "cv_macro_f1": exp_scores["exp4"][0],
                "cv_std": exp_scores["exp4"][1],
            },
            "experiment_5_class_imbalance": {
                "source": str(EXP5_PATH),
                "locked_method": imbalance,
                "cv_macro_f1": exp_scores["exp5"][0],
                "cv_std": exp_scores["exp5"][1],
            },
            "experiment_6_multiclass": {
                "source": str(EXP6_PATH),
                "locked_strategy": multiclass,
                "cv_macro_f1": exp_scores["exp6"][0],
                "cv_std": exp_scores["exp6"][1],
            },
            "experiment_7_regularization": {
                "source": str(EXP7_PATH),
                "role": "diagnostic_only",
                "diagnostic_winner": exp7_diag,
            },
        },

        "artifacts": {
            "final_config": str(
                PROJECT_ROOT
                / "shared"
                / "final_config.py"
            ),
            "best_config_snapshot": str(OUTPUT_PATH),
            "final_pipeline": str(
                FINAL_MODEL_DIR / "final_pipeline.joblib"
            ),
            "metadata": str(
                FINAL_MODEL_DIR / "metadata.json"
            ),
            "final_evaluation": str(
                OUTPUT_DIR
                / "final_model_evaluation.json"
            ),
            "classification_report": str(
                OUTPUT_DIR
                / "final_classification_report.json"
            ),
            "confusion_matrix": str(
                OUTPUT_DIR
                / "final_confusion_matrix.csv"
            ),
            "predictions": str(
                OUTPUT_DIR
                / "final_holdout_predictions.csv"
            ),
        },

        "reproducibility": {
            "random_state": RANDOM_STATE,
            "cv_folds": CV_FOLDS,
            "split_ratio": (
                f"{int(TRAIN_TEST_SPLIT_RATIO * 100)}:"
                f"{int(TEST_SIZE * 100)}"
            ),
            "development_rows": EXPECTED_DEVELOPMENT_ROWS,
            "holdout_rows": EXPECTED_HOLDOUT_ROWS,
            "selection_metric": SELECTION_METRIC,
            "holdout_used_during_experiments": False,
        },
    }


def validate_locked_config(
    config: dict[str, Any],
    dataset_rows: int,
) -> None:

    expected_tfidf = get_locked_tfidf()
    expected_svm = get_locked_svm()
    expected_imbalance = get_locked_imbalance()
    expected_multiclass = get_locked_multiclass()

    assert_equal(
        "dataset.rows",
        config["dataset"]["rows"],
        dataset_rows,
    )

    assert_equal(
        "dataset.text_column",
        config["dataset"]["text_column"],
        TEXT_COLUMN,
    )

    assert_equal(
        "dataset.label_column",
        config["dataset"]["label_column"],
        LABEL_COLUMN,
    )

    assert_equal(
        "dataset.classes",
        config["dataset"]["classes"],
        [
            str(x).lower()
            for x in SENTIMENT_CLASSES
        ],
    )

    assert_equal(
        "split.train_rows",
        config["split"]["train_rows"],
        EXPECTED_DEVELOPMENT_ROWS,
    )

    assert_equal(
        "split.holdout_rows",
        config["split"]["holdout_rows"],
        EXPECTED_HOLDOUT_ROWS,
    )

    assert_close(
        "split.train_size",
        config["split"]["train_size"],
        TRAIN_TEST_SPLIT_RATIO,
    )

    assert_close(
        "split.test_size",
        config["split"]["test_size"],
        TEST_SIZE,
    )

    assert_equal(
        "split.random_state",
        config["split"]["random_state"],
        RANDOM_STATE,
    )

    assert_equal(
        "split.stratify",
        config["split"]["stratify"],
        STRATIFY,
    )

    assert_equal(
        "split.cv_folds",
        config["split"]["cv_folds"],
        CV_FOLDS,
    )

    assert_equal(
        "evaluation.holdout_test_used_for_selection",
        config["evaluation"]["holdout_test_used_for_selection"],
        False,
    )

    assert_equal(
        "model.tfidf",
        config["model"]["tfidf"],
        expected_tfidf,
    )

    assert_equal(
        "model.svm.kernel",
        config["model"]["svm"]["kernel"],
        expected_svm["kernel"],
    )

    assert_close(
        "model.svm.C",
        config["model"]["svm"]["C"],
        expected_svm["C"],
    )

    assert_equal(
        "model.svm.gamma",
        config["model"]["svm"]["gamma"],
        expected_svm["gamma"],
    )

    assert_equal(
        "model.svm.class_weight",
        config["model"]["svm"]["class_weight"],
        expected_svm["class_weight"],
    )

    assert_equal(
        "model.imbalance.method",
        config["model"]["imbalance"]["method"],
        expected_imbalance,
    )

    assert_equal(
        "model.multiclass.strategy",
        config["model"]["multiclass"]["strategy"],
        expected_multiclass,
    )

    print("[PASS] Snapshot final configuration valid.")
    print("[PASS] Semua parameter berasal dari shared/final_config.py.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("LOCK FINAL CONFIGURATION - WAYGROUND SENTIMENT ANALYSIS")
    print("=" * 80)

    print(
        "[INFO] Mode        : LOCK + VALIDATE ONLY"
    )
    print(
        "[INFO] Source      : shared/final_config.py"
    )
    print(
        "[INFO] Selection   : TIDAK dilakukan oleh script ini"
    )
    print(
        "[INFO] Training    : TIDAK dilakukan oleh script ini"
    )
    print(
        f"[INFO] Holdout     : {EXPECTED_HOLDOUT_ROWS} rows; prior exposure must be reported"
    )

    from shared.deployment_integrity import verify_selection_contract, evaluation_provenance, sha256
    selected, run = verify_selection_contract()
    if OUTPUT_PATH.exists():
        raise FileExistsError("Locked snapshot exists; do not overwrite a deployment history")

    # 1. Pastikan semua artefak tersedia.
    for path, label in [
        (DATASET_PATH, "Canonical dataset"),
        (EXP3_PATH, "Experiment 3 result"),
        (EXP4_PATH, "Experiment 4 result"),
        (EXP5_PATH, "Experiment 5 result"),
        (EXP6_PATH, "Experiment 6 result"),
        (EXP7_PATH, "Experiment 7 result"),
    ]:
        require_file(path, label)
        print(f"[OK] {label}: {path.name}")

    # 2. Validasi dataset.
    dataset_rows, class_distribution, expert_count = (
        validate_dataset()
    )

    # 3. Ambil parameter FINAL dari final_config.py.
    locked_tfidf = get_locked_tfidf()
    locked_svm = get_locked_svm()
    locked_imbalance = get_locked_imbalance()
    locked_multiclass = get_locked_multiclass()

    print("\n[LOCKED CONFIG FROM shared/final_config.py]")
    print(f"[TF-IDF]     {locked_tfidf}")
    print(f"[SVM]        {locked_svm}")
    print(f"[IMBALANCE]  {locked_imbalance}")
    print(f"[MULTICLASS] {locked_multiclass.upper()}")

    # 4. Baca hasil eksperimen hanya untuk VALIDASI.
    exp3 = load_json(
        EXP3_PATH,
        "Experiment 3 result",
    )
    exp4 = load_json(
        EXP4_PATH,
        "Experiment 4 result",
    )
    exp5 = load_json(
        EXP5_PATH,
        "Experiment 5 result",
    )
    exp6 = load_json(
        EXP6_PATH,
        "Experiment 6 result",
    )
    exp7 = load_json(
        EXP7_PATH,
        "Experiment 7 result",
    )

    print("\n[VERIFY EXPERIMENT ARTIFACTS]")

    exp3_f1, exp3_std = validate_exp3(
        exp3,
        locked_tfidf,
    )

    exp4_f1, exp4_std = validate_exp4(
        exp4,
        locked_svm,
    )

    exp5_f1, exp5_std = validate_exp5(
        exp5,
        locked_imbalance,
    )

    exp6_f1, exp6_std = validate_exp6(
        exp6,
        locked_multiclass,
    )

    exp7_diag = validate_exp7(exp7)

    # 5. Bentuk snapshot konfigurasi.
    exp_scores = {
        "exp3": (exp3_f1, exp3_std),
        "exp4": (exp4_f1, exp4_std),
        "exp5": (exp5_f1, exp5_std),
        "exp6": (exp6_f1, exp6_std),
    }

    config = build_locked_config(
        dataset_rows=dataset_rows,
        class_distribution=class_distribution,
        expert_count=expert_count,
        exp_scores=exp_scores,
        exp7_diag=exp7_diag,
    )

    # 6. Validasi snapshot.
    validate_locked_config(
        config,
        dataset_rows,
    )

    config["evaluation_provenance"] = evaluation_provenance(run)
    config["selection_sha256"] = sha256(EXPERIMENTS_DIR / "selected_configuration.json")
    config["reproducibility"]["holdout_used_during_experiments"] = True
    config["reproducibility"]["holdout_use_note"] = "Prior exposure occurred in invalidated attempts; corrected selection excluded canonical holdout."

    # 7. Simpan snapshot.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 80)
    print("FINAL CONFIGURATION LOCKED")
    print("=" * 80)
    print(f"[TF-IDF]     {config['model']['tfidf']}")
    print(f"[SVM]        {config['model']['svm']}")
    print(f"[MULTI]      {config['model']['multiclass']}")
    print(f"[IMBALANCE]  {config['model']['imbalance']}")

    if exp4_f1 is not None:
        print(
            f"[CV]         Macro F1={exp4_f1:.4f}"
            + (
                f" +/- {exp4_std:.4f}"
                if exp4_std is not None
                else ""
            )
        )

    print(
        f"[DATA]       {dataset_rows} rows | "
        f"development={EXPECTED_DEVELOPMENT_ROWS} | "
        f"holdout={EXPECTED_HOLDOUT_ROWS}"
    )
    print(
        f"[SOURCE]     shared/final_config.py"
    )
    print(
        f"[SAVE]       {OUTPUT_PATH}"
    )
    print(
        "[DONE]       Lock berhasil. "
        "Tidak ada parameter yang dipilih oleh script ini."
    )


if __name__ == "__main__":
    main()
