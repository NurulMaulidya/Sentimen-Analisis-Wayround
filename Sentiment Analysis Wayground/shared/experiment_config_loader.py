"""
Experiment Configuration Loader - Wayground Sentiment Analysis
=============================================================

Sumber konfigurasi:
    Experiment 3 -> TF-IDF
    Experiment 4 -> SVM kernel/C/gamma
    Experiment 5 -> class imbalance
    Experiment 6 -> multiclass strategy

Prinsip:
    - Tidak membaca best_config.json sebagai sumber pemilihan.
    - Tidak mengunci parameter secara manual.
    - Mengambil konfigurasi dari artefak eksperimen aktual.
    - Memvalidasi konsistensi artifact chain.
    - Experiment 7 hanya digunakan sebagai informasi tambahan.

Catatan Tahap 5 (27 Okt 2025):
    Modul ini sebelumnya ada di scripts/experiment_config_loader.py dan terhapus
    saat refactor Stage 3 (commit 5fc896e) tanpa sengaja, sehingga
    scripts/evaluate_oof_generalization.py, scripts/regularization_comparison.py,
    dan scripts/reproducibility_test.py gagal diimpor. Modul dipulihkan di sini
    (shared/ = lokasi kode bersama) dan jalur keluarannya mengikuti
    shared.constants.EXPERIMENTS_DIR supaya bisa diarahkan ke output_v2/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.constants import EXPERIMENTS_DIR  # noqa: E402


# ============================================================
# EXPERIMENT ARTIFACT CANDIDATES
# ============================================================
# Canonical filenames inside an explicitly selected output namespace.
# A filename suffix is not evidence of dataset/run freshness.

EXP3_CANDIDATES = [
    EXPERIMENTS_DIR / "experiment_3_tfidf_results.json",
]

EXP4_CANDIDATES = [
    EXPERIMENTS_DIR / "experiment_4_svm_results.json",
]

EXP5_CANDIDATES = [
    EXPERIMENTS_DIR / "experiment_5_class_imbalance_results.json",
]

EXP6_CANDIDATES = [
    EXPERIMENTS_DIR / "experiment_6_ovo_ovr_results.json",
]

EXP7_CANDIDATES = [
    EXPERIMENTS_DIR / "experiment_7_regularization_gap_constrained.json",
]


EXPECTED_CLASSES = ["negatif", "netral", "positif"]


# ============================================================
# FILE HELPERS
# ============================================================

def _existing_path(
    candidates: list[Path],
    label: str,
) -> Path:
    """Mengembalikan file artefak pertama yang ditemukan."""
    for path in candidates:
        if path.exists():
            return path

    names = "\n".join(f"    - {path}" for path in candidates)

    raise FileNotFoundError(
        f"\nArtefak {label} tidak ditemukan.\n"
        f"Project root : {PROJECT_ROOT}\n"
        f"Output dir   : {EXPERIMENTS_DIR}\n"
        f"Kandidat:\n{names}\n"
    )


def load_json(
    candidates: list[Path],
    label: str,
) -> tuple[Path, dict[str, Any]]:
    """Membaca satu artefak JSON eksperimen."""
    path = _existing_path(candidates, label)

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON {label} tidak valid: {path}\nError: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(f"{label} harus berupa object JSON: {path}")

    return path, data


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def _normalize_ngram_range(value: Any) -> tuple[int, int]:
    """Normalisasi ngram_range menjadi tuple integer."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"ngram_range harus list/tuple, diterima: {value!r}"
        )

    if len(value) != 2:
        raise ValueError(
            f"ngram_range harus memiliki 2 nilai: {value!r}"
        )

    return (int(value[0]), int(value[1]))


def _normalize_tfidf(config: dict[str, Any]) -> dict[str, Any]:
    """Normalisasi konfigurasi TF-IDF dari artefak eksperimen."""
    required = [
        "max_features",
        "ngram_range",
        "sublinear_tf",
        "min_df",
        "max_df",
    ]

    missing = [key for key in required if key not in config]

    if missing:
        raise ValueError(
            f"Konfigurasi TF-IDF tidak lengkap. Missing: {missing}"
        )

    return {
        "max_features": int(config["max_features"]),
        "ngram_range": _normalize_ngram_range(config["ngram_range"]),
        "sublinear_tf": bool(config["sublinear_tf"]),
        "min_df": int(config["min_df"]),
        "max_df": float(config["max_df"]),
    }


def _normalize_svm(config: dict[str, Any]) -> dict[str, Any]:
    """Normalisasi konfigurasi SVM dari artefak eksperimen."""
    required = ["kernel", "C", "gamma"]

    missing = [key for key in required if key not in config]

    if missing:
        raise ValueError(
            f"Konfigurasi SVM tidak lengkap. Missing: {missing}"
        )

    return {
        "kernel": str(config["kernel"]).lower(),
        "C": float(config["C"]),
        "gamma": config["gamma"],
        "class_weight": config.get("class_weight", None),
    }


# ============================================================
# ARTIFACT CHAIN VALIDATION
# ============================================================

def _assert_same_tfidf(
    actual: dict[str, Any],
    expected: dict[str, Any],
    source: str,
) -> None:
    """Memastikan konfigurasi TF-IDF antar eksperimen identik."""
    actual_norm = _normalize_tfidf(actual)
    expected_norm = _normalize_tfidf(expected)

    if actual_norm != expected_norm:
        raise ValueError(
            f"\nTF-IDF artifact chain tidak konsisten pada {source}.\n\n"
            f"Expected:\n  {actual_norm}\n\n"
            f"Actual:\n  {actual_norm}\n"
        )


def _assert_same_svm(
    actual: dict[str, Any],
    expected: dict[str, Any],
    source: str,
) -> None:
    """Memastikan konfigurasi SVM antar eksperimen identik."""
    actual_norm = _normalize_svm(actual)
    expected_norm = _normalize_svm(expected)

    if (
        actual_norm["kernel"] != expected_norm["kernel"]
        or actual_norm["C"] != expected_norm["C"]
        or actual_norm["gamma"] != expected_norm["gamma"]
    ):
        raise ValueError(
            f"\nSVM artifact chain tidak konsisten pada {source}.\n\n"
            f"Expected:\n  {expected_norm}\n\n"
            f"Actual:\n  {actual_norm}\n"
        )


# ============================================================
# EXPERIMENT CHAIN LOADER
# ============================================================

def load_experiment_chain(
    include_exp7: bool = False,
) -> dict[str, Any]:
    """Membaca dan menggabungkan hasil Experiment 3-6."""
    exp3_path, exp3 = load_json(EXP3_CANDIDATES, "Experiment 3")
    exp4_path, exp4 = load_json(EXP4_CANDIDATES, "Experiment 4")
    exp5_path, exp5 = load_json(EXP5_CANDIDATES, "Experiment 5")
    exp6_path, exp6 = load_json(EXP6_CANDIDATES, "Experiment 6")

    exp7_path = None
    exp7 = None

    if include_exp7:
        exp7_path, exp7 = load_json(EXP7_CANDIDATES, "Experiment 7")

    # --------------------------------------------------------
    # EXPERIMENT 3 - TF-IDF
    # --------------------------------------------------------
    tfidf = (
        exp3.get("locked_tfidf")
        or exp3.get("best_configuration")
        or exp3.get("best")
    )

    if not isinstance(tfidf, dict):
        raise ValueError(
            "Experiment 3 tidak memiliki "
            "locked_tfidf/best_configuration/best."
        )

    tfidf_config = _normalize_tfidf(tfidf)

    exp3_score = float(
        tfidf.get("cv_f1_mean", tfidf.get("cv_mean_f1_macro", 0.0))
    )
    exp3_std = float(
        tfidf.get("cv_f1_std", tfidf.get("cv_std_f1_macro", 0.0))
    )

    # --------------------------------------------------------
    # EXPERIMENT 4 - SVM
    # --------------------------------------------------------
    svm = exp4.get("best")

    if not isinstance(svm, dict):
        raise ValueError("Experiment 4 tidak memiliki object 'best'.")

    svm_config = _normalize_svm(svm)

    exp4_score = float(svm.get("cv_f1_mean", 0.0))
    exp4_std = float(svm.get("cv_f1_std", 0.0))

    # --------------------------------------------------------
    # EXPERIMENT 5 - CLASS IMBALANCE
    # --------------------------------------------------------
    imbalance = exp5.get("locked_imbalance") or exp5.get("best")

    if not isinstance(imbalance, dict) or "method" not in imbalance:
        raise ValueError(
            "Experiment 5 tidak memiliki "
            "locked_imbalance/best yang valid."
        )

    imbalance_method = str(imbalance["method"]).lower()

    # --------------------------------------------------------
    # EXPERIMENT 6 - MULTICLASS
    # --------------------------------------------------------
    locked_multi = exp6.get("locked_multiclass", {})

    if not isinstance(locked_multi, dict):
        locked_multi = {}

    strategy = str(locked_multi.get("strategy", "")).lower()

    if strategy not in {"ovo", "ovr"}:
        winner = exp6.get("winner_by_cv_f1_macro")

        if isinstance(winner, str):
            strategy = winner.lower()

    if strategy not in {"ovo", "ovr"}:
        raise ValueError(
            f"Strategi multiclass Exp6 tidak valid: {strategy!r}"
        )

    exp6_score = float(locked_multi.get("mean_f1_macro", 0.0))
    exp6_std = float(locked_multi.get("std_f1_macro", 0.0))

    # --------------------------------------------------------
    # VALIDATE EXP3 -> EXP4
    # --------------------------------------------------------
    exp4_tfidf = exp4.get("tfidf_locked", {})

    if exp4_tfidf:
        _assert_same_tfidf(tfidf_config, exp4_tfidf, "Exp4.tfidf_locked")

    # --------------------------------------------------------
    # VALIDATE EXP3 -> EXP5
    # --------------------------------------------------------
    exp5_tfidf = exp5.get("experiment_3_tfidf") or exp5.get("artifact_chain", {}).get("experiment_3_tfidf", {})

    if exp5_tfidf:
        _assert_same_tfidf(tfidf_config, exp5_tfidf, "Exp5.experiment_3_tfidf")

    # --------------------------------------------------------
    # VALIDATE EXP4 -> EXP5
    # --------------------------------------------------------
    exp5_svm = exp5.get("experiment_4_svm") or exp5.get("artifact_chain", {}).get("experiment_4_svm", {})

    if exp5_svm:
        _assert_same_svm(svm_config, exp5_svm, "Exp5.experiment_4_svm")

    # --------------------------------------------------------
    # VALIDATE EXP6 ARTIFACT CHAIN
    # --------------------------------------------------------
    artifact_chain = exp6.get("artifact_chain", {})

    if not isinstance(artifact_chain, dict):
        artifact_chain = {}

    chain_tfidf = artifact_chain.get("tfidf")

    if chain_tfidf:
        _assert_same_tfidf(tfidf_config, chain_tfidf, "Exp6.artifact_chain.tfidf")

    chain_svm = artifact_chain.get("svm")

    if chain_svm:
        _assert_same_svm(svm_config, chain_svm, "Exp6.artifact_chain.svm")

    chain_imbalance = artifact_chain.get("imbalance")

    if chain_imbalance:
        chain_method = str(chain_imbalance.get("method", "")).lower()

        if chain_method and chain_method != imbalance_method:
            raise ValueError(
                "\nArtifact chain Exp6 mismatch imbalance.\n"
                f"Exp5 = {imbalance_method}\n"
                f"Exp6 = {chain_method}\n"
            )

    return {
        "exp3": {
            "path": exp3_path,
            "data": exp3,
            "tfidf": tfidf_config,
            "cv_f1": exp3_score,
            "cv_std": exp3_std,
        },
        "exp4": {
            "path": exp4_path,
            "data": exp4,
            "svm": svm_config,
            "cv_f1": exp4_score,
            "cv_std": exp4_std,
        },
        "exp5": {
            "path": exp5_path,
            "data": exp5,
            "method": imbalance_method,
        },
        "exp6": {
            "path": exp6_path,
            "data": exp6,
            "strategy": strategy,
            "cv_f1": exp6_score,
            "cv_std": exp6_std,
        },
        "exp7": (
            {
                "path": exp7_path,
                "data": exp7,
            }
            if include_exp7
            else None
        ),
        "final": {
            "tfidf": tfidf_config,
            "svm": svm_config,
            "imbalance": imbalance_method,
            "multiclass": strategy,
        },
    }


# ============================================================
# PRINT CONFIGURATION
# ============================================================

def print_chain(chain: dict[str, Any]) -> None:
    """Menampilkan konfigurasi final hasil artifact chain."""
    final = chain["final"]

    print()
    print("=" * 72)
    print("EXPERIMENT-BASED CONFIGURATION")
    print("=" * 72)

    print()
    print("[PROJECT]")
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Output dir   : {EXPERIMENTS_DIR}")

    print()
    print("[EXPERIMENT 3 - TF-IDF]")
    print(f"  Source       : {chain['exp3']['path'].name}")
    print(f"  CV Macro F1  : {chain['exp3']['cv_f1']:.6f}")
    print(f"  CV Std       : {chain['exp3']['cv_std']:.6f}")
    print(f"  Configuration: {final['tfidf']}")

    print()
    print("[EXPERIMENT 4 - SVM]")
    print(f"  Source       : {chain['exp4']['path'].name}")
    print(f"  CV Macro F1  : {chain['exp4']['cv_f1']:.6f}")
    print(f"  CV Std       : {chain['exp4']['cv_std']:.6f}")
    print(f"  Configuration: {final['svm']}")

    print()
    print("[EXPERIMENT 5 - CLASS IMBALANCE]")
    print(f"  Source       : {chain['exp5']['path'].name}")
    print(f"  Method       : {final['imbalance']}")

    print()
    print("[EXPERIMENT 6 - MULTICLASS]")
    print(f"  Source       : {chain['exp6']['path'].name}")
    print(f"  CV Macro F1  : {chain['exp6']['cv_f1']:.6f}")
    print(f"  CV Std       : {chain['exp6']['cv_std']:.6f}")
    print(f"  Strategy     : {final['multiclass'].upper()}")

    if chain.get("exp7") is not None:
        print()
        print("[EXPERIMENT 7 - REGULARIZATION]")
        print(f"  Source       : {chain['exp7']['path'].name}")

    print()
    print("=" * 72)
    print("FINAL EXPERIMENT-BASED CONFIGURATION")
    print("=" * 72)

    print()
    print(f"TF-IDF      : {final['tfidf']}")
    print(f"SVM         : {final['svm']}")
    print(f"Imbalance   : {final['imbalance']}")
    print(f"Multiclass  : {final['multiclass'].upper()}")

    print()
    print("[OK] Artifact chain berhasil divalidasi.")
    print("=" * 72)


def main() -> None:
    """Entry point untuk menjalankan loader secara langsung."""
    print()
    print("=" * 72)
    print("WAYGROUND SENTIMENT ANALYSIS")
    print("EXPERIMENT CONFIGURATION LOADER")
    print("=" * 72)

    chain = load_experiment_chain(include_exp7=True)
    print_chain(chain)


if __name__ == "__main__":
    main()
