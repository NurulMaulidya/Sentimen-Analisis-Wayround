"""
Evaluasi Holdout per Subset + Interval Kepercayaan Bootstrap (Tahap 5)
=====================================================================

Tujuan:
- Melaporkan metrik pada (a) SELURUH holdout, (b) subset berlabel PAKAR
  saja, dan (c) holdout berlabel LEXICON saja.
- Menyertakan interval kepercayaan 95% (bootstrap) untuk setiap metrik.
- TIDAK melatih ulang model dan TIDAK memakai holdout untuk memilih
  konfigurasi apa pun. Prediksi diambil dari
  `final_holdout_predictions.csv` yang sudah dihasilkan oleh
  `train_final_model.py` (holdout dievaluasi satu kali).

Catatan metodologis:
- Bootstrap: resampling dengan pengembalian, 2.000 iterasi, dirandom
  dengan random_state=42 agar dapat direproduksi.
- Pada subset kecil (mis. label imperio), interval bootstrap lebih lebar;
  ini dicatat apa adanya, bukan disembunyikan.

Jalankan:
    python scripts/evaluate_holdout_subsets.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.constants import (  # noqa: E402
    CANONICAL_DATASET_PATH,
    EXPERIMENTS_DIR,
    FINAL_MODEL_DIR,
)

PREDICTIONS_PATH = EXPERIMENTS_DIR / "final_holdout_predictions.csv"
OUTPUT_JSON = EXPERIMENTS_DIR / "final_holdout_subset_evaluation.json"

BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 42

CLASSES = ["negatif", "netral", "positif"]


# ---------------------------------------------------------------------------
# METRIK
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Menghitung metrik multiclass tanpa ketergantungan sklearn."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    total = int(len(y_true))

    if total == 0:
        return {
            "n": 0,
            "accuracy": float("nan"),
            "f1_macro": float("nan"),
            "f1_weighted": float("nan"),
            "precision_macro": float("nan"),
            "recall_macro": float("nan"),
        }

    return {
        "n": total,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=CLASSES, average="macro",
                     zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, labels=CLASSES, average="weighted",
                     zero_division=0)
        ),
        "precision_macro": float(
            precision_score(y_true, y_pred, labels=CLASSES, average="macro",
                            zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, labels=CLASSES, average="macro",
                         zero_division=0)
        ),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict[str, float]:
    """
    Interval kepercayaan percentile 95% lewat bootstrap.

    Bootstrap tidak correcting bias, jadi interval bisa sedikit bergeser
    dari interval binomial eksak. Pilihan ini dipakai karena langsung
    berlaku untuk metrik seperti F1-macro.
    """
    n = int(len(y_true))

    if n == 0:
        return {
            "lower": float("nan"),
            "upper": float("nan"),
            "point_estimate": float("nan"),
            "n": 0,
            "iterations": 0,
        }

    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=float)

    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        values[i] = metric(y_true[idx], y_pred[idx])

    lower = float(np.percentile(values, 100 * alpha / 2))
    upper = float(np.percentile(values, 100 * (1 - alpha / 2)))

    return {
        "lower": lower,
        "upper": upper,
        "point_estimate": float(metric(y_true, y_pred)),
        "n": n,
        "iterations": iterations,
        "method": "bootstrap percentile",
    }


def accuracy_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def f1_macro_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(
        f1_score(y_true, y_pred, labels=CLASSES, average="macro",
                 zero_division=0)
    )


# ---------------------------------------------------------------------------
# PEMUATAN DATA
# ---------------------------------------------------------------------------

def read_auto(path: Path) -> pd.DataFrame:
    """Baca CSV dengan deteksi delimiter otomatis."""
    for sep in (",", ";"):
        try:
            return pd.read_csv(path, encoding="utf-8-sig", sep=sep)
        except Exception:
            continue
    raise ValueError(f"Gagal membaca CSV: {path}")


def load_predictions() -> pd.DataFrame:
    """Muat prediksi holdout yang sudah disimpan."""
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediksi holdout tidak ditemukan: {PREDICTIONS_PATH}\n"
            "Jalankan scripts/train_final_model.py terlebih dahulu."
        )
    return read_auto(PREDICTIONS_PATH)


def attach_label_source(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan kolom `label_source` (pakar / lexicon) ke tabel prediksi.

    Sumber label diambil dari dataset canonical lewat `reviewId`.
    """
    canonical = read_auto(CANONICAL_DATASET_PATH)

    if "reviewId" not in canonical.columns:
        raise KeyError(
            f"Dataset canonical tidak punya kolom reviewId: {CANONICAL_DATASET_PATH}"
        )

    source_cols = ["reviewId", "label_source"]
    if "label_source" not in canonical.columns:
        raise KeyError(
            "Dataset canonical tidak punya kolom label_source. "
            "Jalankan ulang pipeline Stage 3.5."
        )

    out = predictions.merge(
        canonical[source_cols],
        on="reviewId",
        how="left",
    )

    missing = int(out["label_source"].isna().sum())
    if missing:
        raise ValueError(
            f"{missing} baris prediksi tidak punya pasangan reviewId "
            "pada dataset canonical."
        )

    out["subset"] = np.where(
        out["label_source"].astype(str).str.contains(
            "expert", case=False
        ),
        "pakar",
        "lexicon",
    )

    return out


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluasi holdout per subset dengan bootstrap CI.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=BOOTSTRAP_ITERATIONS,
        help="Jumlah iterasi bootstrap.",
    )
    args = parser.parse_args()

    print("=" * 78)
    print("EVALUASI HOLDOUT PER SUBSET (Tahap 5)")
    print("=" * 78)

    predictions = attach_label_source(load_predictions())

    actual_col = "actual_sentiment"
    predicted_col = "predicted_sentiment"

    for col in (actual_col, predicted_col):
        if col not in predictions.columns:
            raise KeyError(
                f"Kolom {col!r} tidak ada pada {PREDICTIONS_PATH}. "
                f"Kolom tersedia: {list(predictions.columns)}"
            )

    y_all = predictions[actual_col].astype(str).str.strip().str.lower().to_numpy()
    p_all = predictions[predicted_col].astype(str).str.strip().str.lower().to_numpy()

    subsets = {
        "seluruh_holdout": np.ones(len(predictions), dtype=bool),
        "hanya_pakar": (predictions["subset"] == "pakar").to_numpy(),
        "hanya_lexicon": (predictions["subset"] == "lexicon").to_numpy(),
    }

    results: dict[str, object] = {}

    for name, mask in subsets.items():
        y = y_all[mask]
        p = p_all[mask]

        metrics = compute_metrics(y, p)

        metrics["bootstrap_ci_95"] = {
            "accuracy": bootstrap_ci(
                y, p, accuracy_metric, iterations=args.iterations
            ),
            "f1_macro": bootstrap_ci(
                y, p, f1_macro_metric, iterations=args.iterations
            ),
        }

        metrics["distribusi_actual"] = {
            label: int((y == label).sum()) for label in CLASSES
        }
        metrics["distribusi_prediksi"] = {
            label: int((p == label).sum()) for label in CLASSES
        }

        results[name] = metrics

        print()
        print(f"[{name}]")
        print(f"  n              : {metrics['n']}")
        print(f"  accuracy       : {metrics['accuracy']:.4f}")
        print(f"  f1_macro       : {metrics['f1_macro']:.4f}")
        print(f"  f1_weighted    : {metrics['f1_weighted']:.4f}")
        acc_ci = metrics["bootstrap_ci_95"]["accuracy"]
        f1_ci = metrics["bootstrap_ci_95"]["f1_macro"]
        print(
            f"  CI95 accuracy  : [{acc_ci['lower']:.4f}, {acc_ci['upper']:.4f}]"
        )
        print(
            f"  CI95 f1_macro  : [{f1_ci['lower']:.4f}, {f1_ci['upper']:.4f}]"
        )

    # Kedekatan subset terhadap seluruh holdout.
    print()
    print("[CATATAN]")
    if results["hanya_pakar"]["n"] < 30:
        print(
            "  Subset label pakar kecil "
            f"({results['hanya_pakar']['n']} baris): interval bootstrap "
            "sangat lebar, hasil tidak boleh digeneralisasi."
        )
    if results["hanya_lexikon"]["n"] < 30:
        print(
            "  Subset label lexicon kecil "
            f"({results['hanya_lexikon']['n']} baris): interval bootstrap "
            "sangat lebar."
        )

    payload = {
        "source_predictions": str(PREDICTIONS_PATH),
        "canonical_dataset": str(CANONICAL_DATASET_PATH),
        "final_model_dir": str(FINAL_MODEL_DIR),
        "bootstrap_iterations": args.iterations,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "holdout_used_for_selection": False,
        "catatan": (
            "Holdout dievaluasi satu kali pada model final. Bootstrap hanya "
            "mengukur ketidakpastian estimasi pada prediksi yang sama, "
            "tidak digunakan untuk memilih konfigurasi apa pun."
        ),
        "subsets": results,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print()
    print(f"[OUTPUT] {OUTPUT_JSON}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
