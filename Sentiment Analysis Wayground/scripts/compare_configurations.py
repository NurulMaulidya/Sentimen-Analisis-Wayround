"""
Perbandingan Konfigurasi pada Data Development (Tahap 5)
==========================================================

Tujuan:
- Membandingkan konfigurasi model dengan uji McNemar.
- Menambahkan pembanding opsional:
    * `class_weight='balanced'`
    * n-gram karakter (analisis + word)
    * kernel RBF (vs linear)
    * One-vs-One (vs One-vs-Rest)
- Memeriksa apakah CalibratedClassifierCV menaikkan kalibrasi.

METODOLOGI PENTING:
- Semua perbandingan dilakukan pada DATA DEVELOPMENT dengan prediksi
  out-of-fold (cross_val_predict). Holdout TIDAK DISENTUH di sini, karena
  brief melarang holdout dipakai untuk memilih konfigurasi.
- Uji McNemar membandingkan dua model pada prediksi out-of-fold yang
  SAMA, sehingga pasangan pengamatanaliensmatching (b+c) valid.
- Bootstrap 95% untuk selisih metrik antarkonfigurasi.

Jalankan:
    python scripts/compare_configurations.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.constants import EXPERIMENTS_DIR  # noqa: E402

OUTPUT_JSON = EXPERIMENTS_DIR / "configuration_comparison.json"

RANDOM_STATE = 42
CV_FOLDS = 5
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 42

CLASSES = ["negatif", "netral", "positif"]


# ---------------------------------------------------------------------------
# UTILITAS
# ---------------------------------------------------------------------------

def read_auto(path: Path) -> pd.DataFrame:
    for sep in (",", ";"):
        try:
            return pd.read_csv(path, encoding="utf-8-sig", sep=sep)
        except Exception:
            continue
    raise ValueError(f"Gagal membaca CSV: {path}")


def mcnemar_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> dict[str, Any]:
    """
    Uji McNemar untuk dua prediksi pada himpunan data yang sama.

    b = pred_a benar & pred_b salah
    c = pred_a salah & pred_b benar
    """
    from scipy.stats import binomtest

    a_correct = pred_a == y_true
    b_correct = pred_b == y_true

    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))

    discordant = b + c

    if discordant == 0:
        return {
            "b": b,
            "c": c,
            "discordant_pairs": 0,
            "p_value": 1.0,
            "statistic": 0.0,
            "catatan": (
                "Tidak ada pasangan discordant; kedua model selalu "
                "benar/salah bersamaan pada data yang sama."
            ),
        }

    # Uji binomial eksak dua sisi pada discordant pairs.
    result = binomtest(b, discordant, p=0.5, alternative="two-sided")

    return {
        "b": b,
        "c": c,
        "discordant_pairs": discordant,
        "p_value": float(result.pvalue),
        "statistic": float(result.statistic),
        "metode": "McNemar exact (binomtest)",
        "catatan": (
            f"Konfigurasi A benar pada {b} kasus yang salah diklasifikasikan "
            f"oleh B; B benar pada {c} kasus yang salah diklasifikasikan A."
        ),
    }


def bootstrap_diff_ci(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric_name: str = "accuracy",
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict[str, float]:
    """CI 95% untuk selisih metrik (A - B) dengan bootstrap."""
    from sklearn.metrics import f1_score

    n = int(len(y_true))
    if n == 0:
        return {"lower": float("nan"), "upper": float("nan"),
                "point_estimate": float("nan"), "n": 0}

    def value(
        a: np.ndarray,
        b: np.ndarray,
        truth: np.ndarray,
    ) -> float:
        if metric_name == "f1_macro":
            sa = f1_score(a, truth, labels=CLASSES, average="macro",
                          zero_division=0)
            sb = f1_score(b, truth, labels=CLASSES, average="macro",
                          zero_division=0)
        else:
            sa = float((a == truth).mean())
            sb = float((b == truth).mean())
        return float(sa - sb)

    rng = np.random.default_rng(seed)
    diffs = np.empty(iterations, dtype=float)

    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        # y_true HARUS ikut di-resample. Jika tidak, pasangan
        # (prediksi, label) menjadi tidak sinkron dan interval
        # keystandingannya tidak valid.
        diffs[i] = value(pred_a[idx], pred_b[idx], y_true[idx])

    return {
        "lower": float(np.percentile(diffs, 100 * alpha / 2)),
        "upper": float(np.percentile(diffs, 100 * (1 - alpha / 2))),
        "point_estimate": value(pred_a, pred_b, y_true),
        "n": n,
        "iterations": iterations,
    }


def calibration_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
) -> dict[str, float]:
    """Brier score dan expected calibration error (ECE)."""
    from sklearn.metrics import brier_score_loss

    classes_present = sorted(set(y_true.tolist()))
    onehot = np.zeros_like(proba)
    for i, label in enumerate(classes_present):
        onehot[:, i] = (y_true == label).astype(float)

    brier = float(
        np.mean(np.sum((proba - onehot) ** 2, axis=1))
    )

    # ECE dengan 10 bin.
    confidence = proba.max(axis=1)
    correct = (
        proba.argmax(axis=1) == np.array(
            [classes_present.index(t) for t in y_true]
        )
    ).astype(float)

    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            continue
        ece += (mask.mean()) * abs(correct[mask].mean() - confidence[mask].mean())

    return {
        "brier_score": brier,
        "ece_10_bins": float(ece),
        "mean_confidence": float(confidence.mean()),
        "accuracy": float(correct.mean()),
    }


# ---------------------------------------------------------------------------
# MODEL KONFIGURASI
# ---------------------------------------------------------------------------

def build_configurations(
    locked: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Daftar konfigurasi yang dibandingkan (semua di development).

    TAHAP 5: sebelumnya konfigurasi "final_terkunci" diambil dari artifact
    chain, sehingga ketika Experiment 4 memilih kernel RBF, perbandingan
    "rbf" justru membandingkan RBF dengan RBF (tidak ada pembanding
    linear). Sekarang baseline adalah konfigurasi LINEAR terbaik dari
    Experiment 4, dan semua pembanding (RBF, OvO, class_weight,
    n-gram karakter) diuji dengan C yang sama agar perbandingan
    benar-benar menguji satu variabel.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.svm import SVC

    try:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
    except ImportError:  # pragma: no cover
        ImbPipeline = None
        SMOTE = None

    tfidf_cfg = locked["tfidf"]
    svm_cfg = locked["svm"]

    base_C = float(svm_cfg["C"])

    def base_vectorizer(**kwargs) -> TfidfVectorizer:
        params = {
            "max_features": tfidf_cfg["max_features"],
            "ngram_range": tuple(tfidf_cfg["ngram_range"]),
            "sublinear_tf": tfidf_cfg["sublinear_tf"],
            "min_df": tfidf_cfg["min_df"],
            "max_df": tfidf_cfg["max_df"],
        }
        params.update(kwargs)
        return TfidfVectorizer(**params)

    def base_svm(**kwargs) -> SVC:
        params = {
            "kernel": svm_cfg["kernel"],
            "C": base_C,
            "gamma": svm_cfg["gamma"],
            "class_weight": svm_cfg.get("class_weight"),
        }
        params.update(kwargs)
        return SVC(**params)

    def multiclass(kind: str, **svm_kwargs):
        if kind == "ovo":
            return OneVsOneClassifier(base_svm(**svm_kwargs))
        return OneVsRestClassifier(base_svm(**svm_kwargs))

    strategy = locked.get("multiclass", "ovr")

    configs: list[dict[str, Any]] = [
        {
            "nama": "linear_ovr",
            "deskripsi": (
                "Baseline: kernel linear, One-vs-Rest, C yang sama"
            ),
            "pipeline": lambda: Pipeline([
                ("tfidf", base_vectorizer()),
                ("classifier", multiclass("ovr", kernel="linear")),
            ]),
        },
        {
            "nama": "linear_ovo",
            "deskripsi": "Pembanding: One-vs-One (kernel linear)",
            "pipeline": lambda: Pipeline([
                ("tfidf", base_vectorizer()),
                ("classifier", multiclass("ovo", kernel="linear")),
            ]),
        },
        {
            "nama": "rbf_ovr",
            "deskripsi": "Pembanding: kernel RBF (C sama)",
            "pipeline": lambda: Pipeline([
                ("tfidf", base_vectorizer()),
                ("classifier", multiclass("ovr", kernel="rbf")),
            ]),
        },
        {
            "nama": "class_weight_balanced",
            "deskripsi": "Pembanding opsional: class_weight='balanced'",
            "pipeline": lambda: Pipeline([
                ("tfidf", base_vectorizer()),
                ("classifier", multiclass(
                    "ovr", kernel="linear", class_weight="balanced"
                )),
            ]),
        },
        {
            "nama": "ngram_karakter",
            "deskripsi": (
                "Pembanding opsional: gabungan n-gram kata dan karakter"
            ),
            "pipeline": lambda: Pipeline([
                ("features", FeatureUnion([
                    ("word", base_vectorizer()),
                    ("char", base_vectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2,
                        max_features=3000,
                    )),
                ])),
                ("classifier", multiclass("ovr", kernel="linear")),
            ]),
        },
    ]

    # Pembanding metode ketidakseimbangan kelas (Experiment 5 memakai
    # pipeline imblearn; pipeline sklearn biasa tidak bisa disisipkan
    # sampler di tengah).
    if ImbPipeline is not None and SMOTE is not None:
        configs.append({
            "nama": "smote",
            "deskripsi": (
                "Pembanding: SMOTE pada ruang fitur TF-IDF "
                "(metode pemenang Experiment 5)"
            ),
            "pipeline": lambda: ImbPipeline([
                ("tfidf", base_vectorizer()),
                ("sampler", SMOTE(random_state=42)),
                ("classifier", multiclass("ovr", kernel="linear")),
            ]),
        })

    # Varian kernel RBF dengan C dari pemenang Experiment 4 (bila berbeda).
    if abs(float(svm_cfg.get("C_exp4_winner", base_C)) - base_C) > 1e-12:
        configs.append({
            "nama": "rbf_ovr_C_exp4",
            "deskripsi": (
                "RBF dengan C dari pemenang Experiment 4 "
                f"(C={svm_cfg.get('C_exp4_winner')})"
            ),
            "pipeline": lambda: Pipeline([
                ("tfidf", base_vectorizer()),
                ("classifier", multiclass(
                    "ovr", kernel="rbf",
                    C=float(svm_cfg["C_exp4_winner"]),
                )),
            ]),
        })

    return configs


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bandingkan konfigurasi pada development (out-of-fold).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=BOOTSTRAP_ITERATIONS,
    )
    args = parser.parse_args()

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC

    from shared.splitting import split_parts_for_dataset
    from shared.constants import CANONICAL_DATASET_PATH

    print("=" * 78)
    print("PERBANDINGAN KONFIGURASI - DATA DEVELOPMENT (Tahap 5)")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Konfigurasi terkunci.
    # ------------------------------------------------------------------
    from shared.experiment_config_loader import load_experiment_chain

    chain = load_experiment_chain(include_exp7=True)
    chain_final = chain["final"]
    print(f"Artifact chain (pemenang CV Experiment 3-6): {chain_final}")

    # ------------------------------------------------------------------
    # Baseline perbandingan = konfigurasi LINEAR terbaik dari
    # Experiment 4 (bukan pemenang mentah, yang bisa berupa RBF).
    # Dengan begitu perbandingan RBF vs linear benar-benar menguji
    # satu variabel pada C yang sama.
    # ------------------------------------------------------------------
    exp4_rows = chain["exp4"]["data"].get("results", [])
    linear_rows = [
        row for row in exp4_rows
        if str(row.get("kernel", "")).lower() == "linear"
    ]
    if not linear_rows:
        raise RuntimeError(
            "Experiment 4 tidak memuat hasil kernel linear untuk baseline."
        )
    best_linear = sorted(
        linear_rows,
        key=lambda r: (-float(r["cv_f1_mean"]), float(r["cv_f1_std"]),
                       float(r["C"])),
    )[0]

    exp4_winner = chain["exp4"]["data"].get("best", {})

    locked = {
        "tfidf": chain_final["tfidf"],
        "svm": {
            "kernel": "linear",
            "C": float(best_linear["C"]),
            "gamma": str(best_linear.get("gamma", "scale")),
            "class_weight": None,
            "C_exp4_winner": float(exp4_winner.get("C", best_linear["C"])),
        },
        "imbalance": chain_final["imbalance"],
        "multiclass": "ovr",
    }

    print(f"Baseline perbandingan (linear terbaik dari Exp4): {locked['svm']}")
    print(
        f"Pemenang mentah Experiment 4: kernel="
        f"{exp4_winner.get('kernel')} C={exp4_winner.get('C')} "
        f"CV F1={exp4_winner.get('cv_f1_mean'):.4f}"
    )

    # ------------------------------------------------------------------
    # Data development.
    # ------------------------------------------------------------------
    df = read_auto(CANONICAL_DATASET_PATH)
    df["processed_text"] = df["processed_text"].fillna("").astype(str)
    df["sentiment_label_final"] = (
        df["sentiment_label_final"].astype(str).str.strip().str.lower()
    )

    development, holdout, split_info = split_parts_for_dataset(
        df, label_col="sentiment_label_final",
    )
    print(f"Development : {len(development)} baris")
    print(f"Sumber split: {split_info.get('source')}")

    X = development["processed_text"].to_numpy()
    y = development["sentiment_label_final"].to_numpy()

    groups = development["processed_text"].to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    # ------------------------------------------------------------------
    # Prediksi out-of-fold untuk tiap konfigurasi.
    # ------------------------------------------------------------------
    configs = build_configurations(locked)

    predictions: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}

    for cfg in configs:
        print()
        print(f"[{cfg['nama']}] {cfg['deskripsi']}")
        started = time.time()

        pipeline = cfg["pipeline"]()

        # Terapkan override SVM bila ada.
        if "svm_kwargs" in cfg:
            classifier = pipeline.named_steps["classifier"]
            estimator = classifier.estimator
            for key, value in cfg["svm_kwargs"].items():
                setattr(estimator, key, value)

        pred = cross_val_predict(
            pipeline,
            X,
            y,
            groups=groups,
            cv=splitter,
            n_jobs=1,
        )
        predictions[cfg["nama"]] = pred
        timings[cfg["nama"]] = time.time() - started

        acc = float((pred == y).mean())
        print(f"  OOF accuracy : {acc:.4f}  ({timings[cfg['nama']]:.1f}s)")

    # ------------------------------------------------------------------
    # Perbandingan terhadap konfigurasi terkunci.
    # ------------------------------------------------------------------
    print()
    print("[PERBANDINGAN vs linear_ovr (baseline)]")

    comparisons: dict[str, Any] = {}

    baseline = predictions["linear_ovr"]

    for name, pred in predictions.items():
        if name == "linear_ovr":
            continue

        mcnemar = mcnemar_test(y, pred, baseline)
        diff_acc = bootstrap_diff_ci(
            y, pred, baseline, "accuracy", iterations=args.iterations
        )
        diff_f1 = bootstrap_diff_ci(
            y, pred, baseline, "f1_macro", iterations=args.iterations
        )

        comparisons[name] = {
            "mcnemar": mcnemar,
            "selisih_accuracy_ci95": diff_acc,
            "selisih_f1_macro_ci95": diff_f1,
        }

        print()
        print(f"  {name}")
        print(f"    McNemar b={mcnemar['b']} c={mcnemar['c']} "
              f"p={mcnemar['p_value']:.4f}")
        print(f"    selisih accuracy : "
              f"{diff_acc['point_estimate']:+.4f} "
              f"[{diff_acc['lower']:+.4f}, {diff_acc['upper']:+.4f}]")
        print(f"    selisih f1_macro : "
              f"{diff_f1['point_estimate']:+.4f} "
              f"[{diff_f1['lower']:+.4f}, {diff_f1['upper']:+.4f}]")

    # ------------------------------------------------------------------
    # Kalibrasi: CalibratedClassifierCV hanya dipakai bila menaikkan
    # kalibrasi pada CV development.
    # ------------------------------------------------------------------
    print()
    print("[KALIBRASI]")

    calibration: dict[str, Any] = {}

    base_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=locked["tfidf"]["max_features"],
            ngram_range=tuple(locked["tfidf"]["ngram_range"]),
            sublinear_tf=locked["tfidf"]["sublinear_tf"],
            min_df=locked["tfidf"]["min_df"],
            max_df=locked["tfidf"]["max_df"],
        )),
        ("classifier", OneVsRestClassifier(SVC(
            kernel=locked["svm"]["kernel"],
            C=locked["svm"]["C"],
            gamma=locked["svm"]["gamma"],
            class_weight=locked["svm"].get("class_weight"),
        ))),
    ])

    # Baseline: decision_function sebagai confidence.
    baseline_scores = cross_val_predict(
        base_pipeline,
        X,
        y,
        groups=groups,
        cv=splitter,
        n_jobs=1,
        method="decision_function",
    )

    # Normalisasi margin menjadi "probabilitas" pseudo-konfidensi.
    def margin_to_confidence(scores: np.ndarray) -> np.ndarray:
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    baseline_proba = margin_to_confidence(baseline_scores)

    # Calibrated: sigmoid pada SVC di dalam OvR.
    calibrated_pipeline = Pipeline([
        ("tfidf", base_pipeline.named_steps["tfidf"]),
        ("classifier", CalibratedClassifierCV(
            OneVsRestClassifier(SVC(
                kernel=locked["svm"]["kernel"],
                C=locked["svm"]["C"],
                gamma=locked["svm"]["gamma"],
                class_weight=locked["svm"].get("class_weight"),
            )),
            method="sigmoid",
            cv=3,
        )),
    ])

    calibrated_proba = cross_val_predict(
        calibrated_pipeline,
        X,
        y,
        groups=groups,
        cv=splitter,
        n_jobs=1,
        method="predict_proba",
    )

    base_cal = calibration_metrics(y, baseline_proba)
    cal_cal = calibration_metrics(y, calibrated_proba)

    calibration = {
        "baseline_margin": base_cal,
        "calibrated_sigmoid": cal_cal,
        "mcnemar": mcnemar_test(
            y,
            calibrated_proba.argmax(axis=1),
            baseline_proba.argmax(axis=1),
        ),
        "keputusan": (
            "Menggunakan CalibratedClassifierCV"
            if (
                cal_cal["brier_score"] < base_cal["brier_score"]
                and cal_cal["ece_10_bins"] <= base_cal["ece_10_bins"]
            )
            else "Tetap memakai decision_margin (confidence=None)"
        ),
    }

    print(f"  baseline (margin)  : brier={base_cal['brier_score']:.4f} "
          f"ece={base_cal['ece_10_bins']:.4f}")
    print(f"  calibrated (sigmoid): brier={cal_cal['brier_score']:.4f} "
          f"ece={cal_cal['ece_10_bins']:.4f}")
    print(f"  keputusan          : {calibration['keputusan']}")

    payload = {
        "catatan": (
            "Seluruh perbandingan dilakukan pada data development dengan "
            "prediksi out-of-fold (StratifiedGroupKFold, random_state=42). "
            "Holdout tidak digunakan untuk pemilihan konfigurasi."
        ),
        "development_rows": int(len(development)),
        "split_source": split_info.get("source"),
        "baseline_config": locked,
        "exp4_raw_winner": exp4_winner,
        "cv_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "oof_accuracy": {
            name: float((pred == y).mean())
            for name, pred in predictions.items()
        },
        "runtime_seconds": timings,
        "perbandingan": comparisons,
        "kalibrasi": calibration,
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
