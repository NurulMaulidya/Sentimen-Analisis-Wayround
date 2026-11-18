from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from shared.constants import (  # noqa: E402
    EXPERIMENTS_DIR,
    CANONICAL_DATASET_PATH,
)

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


DATASET_PATH = CANONICAL_DATASET_PATH

# Artifact chain: Exp3 -> TF-IDF, Exp4 -> SVM/kernel,
# Exp5 -> imbalance, Exp6 -> multiclass strategy.
EXP3_RESULT_CANDIDATES = (
    EXPERIMENTS_DIR / "experiment_3_tfidf_results.json",
)
EXP4_RESULT_CANDIDATES = (
    EXPERIMENTS_DIR / "experiment_4_svm_results.json",
)
EXP5_RESULT_CANDIDATES = (
    EXPERIMENTS_DIR / "experiment_5_class_imbalance_results.json",
)
EXP6_RESULT_CANDIDATES = (
    EXPERIMENTS_DIR / "experiment_6_ovo_ovr_results.json",
)

OUTPUT_JSON = EXPERIMENTS_DIR / "experiment_7_regularization_gap_constrained.json"
OUTPUT_CSV = EXPERIMENTS_DIR / "experiment_7_regularization_gap_constrained.csv"

GAP_THRESHOLD = 0.03

# C is the ONLY hyperparameter intentionally changed in Experiment 7.
C_VALUES = (0.01, 0.05, 0.06, 0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0)


def resolve_artifact(candidates: tuple[Path, ...], name: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    joined = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"{name} tidak ditemukan. Jalankan eksperimen sebelumnya terlebih dahulu.\n"
        f"Dicari pada:\n{joined}"
    )


def load_json(path: Path, name: str) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{name} harus berupa object JSON: {path}")
    return data


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        header = f.readline()

    if header.count(",") == 0 and header.count(";") == 0:
        raise ValueError(f"Delimiter CSV tidak dikenali: {path}")

    return "," if header.count(",") >= header.count(";") else ";"


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    df = pd.read_csv(
        DATASET_PATH,
        sep=detect_delimiter(DATASET_PATH),
        encoding="utf-8-sig",
        low_memory=False,
    )

    text_col = "processed_text"
    label_col = "sentiment_label_final"

    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(f"Kolom dataset tidak lengkap: {sorted(missing)}")

    # Kolom `reviewId` dipertahankan agar split dapat diselaraskan dengan
    # artefak Stage 5 (Tahap 5).
    keep_cols = [text_col, label_col]
    if "reviewId" in df.columns:
        keep_cols.append("reviewId")
    df = df[keep_cols].copy()
    df[text_col] = df[text_col].fillna("").astype(str).str.strip()
    df[label_col] = df[label_col].fillna("").astype(str).str.strip().str.lower()
    df = df[(df[text_col] != "") & (df[label_col] != "")].reset_index(drop=True)

    expected_classes = {"negatif", "netral", "positif"}
    if set(df[label_col].unique()) != expected_classes:
        raise ValueError(
            f"Kelas dataset harus {sorted(expected_classes)}, "
            f"ditemukan {sorted(df[label_col].unique())}"
        )

    return df, text_col, label_col


def normalize_ngram(value: Any) -> tuple:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    raise ValueError(f"ngram_range tidak valid: {value!r}")


def _candidate_dicts(data: dict, top_keys: tuple[str, ...], nested_keys: tuple[str, ...]):
    candidates = []

    for key in top_keys:
        value = data.get(key)
        if isinstance(value, dict):
            candidates.append(value)
            for nested_key in nested_keys:
                nested = value.get(nested_key)
                if isinstance(nested, dict):
                    candidates.append(nested)

    return candidates


def extract_tfidf(exp3: dict) -> dict:
    required = {"max_features", "ngram_range", "sublinear_tf", "min_df", "max_df"}

    candidates = _candidate_dicts(
        exp3,
        ("locked_tfidf", "best_configuration", "best"),
        ("tfidf", "best_tfidf", "configuration"),
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
        "Parameter TF-IDF dari Experiment 3 tidak ditemukan "
        "di artifact yang tersedia."
    )


def extract_svm(exp4: dict) -> dict:
    candidates = _candidate_dicts(
        exp4,
        ("locked_svm", "best_configuration", "best"),
        ("svm", "best_svm", "configuration"),
    )

    for candidate in candidates:
        if "kernel" in candidate and "C" in candidate:
            return {
                "kernel": str(candidate["kernel"]).lower(),
                "C": float(candidate["C"]),
                "gamma": candidate.get("gamma", "scale"),
            }

    raise ValueError(
        "Parameter SVM dari Experiment 4 tidak ditemukan "
        "di artifact yang tersedia."
    )


def extract_imbalance(exp5: dict) -> dict:
    candidates = []

    locked = exp5.get("locked_imbalance")
    if isinstance(locked, dict):
        candidates.append(locked)

    best = exp5.get("best")
    if isinstance(best, dict):
        candidates.append(best)

    for candidate in candidates:
        if "method" in candidate:
            return {
                "method": str(candidate["method"]).lower(),
                "cv_macro_f1_mean": float(
                    candidate.get("cv_macro_f1_mean", np.nan)
                ),
                "cv_macro_f1_std": float(
                    candidate.get("cv_macro_f1_std", np.nan)
                ),
            }

    raise ValueError(
        "Metode class imbalance dari Experiment 5 tidak ditemukan."
    )


def extract_multiclass(exp6: dict) -> dict:
    locked = exp6.get("locked_multiclass")
    if isinstance(locked, dict) and "strategy" in locked:
        return {
            "strategy": str(locked["strategy"]).lower(),
            "cv_macro_f1_mean": float(
                locked.get("mean_f1_macro", np.nan)
            ),
            "cv_macro_f1_std": float(
                locked.get("std_f1_macro", np.nan)
            ),
        }

    for key in ("winner_by_cv_f1_macro", "winner", "best"):
        value = exp6.get(key)

        if isinstance(value, str):
            return {"strategy": value.lower()}

        if isinstance(value, dict) and "strategy" in value:
            return {
                "strategy": str(value["strategy"]).lower(),
                "cv_macro_f1_mean": float(
                    value.get("mean_f1_macro", value.get("cv_macro_f1_mean", np.nan))
                ),
                "cv_macro_f1_std": float(
                    value.get("std_f1_macro", value.get("cv_macro_f1_std", np.nan))
                ),
            }

    raise ValueError(
        "Strategi multiclass dari Experiment 6 tidak ditemukan."
    )


def validate_chain(tfidf: dict, svm: dict, imbalance: dict, multiclass: dict):
    if tfidf["max_features"] <= 0:
        raise ValueError("TF-IDF max_features harus > 0.")

    if len(tfidf["ngram_range"]) != 2:
        raise ValueError("TF-IDF ngram_range tidak valid.")

    if svm["kernel"] not in {"linear", "rbf", "poly", "sigmoid"}:
        raise ValueError(f"Kernel SVM tidak valid: {svm['kernel']}")

    if imbalance["method"] not in {
        "none",
        "class_weight_balanced",
        "random_over",
        "random_under",
        "smote",
    }:
        raise ValueError(f"Metode imbalance tidak dikenali: {imbalance['method']}")

    if multiclass["strategy"] not in {"ovo", "ovr"}:
        raise ValueError(
            f"Strategi multiclass tidak valid: {multiclass['strategy']}"
        )

    print("[ARTIFACT CHAIN] PASS")
    print(f"  Exp3 TF-IDF     : {tfidf}")
    print(f"  Exp4 SVM kernel : {svm['kernel']}")
    print(f"  Exp4 C baseline : {svm['C']}  (baseline, bukan nilai yang dipaksakan)")
    print(f"  Exp5 imbalance  : {imbalance['method']}")
    print(f"  Exp6 multiclass : {multiclass['strategy']}")


def build_pipeline(tfidf_cfg: dict, svm_cfg: dict, imbalance_method: str,
                    multiclass_strategy: str, C: float):
    vectorizer = TfidfVectorizer(
        max_features=tfidf_cfg["max_features"],
        ngram_range=tfidf_cfg["ngram_range"],
        sublinear_tf=tfidf_cfg["sublinear_tf"],
        min_df=tfidf_cfg["min_df"],
        max_df=tfidf_cfg["max_df"],
    )

    class_weight = (
        "balanced" if imbalance_method == "class_weight_balanced" else None
    )

    base_svm = SVC(
        kernel=svm_cfg["kernel"],
        C=float(C),
        gamma=svm_cfg.get("gamma", "scale"),
        class_weight=class_weight,
        probability=False,
    )

    if multiclass_strategy == "ovo":
        classifier = OneVsOneClassifier(base_svm)
    else:
        classifier = OneVsRestClassifier(base_svm)

    # No resampling method is inserted here unless Experiment 5 selected it.
    # For resampling methods, use an imblearn pipeline so resampling happens
    # inside each CV training fold.
    if imbalance_method == "none" or imbalance_method == "class_weight_balanced":
        return Pipeline([
            ("tfidf", vectorizer),
            ("classifier", classifier),
        ])

    try:
        from imblearn.over_sampling import RandomOverSampler, SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.pipeline import Pipeline as ImbPipeline
    except ImportError as exc:
        raise ImportError(
            "Experiment 5 memilih metode resampling, tetapi imbalanced-learn "
            "belum terpasang. Install dengan: pip install imbalanced-learn"
        ) from exc

    if imbalance_method == "random_over":
        sampler = RandomOverSampler(random_state=RANDOM_STATE)
    elif imbalance_method == "random_under":
        sampler = RandomUnderSampler(random_state=RANDOM_STATE)
    elif imbalance_method == "smote":
        sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    else:
        raise ValueError(f"Metode imbalance tidak didukung: {imbalance_method}")

    return ImbPipeline([
        ("tfidf", vectorizer),
        ("sampler", sampler),
        ("classifier", classifier),
    ])


def evaluate(pipe, X, y, cv, groups=None):
    scores = cross_val_score(
        pipe,
        X,
        y,
        cv=cv,
        groups=groups,
        scoring="f1_macro",
        n_jobs=1,
    )

    # Train F1 is diagnostic only. It is NOT the primary selection metric.
    pipe.fit(X, y)
    train_f1 = f1_score(
        y,
        pipe.predict(X),
        average="macro",
        zero_division=0,
    )

    cv_f1 = float(scores.mean())
    gap = float(train_f1 - cv_f1)

    return (
        float(train_f1),
        cv_f1,
        float(scores.std(ddof=1)),
        gap,
        [float(x) for x in scores],
    )


def main():
    start = time.perf_counter()

    print("=" * 80)
    print("EXPERIMENT 7 - SVM REGULARIZATION / GAP-CONSTRAINED GENERALIZATION")
    print("=" * 80)
    print(
        "Tujuan: mengevaluasi C dengan konfigurasi Exp3-Exp6 tetap terkunci "
        "dan holdout tidak digunakan."
    )
    print(
        "Catatan: gap < 3% adalah constraint analitis, bukan alasan untuk "
        "mengabaikan CV Macro F1."
    )

    exp3_path = resolve_artifact(EXP3_RESULT_CANDIDATES, "Experiment 3")
    exp4_path = resolve_artifact(EXP4_RESULT_CANDIDATES, "Experiment 4")
    exp5_path = resolve_artifact(EXP5_RESULT_CANDIDATES, "Experiment 5")
    exp6_path = resolve_artifact(EXP6_RESULT_CANDIDATES, "Experiment 6")

    exp3 = load_json(exp3_path, "Experiment 3")
    exp4 = load_json(exp4_path, "Experiment 4")
    exp5 = load_json(exp5_path, "Experiment 5")
    exp6 = load_json(exp6_path, "Experiment 6")

    tfidf_cfg = extract_tfidf(exp3)
    svm_cfg = extract_svm(exp4)
    imbalance_cfg = extract_imbalance(exp5)
    multiclass_cfg = extract_multiclass(exp6)

    validate_chain(
        tfidf_cfg,
        svm_cfg,
        imbalance_cfg,
        multiclass_cfg,
    )

    df, text_col, label_col = load_dataset()

    # TAHAP 5: sebelumnya memakai `train_test_split` per baris, sehingga
    # "development" Experiment 7 (2.465 baris) BERBEDA dari development
    # Experiment 3-6 (2.470 baris) dan kebocoran `processed_text` kembali
    # muncul. Sekarang split diambil dari artefak Stage 5 (berkelompok).
    from shared.splitting import build_groups, split_parts_for_dataset

    development, holdout, split_info = split_parts_for_dataset(
        df,
        group_col=GROUP_COLUMN,
        label_col=label_col,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    X_dev = development[text_col].astype(str)
    y_dev = development[label_col].astype(str)
    groups_dev = build_groups(development, GROUP_COLUMN).to_numpy()

    del holdout

    # TAHAP 5: CV berkelompok (grup = processed_text).
    cv = make_grouped_cv(CV_FOLDS)

    print(
        "Sumber split  : "
        f"{split_info.get('source', '-')} ({split_info.get('path', '-')})"
    )

    print(f"Dataset       : {len(df)}")
    print(f"Development   : {len(X_dev)}")
    print("Holdout       : NOT USED")
    print(f"TF-IDF        : {tfidf_cfg}")
    print(f"SVM kernel    : {svm_cfg['kernel']}")
    print(f"SVM gamma     : {svm_cfg.get('gamma', 'scale')}")
    print(f"Exp4 C        : {svm_cfg['C']} (reference only)")
    print(f"Multiclass    : {multiclass_cfg['strategy'].upper()}")
    print(f"Imbalance     : {imbalance_cfg['method']}")
    print(f"Gap threshold : < {GAP_THRESHOLD:.0%}")
    print(f"C candidates  : {list(C_VALUES)}")

    results = []

    for i, C in enumerate(C_VALUES, 1):
        pipe = build_pipeline(
            tfidf_cfg=tfidf_cfg,
            svm_cfg=svm_cfg,
            imbalance_method=imbalance_cfg["method"],
            multiclass_strategy=multiclass_cfg["strategy"],
            C=C,
        )

        train_f1, cv_f1, cv_std, gap, cv_scores = evaluate(
            pipe,
            X_dev,
            y_dev,
            cv,
            groups=groups_dev,
        )

        result = {
            "C": float(C),
            "train_f1": train_f1,
            "cv_f1_mean": cv_f1,
            "cv_f1_std": cv_std,
            "train_cv_gap": gap,
            "gap_percentage_points": gap * 100,
            "gap_pass": bool(gap < GAP_THRESHOLD),
            "cv_f1_scores": cv_scores,
        }

        results.append(result)

        print("-" * 80)
        print(f"[{i}/{len(C_VALUES)}] C={C}")
        print(f"Train F1      : {train_f1:.4f}")
        print(f"CV Macro F1   : {cv_f1:.4f} +/- {cv_std:.4f}")
        print(
            f"Gap           : {gap:.4f} ({gap * 100:.2f} pp) -> "
            f"{'PASS' if gap < GAP_THRESHOLD else 'FAIL'}"
        )

    # ------------------------------------------------------------------
    # SELEKSI (TAHAP 5)
    # ------------------------------------------------------------------
    # Kriteria utama tetap CV Macro F1.
    #
    # TAHAP 5 - PERBAIKAN SEMANTIK: sebelumnya kandidat yang memenuhi
    # gap < 3% dipilih apa adanya, sehingga C=0.01 (F1 = 0,2679 - model
    # yang hampir tidak bisa membedakan kelas) Selected C terambil.
    # Ambang gap adalah DIAGNOSTIK, bukan syarat yang boleh mengalahkan
    # kualitas prediksi: gap kecil pada model yang tidak belajar apa pun
    # bukan generalisasi yang baik.
    #
    # Aturan baru:
    #   1. Kandidat utama = CV Macro F1 tertinggi (diagnostik gap dilaporkan).
    #   2. Kandidat "gap-compliant near-best" = yang lolos gap DAN F1-nya
    #      tidak jauh dari kandidat utama (>= 90% dari F1 utama).
    #   3. Status dicatat apa adanya; Experiment 7 TIDAK mengganti
    #      konfigurasi final secara otomatis.
    best = sorted(
        results,
        key=lambda r: (
            -r["cv_f1_mean"],
            r["cv_f1_std"],
            r["C"],
        ),
    )[0]

    near_best_floor = 0.90 * best["cv_f1_mean"]
    gap_compliant_near_best = [
        r
        for r in results
        if r["gap_pass"] and r["cv_f1_mean"] >= near_best_floor
    ]

    if best["gap_pass"]:
        selection_status = "CV_BEST_PASSES_GAP"
    elif gap_compliant_near_best:
        selection_status = "CV_BEST_FAILS_GAP_NEAR_BEST_CANDIDATE_AVAILABLE"
    else:
        selection_status = "NO_GAP_COMPLIANT_CANDIDATE_WITH_COMPETITIVE_F1"

    recommended = sorted(
        gap_compliant_near_best,
        key=lambda r: (
            -r["cv_f1_mean"],
            r["train_cv_gap"],
            r["C"],
        ),
    )[0] if gap_compliant_near_best else None

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(
        f"{'C':>7} {'Train':>9} {'CV F1':>9} "
        f"{'Std':>9} {'Gap':>9} {'Status':>9}"
    )

    for r in results:
        print(
            f"{r['C']:>7.2f} "
            f"{r['train_f1']:>9.4f} "
            f"{r['cv_f1_mean']:>9.4f} "
            f"{r['cv_f1_std']:>9.4f} "
            f"{r['gap_percentage_points']:>8.2f}% "
            f"{'PASS' if r['gap_pass'] else 'FAIL':>9}"
        )

    print("\nSTATUS              :", selection_status)
    print("BEST BY CV MACRO F1 :", best["C"])
    print("CV Macro F1         :", f"{best['cv_f1_mean']:.4f}")
    print("Train Macro F1      :", f"{best['train_f1']:.4f}")
    print("Train-CV Gap        :", f"{best['gap_percentage_points']:.2f} pp")
    print("Gap < 3%            :", "PASS" if best["gap_pass"] else "FAIL")

    if recommended is not None:
        print(
            "Kandidat gap-lolos "
            "terdekat : C="
            f"{recommended['C']} "
            f"(CV F1={recommended['cv_f1_mean']:.4f}, "
            f"gap={recommended['gap_percentage_points']:.2f} pp)"
        )
    else:
        print(
            "Kandidat gap-lolos terdekat : TIDAK ADA "
            "(tidak ada kandidat dengan F1 >= 90% dari F1 terbaik "
            "yang juga lolos gap < 3%)"
        )

    print(
        "CATATAN: Experiment 7 bersifat DIAGNOSTIK. Nilai C final tetap "
        "mengikuti Experiment 4 dan TIDAK diubah oleh skrip ini."
    )

    output = {
        "experiment": "experiment_7_regularization_gap_constrained",
        "version": "FINAL",
        "gap_threshold": GAP_THRESHOLD,
        "selection_status": selection_status,
        "holdout_used_for_selection": False,
        "selection_metric": "cv_macro_f1_mean",
        "selection_constraint": (
            "Diagnostik: kandidat utama tetap CV Macro F1 tertinggi. "
            "Kandidat gap-compliant (gap < 0.03) dengan F1 >= 90% dari "
            "F1 utama dilaporkan sebagai alternatif."
        ),
        "diagnostic_only": True,
        "changes_final_configuration": False,
        "recommended_gap_compliant_candidate": recommended,
        "near_best_f1_floor": near_best_floor,
        "only_changed_parameter": "C",
        "artifact_chain": {
            "experiment_3_source": str(exp3_path),
            "experiment_4_source": str(exp4_path),
            "experiment_5_source": str(exp5_path),
            "experiment_6_source": str(exp6_path),
            "tfidf": tfidf_cfg,
            "svm_from_experiment_4": svm_cfg,
            "imbalance_from_experiment_5": imbalance_cfg,
            "multiclass_from_experiment_6": multiclass_cfg,
            "artifact_validation": "PASS",
        },
        "locked_base_configuration": {
            "kernel": svm_cfg["kernel"],
            "gamma": svm_cfg.get("gamma", "scale"),
            "multiclass": multiclass_cfg["strategy"],
            "imbalance": imbalance_cfg["method"],
            "tfidf": tfidf_cfg,
        },
        "reference_C_from_experiment_4": svm_cfg["C"],
        "C_values": list(C_VALUES),
        "best": best,
        "results": results,
        "runtime_seconds": round(time.perf_counter() - start, 4),
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pd.DataFrame(results).to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nJSON:", OUTPUT_JSON)
    print("CSV :", OUTPUT_CSV)
    print("\n[DONE] Experiment 7 selesai.")
    print("Holdout tetap tidak digunakan untuk seleksi.")

    return output


if __name__ == "__main__":
    main()
