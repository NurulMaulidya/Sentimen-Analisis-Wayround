"""
Analisis aspek pada DATASET CANONICAL dan MODEL FINAL TERKUNCI.

TUJUAN
    Menghasilkan satu-satunya artefak analisis aspek yang sah untuk laporan:
    - dataset canonical yang sama dengan training/evaluasi (Stage 3 v4)
    - konfigurasi model yang sama dengan model final (shared.final_config)
    - tanpa memakai holdout untuk training/seleksi apa pun

ALUR
    data/pipeline_output_v4/stage3_final_labeled.csv
        -> deteksi aspek per kalimat (pipeline/aspect_detection.py)
        -> data/pipeline_output_v4/stage4_aspects.csv
        -> ringkasan per aspek:
             * jumlah ulasan per aspek
             * distribusi label final (Positif/Netral/Negatif)
             * sentimen aspek hasil INFERENCE model final terpasang
             * diagnostik riset per aspek (TF-IDF + SVM, grouped CV,
               HANYA development, TIDAK dipakai untuk deployed model)

CATATAN METODOLOGI
    1. Label aspek pada bagian ringkasan mewarisi label final tingkat
       ulasan (sentiment_label_final). Analisis aspek TIDAK mengubah
       label dan TIDAK menggantikan SVM.
    2. Bagian "model_sentiment" memakai artefak final yang terpasang
       (models/final_v4/final_pipeline.joblib) dalam mode inferensi saja:
       tidak ada fit(), tidak ada TF-IDF baru, tidak ada model kedua.
    3. Bagian "research_diagnostic" melatih model per aspek KHUSUS untuk
       diagnostik penelitian ( grouped CV pada development ). Model ini
       TIDAK dideploy, TIDAK dipakai backend/frontend, dan TIDAK memakai
       holdout.

CLI
    python scripts/run_aspect_analysis.py --output-dir runs/analisis-aspek

OUTPUT
    data/pipeline_output_v4/stage4_aspects.csv
    output_v4/experiments/aspect_analysis.json
    output_v4/experiments/aspect_analysis_summary.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.constants import ASPECTS, PIPELINE_OUTPUT_DIR  # noqa: E402
from shared.deployment_integrity import load_verified_pipeline  # noqa: E402
from shared.final_config import (  # noqa: E402
    CV_FOLDS,
    EXPERIMENTS_DIR,
    FINAL_MODEL_CONFIG,
    LABEL_COLUMN,
    RANDOM_STATE,
    TEXT_COLUMN,
)
from shared.preprocessing import preprocess_text  # noqa: E402
from shared.splitting import build_groups, load_canonical_split  # noqa: E402
from pipeline.aspect_detection import detect_aspect_batch  # noqa: E402

EXPECTED_CLASSES = ["negatif", "netral", "positif"]
LABEL_DISPLAY = {
    "positif": "Positif",
    "netral": "Netral",
    "negatif": "Negatif",
}

ASPECT_OUTPUT_PATH = PIPELINE_OUTPUT_DIR / "stage4_aspects.csv"
JSON_OUTPUT_PATH = EXPERIMENTS_DIR / "aspect_analysis.json"
CSV_OUTPUT_PATH = EXPERIMENTS_DIR / "aspect_analysis_summary.csv"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _distribution(labels) -> dict:
    """Distribusi label kanonik (Positif/Netral/Negatif) dengan persentase."""
    series = pd.Series(list(labels), dtype="object")
    total = int(series.shape[0])
    result = {}
    for key in EXPECTED_CLASSES:
        count = int((series == key).sum())
        result[LABEL_DISPLAY[key]] = {
            "count": count,
            "percentage": round(count / total * 100, 4) if total else 0.0,
        }
    result["total"] = total
    return result


def _build_research_pipeline() -> Pipeline:
    """Pipeline diagnostik per aspek: TF-IDF + SVM OVO dengan konfigurasi final."""
    tfidf = FINAL_MODEL_CONFIG["tfidf"]
    svm = FINAL_MODEL_CONFIG["svm"]
    strategy = FINAL_MODEL_CONFIG["multiclass"]["strategy"]
    base = SVC(
        kernel=svm["kernel"],
        C=svm["C"],
        gamma=svm["gamma"],
        class_weight=svm["class_weight"],
        random_state=RANDOM_STATE,
    )
    classifier = OneVsOneClassifier(base) if strategy == "ovo" else base
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=int(tfidf["max_features"]),
            ngram_range=tuple(int(x) for x in tfidf["ngram_range"]),
            sublinear_tf=bool(tfidf["sublinear_tf"]),
            min_df=int(tfidf["min_df"]),
            max_df=float(tfidf["max_df"]),
        )),
        ("classifier", classifier),
    ])


def _grouped_cv_metrics(frame: pd.DataFrame, min_class_support: int = 5) -> dict:
    """
    Grouped CV pada DEVELOPMENT untuk diagnostik per aspek.

    Grup = processed_text (baris dengan teks identik tidak boleh terpisah
    antar fold). Model per aspek ini hanya diagnostik penelitian.
    """
    labels = frame["label"].astype(str)
    if frame["label"].nunique() < 2:
        return {"status": "skip", "reason": "kurang dari dua kelas"}

    groups = build_groups(frame)
    usable = labels.value_counts()
    keep = usable[usable >= min_class_support].index.tolist()
    frame = frame[labels.isin(keep)]
    labels = frame["label"].astype(str)
    groups = build_groups(frame)
    if len(frame) < CV_FOLDS * 2 or labels.nunique() < 2:
        return {"status": "skip", "reason": "data tidak cukup untuk grouped CV"}

    splitter = StratifiedGroupKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    texts = frame["processed_text"].astype(str).values
    targets = labels.values

    fold_scores = []
    for train_idx, valid_idx in splitter.split(texts, targets, groups=groups):
        pipeline = _build_research_pipeline()
        pipeline.fit(texts[train_idx], targets[train_idx])
        predicted = pipeline.predict(texts[valid_idx])
        fold_scores.append(float(
            f1_score(targets[valid_idx], predicted, labels=EXPECTED_CLASSES,
                     average="macro", zero_division=0)
        ))

    return {
        "status": "complete",
        "method": f"StratifiedGroupKFold (n_splits={CV_FOLDS}, group=processed_text)",
        "scope": "development_only",
        "folds": len(fold_scores),
        "cv_f1_macro_mean": round(sum(fold_scores) / len(fold_scores), 6),
        "cv_f1_macro_std": round(
            (sum((s - sum(fold_scores) / len(fold_scores)) ** 2 for s in fold_scores)
             / max(1, len(fold_scores) - 1)) ** 0.5, 6),
        "note": "Diagnostik per aspek pada development. Bukan model yang dideploy.",
    }


# ---------------------------------------------------------------------------
# Pipeline utama
# ---------------------------------------------------------------------------

def main(output_dir: Path | None = None) -> int:
    aspect_output = output_dir / ASPECT_OUTPUT_PATH.name if output_dir else ASPECT_OUTPUT_PATH
    json_output = output_dir / JSON_OUTPUT_PATH.name if output_dir else JSON_OUTPUT_PATH
    csv_output = output_dir / CSV_OUTPUT_PATH.name if output_dir else CSV_OUTPUT_PATH
    print("=" * 78)
    print("ANALISIS ASPEK - DATASET CANONICAL & MODEL FINAL TERKUNCI")
    print("=" * 78)

    from shared.constants import CANONICAL_DATASET_PATH

    dataset_path = CANONICAL_DATASET_PATH
    if not dataset_path.exists():
        print(f"[ERROR] Dataset canonical tidak ditemukan: {dataset_path}")
        return 1
    print(f"[OK] Dataset canonical : {dataset_path}")

    df = pd.read_csv(dataset_path, encoding="utf-8-sig", low_memory=False)
    if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
        print(f"[ERROR] Kolom wajib tidak ditemukan: {TEXT_COLUMN}, {LABEL_COLUMN}")
        return 1
    df["label"] = df[LABEL_COLUMN].astype(str).str.strip().str.lower()
    df["label_display"] = df["label"].map(LABEL_DISPLAY)
    print(f"[OK] Baris dataset     : {len(df)}")
    print(f"[OK] Distribusi label  : {df['label_display'].value_counts().to_dict()}")

    development, holdout, split_info = load_canonical_split()
    dev_ids = set(development["reviewId"].astype(str)) if "reviewId" in development.columns else set()
    print(f"[OK] Split canonical   : {split_info.get('source')} "
          f"(dev {len(development)}, holdout {len(holdout)})")

    # ------------------------------------------------------------------
    # 1. Deteksi aspek (pipeline/aspect_detection.py)
    # ------------------------------------------------------------------
    print("\n[1] DETEKSI ASPEK PER KALIMAT")
    aspects_df = detect_aspect_batch(df, text_column="content", review_id_column="reviewId")
    if aspects_df.empty:
        print("[ERROR] Tidak ada aspek yang terdeteksi.")
        return 1
    aspects_df["sentiment_label_final"] = aspects_df["reviewId"].map(
        df.set_index("reviewId")["label"]
    )
    aspect_output.parent.mkdir(parents=True, exist_ok=True)
    aspects_df.to_csv(aspect_output, index=False, encoding="utf-8-sig")
    print(f"[OK] Baris aspek       : {len(aspects_df)}")
    print(f"[OK] Tersimpan         : {aspect_output}")

    # Kalimat aspek diproses ulang dengan preprocessing canonical.
    aspects_df["processed_text"] = [
        preprocess_text(sentence) for sentence in aspects_df["sentence"].astype(str)
    ]

    # ------------------------------------------------------------------
    # 2. Inferensi model final pada kalimat aspek (TANPA training)
    # ------------------------------------------------------------------
    print("\n[2] INFERENSI MODEL FINAL PADA KALIMAT ASPEK")
    pipeline, metadata = load_verified_pipeline()
    model_labels = pipeline.predict(aspects_df["processed_text"].astype(str).values)
    aspects_df["model_label"] = [str(label).lower() for label in model_labels]
    aspects_df["model_label_display"] = aspects_df["model_label"].map(LABEL_DISPLAY)
    print(f"[OK] Model version     : {metadata['model_version']}")

    # ------------------------------------------------------------------
    # 3. Ringkasan per aspek
    # ------------------------------------------------------------------
    print("\n[3] RINGKASAN PER ASPEK")
    review_ids = set(aspects_df["reviewId"].astype(str))
    results = {}
    summary_rows = []

    for aspect in ASPECTS:
        subset = aspects_df[aspects_df["aspect"] == aspect]
        if subset.empty:
            continue

        aspect_review_ids = set(subset["reviewId"].astype(str))
        review_rows = df[df["reviewId"].astype(str).isin(aspect_review_ids)]
        development_rows = (
            review_rows[review_rows["reviewId"].astype(str).isin(dev_ids)]
            if dev_ids else review_rows
        )

        # Diagnostik riset hanya memakai kalimat aspek dari baris development.
        development_sentence_mask = (
            subset["reviewId"].astype(str).isin(dev_ids) if dev_ids else True
        )
        research_frame = pd.DataFrame({
            "processed_text": subset.loc[development_sentence_mask, "processed_text"].astype(str),
            "label": subset.loc[development_sentence_mask, "sentiment_label_final"],
        })
        research_frame = research_frame[research_frame["label"].isin(EXPECTED_CLASSES)]
        research_frame = research_frame[research_frame["processed_text"].str.strip() != ""]

        cv = _grouped_cv_metrics(research_frame)

        entry = {
            "aspect": aspect,
            "reviews_with_aspect": int(len(aspect_review_ids)),
            "sentences_with_aspect": int(len(subset)),
            "review_label_distribution": _distribution(review_rows["label"]),
            "sentence_label_distribution": _distribution(subset["sentiment_label_final"]),
            "model_sentiment_distribution": _distribution(subset["model_label"]),
            "development_reviews_with_aspect": int(len(development_rows)),
            "research_diagnostic": cv,
            "definitions": {
                "review_label_distribution":
                    "Label final tingkat ulasan (mewarisi label review yang memuat aspek).",
                "sentence_label_distribution":
                    "Label final yang diwarisi kalimat aspek (dari ulasan induknya).",
                "model_sentiment_distribution":
                    "Hasil inferensi artefak final terpasang pada kalimat aspek.",
                "research_diagnostic":
                    "TF-IDF + SVM per aspek, grouped CV pada development saja.",
            },
        }
        results[aspect] = entry

        summary_rows.append({
            "aspect": aspect,
            "reviews_with_aspect": entry["reviews_with_aspect"],
            "sentences_with_aspect": entry["sentences_with_aspect"],
            "Positif": entry["model_sentiment_distribution"]["Positif"]["count"],
            "Netral": entry["model_sentiment_distribution"]["Netral"]["count"],
            "Negatif": entry["model_sentiment_distribution"]["Negatif"]["count"],
            "review_Positif": entry["review_label_distribution"]["Positif"]["count"],
            "review_Netral": entry["review_label_distribution"]["Netral"]["count"],
            "review_Negatif": entry["review_label_distribution"]["Negatif"]["count"],
            "cv_f1_macro_mean": cv.get("cv_f1_macro_mean"),
            "cv_f1_macro_std": cv.get("cv_f1_macro_std"),
        })
        print(f"     - {aspect}: {entry['reviews_with_aspect']} ulasan, "
              f"{entry['sentences_with_aspect']} kalimat")

    payload = {
        "status": "final",
        "scope": "research_aspect_analysis",
        "dataset": {
            "path": str(dataset_path.relative_to(PROJECT_ROOT)),
            "rows": int(len(df)),
            "text_column": TEXT_COLUMN,
            "label_column": LABEL_COLUMN,
            "class_distribution": df["label_display"].value_counts().to_dict(),
        },
        "split": {
            "source": split_info.get("source"),
            "development_rows": int(len(development)),
            "holdout_rows": int(len(holdout)),
            "holdout_used_for_selection": False,
            "holdout_included_in_descriptive_distribution": True,
            "holdout_used_in_aspect_cv": False,
        },
        "model": {
            "artifact": metadata["artifacts"]["pipeline"],
            "model_version": metadata["model_version"],
            "configuration": metadata["model"],
            "usage": "inference_only_no_refit",
        },
        "preprocessing": {
            "module": "shared/preprocessing.py",
            "stages": ["Cleaning", "Case Folding", "Tokenizing", "Normalization",
                       "Negation Handling", "Stopword Removal", "Stemming"],
        },
        "aspects": results,
        "notes": [
            "Analisis aspek adalah modul tambahan; tidak mengubah label SVM.",
            "Distribusi label aspek mewarisi label final tingkat ulasan.",
            "Sentimen model dihitung dengan artefak final terpasang (tanpa fit ulang).",
            "Distribusi deskriptif mencakup seluruh dataset. CV per aspek hanya memakai development.",
        ],
    }

    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(summary_rows).to_csv(csv_output, index=False, encoding="utf-8-sig")
    print(f"\n[OK] JSON  : {json_output}")
    print(f"[OK] CSV   : {csv_output}")
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder untuk ketiga laporan aspek.")
    raise SystemExit(main(parser.parse_args().output_dir))
