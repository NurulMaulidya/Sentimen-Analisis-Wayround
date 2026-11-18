"""
Evaluasi Lexicon v2 terhadap Label Pakar
========================================

Tujuan:
- Mengukur akurasi, F1-macro, dan confusion matrix lexicon v2 terhadap
  288 label pakar yang sudah tercatat pada dataset canonical.
- Membandingkannya dengan:
    (a) lexicon lama (v1)
    (b) baseline rating (rating >= 4 Positif, <= 2 Negatif, 3 Netral)
    (c) kandidat label gabungan (lexicon + rating)
- TIDAK pernah memakai holdout untuk pemilihan lexicon.

Semua angka ditulis ke output/lexicon_audit/lexicon_v2_evaluation.json
dan ditampilkan sebagai ringkasan di terminal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from shared.constants import (  # noqa: E402
    CANONICAL_DATASET_PATH,
    LEXICON_AUDIT_DIR,
)

from shared.labeling import (  # noqa: E402
    DEFAULT_NEUTRAL_THRESHOLD,
    label_text,
)
from shared.lexicon import TwoLexicon, get_lexicon  # noqa: E402

LEXICON_V2 = get_lexicon("v2")

CANONICAL = CANONICAL_DATASET_PATH
OUTPUT_DIR = LEXICON_AUDIT_DIR
OUTPUT_JSON = OUTPUT_DIR / "lexicon_v2_evaluation.json"

CLASSES = ["Positif", "Negatif", "Netral"]


def rating_to_label(rating) -> str:
    """Baseline rating: >=4 Positif, <=2 Negatif, 3 Netral."""
    if pd.isna(rating):
        return "Netral"
    value = int(rating)
    if value >= 4:
        return "Positif"
    if value <= 2:
        return "Negatif"
    return "Netral"


def evaluate(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=CLASSES
        ).tolist(),
        "confusion_matrix_labels": CLASSES,
        "per_class": classification_report(
            y_true, y_pred, labels=CLASSES,
            output_dict=True, zero_division=0,
        ),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CANONICAL.exists():
        raise FileNotFoundError(f"Dataset canonical tidak ditemukan: {CANONICAL}")

    df = pd.read_csv(CANONICAL, encoding="utf-8-sig")

    # ---------------------------------------------------------------
    # Subset label pakar: baris yang punya label pakarnya sendiri.
    # ---------------------------------------------------------------
    expert_mask = df["sentiment_label_expert"].notna()
    expert = df[expert_mask].copy()

    print("=" * 78)
    print("EVALUASI LEXICON v2 vs LABEL PAKAR")
    print("=" * 78)
    print(f"Dataset canonical      : {CANONICAL.name} ({len(df)} baris)")
    print(f"Baris berlabelNevalid  : {len(expert)}")
    print(f"Distribusi label pakar : "
          f"{expert['sentiment_label_expert'].value_counts().to_dict()}")
    print()

    y_true = expert["sentiment_label_expert"].astype(str).tolist()
    texts = expert["processed_text"].fillna("").astype(str).tolist()

    # ---------------------------------------------------------------
    # (a) Lexicon v1 - dihitung ulang dengan lexicon v1.
    #
    # CATATAN Tahap 5: sebelumnya nilai ini diambil dari kolom
    # `sentiment_label_inset` yang tersimpan di dataset canonical.
    # kolom itu SEKARANG diisi label lexicon v2 (Stage 3 memakai lexicon
    # v2 sebagai default), sehingga perbandingan "v1 vs v2" menjadi
    # tidak bermakna - keduanya identik. Karena itu v1 dihitung ulang
    # secara eksplisit dari file lexicon v1.
    # ---------------------------------------------------------------
    lexicon_v1 = get_lexicon("v1")
    v1_pred = [
        label_text(t, lexicon=lexicon_v1)["label"] for t in texts
    ]

    # ---------------------------------------------------------------
    # (b) Baseline rating.
    # ---------------------------------------------------------------
    rating_pred = [rating_to_label(r) for r in expert["rating"]]

    # ---------------------------------------------------------------
    # (c) Lexicon v2 - dihitung ulang dengan fungsi pelabelan canonical.
    # ---------------------------------------------------------------
    v2_pred = [label_text(t, lexicon=LEXICON_V2)["label"] for t in texts]

    # ---------------------------------------------------------------
    # (c2) Varian lexicon v2 lain.
    # ---------------------------------------------------------------
    variant_lexicons: dict[str, TwoLexicon] = {}
    variant_preds: dict[str, list[str]] = {}

    for version, folder in (
        ("v2_brief_lengkap", "v2_brief_lengkap"),
        ("v2_tanpa_hapus_topic", "v2_tanpa_hapus_topic"),
    ):
        alt_dir = PROJECT_ROOT / "data" / "lexicon" / folder
        if not (alt_dir / "lexicon_positive.tsv").exists():
            continue
        lexicon = TwoLexicon(
            positive_path=alt_dir / "lexicon_positive.tsv",
            negative_path=alt_dir / "lexicon_negative.tsv",
            version=version,
        )
        variant_lexicons[version] = lexicon
        variant_preds[version] = [
            label_text(t, lexicon=lexicon)["label"] for t in texts
        ]

    # ---------------------------------------------------------------
    # (d) Kandidat gabungan lexicon v2 + rating.
    #     Aturan: rating >= 4 -> Positif; rating <= 2 -> Negatif;
    #              selain itu pakai hasil lexicon v2.
    # ---------------------------------------------------------------
    combined_pred = [
        rating_to_label(r) if r not in (3, 3.0) else lexicon_label
        for r, lexicon_label in zip(expert["rating"], v2_pred)
    ]

    results = {
        "expert_rows": int(len(expert)),
        "neutral_threshold": DEFAULT_NEUTRAL_THRESHOLD,
        "baseline_rating_rule": "rating>=4 Positif, rating<=2 Negatif, rating==3 Netral",
        "methods": {
            "lexicon_v1": evaluate(y_true, v1_pred),
            "lexicon_v2": evaluate(y_true, v2_pred),
            "baseline_rating": evaluate(y_true, rating_pred),
            "kandidat_gabungan_v2_rating": evaluate(y_true, combined_pred),
        },
    }

    for version, preds in variant_preds.items():
        results["methods"][f"lexicon_{version}"] = evaluate(y_true, preds)

    # ---------------------------------------------------------------
    # Distribusi label pada SELURUH baris dataset untuk tiap metode.
    # ---------------------------------------------------------------
    all_texts = df["processed_text"].fillna("").astype(str).tolist()
    all_v1 = [label_text(t, lexicon=lexicon_v1)["label"] for t in all_texts]
    all_v2 = [label_text(t, lexicon=LEXICON_V2)["label"] for t in all_texts]
    all_variant_preds = {
        version: [
            label_text(t, lexicon=lexicon)["label"] for t in all_texts
        ]
        for version, lexicon in variant_lexicons.items()
    }
    all_rating = [rating_to_label(r) for r in df["rating"]]

    results["distribusi_seluruh_dataset"] = {
        "lexicon_v1": pd.Series(all_v1).value_counts().to_dict(),
        "lexicon_v2": pd.Series(all_v2).value_counts().to_dict(),
        "baseline_rating": pd.Series(all_rating).value_counts().to_dict(),
    }
    for version, preds in all_variant_preds.items():
        results["distribusi_seluruh_dataset"][f"lexicon_{version}"] = (
            pd.Series(preds).value_counts().to_dict()
        )

    # ---------------------------------------------------------------
    # Cetak ringkasan.
    # ---------------------------------------------------------------
    print("RINGKASAN PADA 288 LABEL PAKAR")
    print("-" * 78)
    header = f"{'Metode':<32}{'Akurasi':>10}{'F1-macro':>12}{'F1-weighted':>14}"
    print(header)
    print("-" * 78)

    labels = {
        "lexicon_v1": "Lexicon lama (v1)",
        "lexicon_v2": "Lexicon baru (v2)",
        "baseline_rating": "Baseline rating",
        "kandidat_gabungan_v2_rating": "Kandidat gabungan v2+rating",
    }
    labels["lexicon_v2_brief_lengkap"] = "Lexicon v2 (brief lengkap: R1+R3+R4)"
    labels["lexicon_v2_tanpa_hapus_topic"] = "Lexicon v2 (R3+R4, tanpa hapus topic)"

    for key, label in labels.items():
        item = results["methods"][key]
        print(
            f"{label:<32}"
            f"{item['accuracy']:>10.4f}"
            f"{item['f1_macro']:>12.4f}"
            f"{item['f1_weighted']:>14.4f}"
        )

    print()
    for key, label in labels.items():
        print(f"Confusion matrix (baris=pakar, kolom=prediksi) - {label}")
        matrix = results["methods"][key]["confusion_matrix"]
        print("              " + "".join(f"{c:>10}" for c in CLASSES))
        for name, row in zip(CLASSES, matrix):
            print(f"  {name:<10}" + "".join(f"{v:>10}" for v in row))
        print()

    print("Distribusi label pada seluruh 3.060 baris")
    print("-" * 78)
    for key, dist in results["distribusi_seluruh_dataset"].items():
        print(f"  {key:<28} {dist}")

    OUTPUT_JSON.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print(f"Hasil disimpan: {OUTPUT_JSON}")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
