"""
Canonical pipeline/labeling.py
==============================
Pelabelan sentimen menggunakan InSet Lexicon.
Tanggung jawab:
- label satu review
- label DataFrame
- label sentimen per aspek
- distribusi sentimen
- analisis kualitas labeling
- export data dan laporan
Sumber logika:
- shared.preprocessing
- shared.lexicon
- shared.constants
File ini TIDAK melakukan training SVM.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys
import pandas as pd
# ============================================================================
# PROJECT ROOT
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ============================================================================
# CANONICAL IMPORTS
from shared.constants import PIPELINE_OUTPUT_DIR  # noqa: E402
# ============================================================================
from shared.final_config import (
    NEUTRAL_THRESHOLD,
    SENTIMENT_CLASSES,
    ASPECTS,
)
from shared.preprocessing import preprocess_text
from shared.labeling import label_text as active_label_text, get_active_lexicon
from shared.lexicon import (
    classify_sentiment,
    get_label_distribution,
)
# ============================================================================
# INTERNAL CLASSIFIER ADAPTER
# ============================================================================
def _classify(
    processed_text: str,
) -> Tuple[str, float, float]:
    """
    Adapter untuk classify_sentiment().
    """
    result = active_label_text(processed_text)
    if isinstance(result, dict):
        return (
            result["label"],
            result["raw_score"],
            result["normalized_score"],
        )
    if isinstance(result, (tuple, list)) and len(result) >= 3:
        return (
            result[0],
            result[1],
            result[2],
        )
    raise TypeError(
        "classify_sentiment() harus mengembalikan "
        "dict atau tuple/list minimal 3 elemen."
    )
# ============================================================================
# LABEL REVIEW
# ============================================================================
def label_review(
    text: str,
    processed_text: Optional[str] = None,
) -> Tuple[str, float, float]:
    """
    Label satu review menggunakan InSet Lexicon.
    Parameters
    ----------
    text:
        Teks asli review.
    processed_text:
        Teks yang sudah dipreprocessing.
        Jika None, preprocessing akan dilakukan.
    Returns
    -------
    tuple
        (label, raw_score, normalized_score)
    """
    if processed_text is None:
        processed_text = preprocess_text(text)
    return _classify(processed_text)
# ============================================================================
# LABEL DATAFRAME
# ============================================================================
def label_dataframe(
    df: pd.DataFrame,
    text_column: str = "content",
    original_column: str = "content",
    preprocess_if_needed: bool = True,
) -> pd.DataFrame:
    """
    Label seluruh DataFrame menggunakan InSet Lexicon.
    Parameters
    ----------
    df:
        DataFrame review.
    text_column:
        Kolom teks jika preprocessing tidak diperlukan.
    original_column:
        Kolom teks asli untuk preprocessing.
    preprocess_if_needed:
        Jika True, gunakan processed_text jika tersedia.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df harus berupa pandas.DataFrame."
        )
    result = df.copy()
    # --------------------------------------------------------
    # Tentukan kolom teks
    # --------------------------------------------------------
    if preprocess_if_needed:
        if "processed_text" not in result.columns:
            if original_column not in result.columns:
                raise KeyError(
                    f"Kolom sumber '{original_column}' "
                    "tidak ditemukan."
                )
            result["processed_text"] = (
                result[original_column]
                .fillna("")
                .astype(str)
                .apply(preprocess_text)
            )
        working_column = "processed_text"
    else:
        if text_column not in result.columns:
            raise KeyError(
                f"Kolom teks '{text_column}' "
                "tidak ditemukan."
            )
        working_column = text_column
    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------
    labels = []
    scores = []
    normalized_scores = []
    for text in (
        result[working_column]
        .fillna("")
        .astype(str)
    ):
        label, score, normalized = _classify(text)
        labels.append(label)
        scores.append(score)
        normalized_scores.append(normalized)
    # --------------------------------------------------------
    # Tambahkan hasil
    # --------------------------------------------------------
    result["sentiment_label"] = labels
    result["sentiment_score"] = scores
    result["sentiment_normalized"] = normalized_scores
    return result
# ============================================================================
# LABEL SENTIMENT PER ASPEK
# ============================================================================
def label_with_aspects(
    df: pd.DataFrame,
    aspects_df: pd.DataFrame,
    text_column: str = "processed_text",
    review_id_column: str = "reviewId",
) -> pd.DataFrame:
    """
    Label sentimen pada setiap kalimat aspek.
    Parameters
    ----------
    df:
        DataFrame review.
    aspects_df:
        DataFrame hasil deteksi aspek.
        Minimal memiliki:
        - reviewId
        - sentence
        - aspect
    text_column:
        Kolom teks pada df.
    review_id_column:
        ID review untuk proses merge.
    Returns
    -------
    pandas.DataFrame
        DataFrame aspek dengan:
        - sentence
        - aspect
        - sentiment_label
        - sentiment_score
        - sentiment_normalized
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df harus berupa pandas.DataFrame."
        )
    if not isinstance(aspects_df, pd.DataFrame):
        raise TypeError(
            "aspects_df harus berupa pandas.DataFrame."
        )
    # --------------------------------------------------------
    # Validasi df
    # --------------------------------------------------------
    required_review_columns = {
        review_id_column,
        text_column,
    }
    missing_review = (
        required_review_columns
        - set(df.columns)
    )
    if missing_review:
        raise KeyError(
            "Kolom review tidak ditemukan: "
            f"{sorted(missing_review)}"
        )
    # --------------------------------------------------------
    # Validasi aspects_df
    # --------------------------------------------------------
    required_aspect_columns = {
        review_id_column,
        "sentence",
        "aspect",
    }
    missing_aspect = (
        required_aspect_columns
        - set(aspects_df.columns)
    )
    if missing_aspect:
        raise KeyError(
            "Kolom aspek tidak ditemukan: "
            f"{sorted(missing_aspect)}"
        )
    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------
    merged = aspects_df.merge(
        df[
            [
                review_id_column,
                text_column,
            ]
        ],
        on=review_id_column,
        how="left",
    )
    # --------------------------------------------------------
    # Label setiap kalimat aspek
    # --------------------------------------------------------
    rows = []
    for _, row in merged.iterrows():
        if pd.isna(row["sentence"]):
            sentence = ""
        else:
            sentence = str(row["sentence"])
        processed = preprocess_text(sentence)
        label, score, normalized = _classify(
            processed
        )
        rows.append(
            {
                review_id_column: row[
                    review_id_column
                ],
                "sentence": sentence,
                "aspect": row["aspect"],
                "sentiment_label": label,
                "sentiment_score": score,
                "sentiment_normalized": normalized,
            }
        )
    return pd.DataFrame(rows)
# ============================================================================
# ASPECT SENTIMENT DISTRIBUTION
# ============================================================================
def get_aspect_sentiment_distribution(
    labeled_aspects_df: pd.DataFrame,
) -> Dict:
    """
    Hitung distribusi sentimen per aspek.
    """
    if (
        labeled_aspects_df is None
        or labeled_aspects_df.empty
    ):
        return {}
    required_columns = {
        "aspect",
        "sentiment_label",
    }
    missing = (
        required_columns
        - set(labeled_aspects_df.columns)
    )
    if missing:
        raise KeyError(
            "Kolom distribusi aspek tidak ditemukan: "
            f"{sorted(missing)}"
        )
    distribution = {}
    # --------------------------------------------------------
    # Per aspek
    # --------------------------------------------------------
    for aspect in ASPECTS:
        aspect_data = labeled_aspects_df[
            labeled_aspects_df["aspect"] == aspect
        ]
        if aspect_data.empty:
            continue
        counts = (
            aspect_data["sentiment_label"]
            .value_counts()
            .to_dict()
        )
        distribution[aspect] = {
            "Positif": int(
                counts.get("Positif", 0)
            ),
            "Netral": int(
                counts.get("Netral", 0)
            ),
            "Negatif": int(
                counts.get("Negatif", 0)
            ),
            "Total": int(
                len(aspect_data)
            ),
        }
    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------
    total_counts = (
        labeled_aspects_df[
            "sentiment_label"
        ]
        .value_counts()
        .to_dict()
    )
    distribution["Overall"] = {
        "Positif": int(
            total_counts.get("Positif", 0)
        ),
        "Netral": int(
            total_counts.get("Netral", 0)
        ),
        "Negatif": int(
            total_counts.get("Negatif", 0)
        ),
        "Total": int(
            len(labeled_aspects_df)
        ),
    }
    return distribution
# ============================================================================
# ASPECT SENTIMENT PERCENTAGE
# ============================================================================
def get_aspect_sentiment_percentage(
    distribution: Dict,
) -> Dict:
    """
    Konversi distribusi sentimen menjadi persentase.
    """
    result = {}
    for aspect, counts in distribution.items():
        total = int(
            counts.get("Total", 0)
        )
        if total <= 0:
            result[aspect] = {
                "Positif": 0.0,
                "Netral": 0.0,
                "Negatif": 0.0,
                "Total": 0,
            }
            continue
        result[aspect] = {
            "Positif": round(
                100
                * counts.get("Positif", 0)
                / total,
                2,
            ),
            "Netral": round(
                100
                * counts.get("Netral", 0)
                / total,
                2,
            ),
            "Negatif": round(
                100
                * counts.get("Negatif", 0)
                / total,
                2,
            ),
            "Total": total,
        }
    return result
# ============================================================================
# LABELING QUALITY
# ============================================================================
def analyze_labeling_quality(
    df: pd.DataFrame,
    label_column: str = "sentiment_label",
) -> Dict:
    """
    Analisis statistik kualitas pelabelan.
    """
    distribution = get_label_distribution(
        df,
        label_column,
    )
    # --------------------------------------------------------
    # Raw score
    # --------------------------------------------------------
    if "sentiment_score" in df.columns:
        score_stats = {
            "mean": float(
                df["sentiment_score"].mean()
            ),
            "std": float(
                df["sentiment_score"].std()
            ),
            "min": float(
                df["sentiment_score"].min()
            ),
            "max": float(
                df["sentiment_score"].max()
            ),
            "median": float(
                df["sentiment_score"].median()
            ),
        }
    else:
        score_stats = {
            "error": "Column not found"
        }
    # --------------------------------------------------------
    # Normalized score
    # --------------------------------------------------------
    if "sentiment_normalized" in df.columns:
        normalized_stats = {
            "mean": float(
                df[
                    "sentiment_normalized"
                ].mean()
            ),
            "std": float(
                df[
                    "sentiment_normalized"
                ].std()
            ),
            "min": float(
                df[
                    "sentiment_normalized"
                ].min()
            ),
            "max": float(
                df[
                    "sentiment_normalized"
                ].max()
            ),
            "median": float(
                df[
                    "sentiment_normalized"
                ].median()
            ),
        }
    else:
        normalized_stats = {
            "error": "Column not found"
        }
    return {
        "distribution": distribution,
        "score_stats": score_stats,
        "normalized_stats": normalized_stats,
        "threshold_used": NEUTRAL_THRESHOLD,
    }
# ============================================================================
# EDGE CASES
# ============================================================================
def find_edge_cases(
    df: pd.DataFrame,
    score_column: str = "sentiment_normalized",
    threshold: float = NEUTRAL_THRESHOLD,
) -> Dict:
    """
    Temukan kasus yang dekat dengan threshold.
    Catatan:
    threshold final project = 0.0.
    Karena threshold 0.0 menghasilkan margin 0,
    fungsi mengembalikan daftar kosong untuk konfigurasi final.
    """
    if score_column not in df.columns:
        raise KeyError(
            f"Kolom skor '{score_column}' "
            "tidak ditemukan."
        )
    margin = abs(
        float(threshold)
    ) * 0.2
    if margin == 0:
        return {
            "near_positive": 0,
            "near_negative": 0,
            "examples_near_positive": [],
            "examples_near_negative": [],
        }
    near_positive = df[
        (
            df[score_column]
            > threshold - margin
        )
        & (
            df[score_column]
            <= threshold
        )
    ]
    near_negative = df[
        (
            df[score_column]
            >= -threshold
        )
        & (
            df[score_column]
            < -threshold + margin
        )
    ]
    return {
        "near_positive": int(
            len(near_positive)
        ),
        "near_negative": int(
            len(near_negative)
        ),
        "examples_near_positive":
            near_positive
            .head(5)
            .to_dict("records"),
        "examples_near_negative":
            near_negative
            .head(5)
            .to_dict("records"),
    }
# ============================================================================
# SAVE LABELED DATA
# ============================================================================
def save_labeled_data(
    df: pd.DataFrame,
    output_path: Path,
    aspects_df: Optional[
        pd.DataFrame
    ] = None,
) -> None:
    """
    Simpan data berlabel ke CSV.
    """
    output_path = Path(
        output_path
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )
    if aspects_df is not None:
        aspects_path = (
            output_path.parent
            / f"{output_path.stem}_aspects.csv"
        )
        aspects_df.to_csv(
            aspects_path,
            index=False,
            encoding="utf-8-sig",
        )
# ============================================================================
# REPORT
# ============================================================================
def generate_labeling_report(
    df: pd.DataFrame,
    labeled_aspects_df: Optional[
        pd.DataFrame
    ] = None,
) -> str:
    """
    Generate laporan pelabelan dalam format teks.
    """
    report = [
        "=" * 80,
        "LAPORAN PELABELAN SENTIMEN",
        "=" * 80,
        "",
        "[DISTRIBUSI SENTIMEN KESELURUHAN]",
    ]
    distribution = get_label_distribution(
        df,
        "sentiment_label",
    )
    for label in SENTIMENT_CLASSES:
        if label not in distribution:
            continue
        value = distribution[label]
        if isinstance(value, dict):
            count = value.get(
                "count",
                0,
            )
            percentage = value.get(
                "percentage",
                0,
            )
        else:
            count = 0
            percentage = 0
        report.append(
            f"  {label:10s}: "
            f"{count:5d} "
            f"({percentage:5.2f}%)"
        )
    report.extend(
        [
            "",
            "[STATISTIK SKOR]",
        ]
    )
    if (
        "sentiment_score" in df.columns
        and
        "sentiment_normalized"
        in df.columns
    ):
        report.append(
            "  Skor Total - "
            f"Mean: "
            f"{df['sentiment_score'].mean():.4f}, "
            f"Std: "
            f"{df['sentiment_score'].std():.4f}"
        )
        report.append(
            "  Skor Normalisasi - "
            f"Mean: "
            f"{df['sentiment_normalized'].mean():.4f}, "
            f"Std: "
            f"{df['sentiment_normalized'].std():.4f}"
        )
    else:
        report.append(
            "  [INFO] Kolom skor tidak tersedia"
        )
    report.append(
        f"  Threshold Netral: "
        f"{NEUTRAL_THRESHOLD}"
    )
    # --------------------------------------------------------
    # Aspect distribution
    # --------------------------------------------------------
    if (
        labeled_aspects_df is not None
        and not labeled_aspects_df.empty
    ):
        report.extend(
            [
                "",
                "[DISTRIBUSI SENTIMEN PER ASPEK]",
            ]
        )
        aspect_distribution = (
            get_aspect_sentiment_distribution(
                labeled_aspects_df
            )
        )
        aspect_percentage = (
            get_aspect_sentiment_percentage(
                aspect_distribution
            )
        )
        for aspect in ASPECTS:
            if aspect not in aspect_percentage:
                continue
            pct = aspect_percentage[
                aspect
            ]
            report.extend(
                [
                    "",
                    f"  [{aspect}]",
                    (
                        "    Positif: "
                        f"{pct['Positif']:.1f}%"
                    ),
                    (
                        "    Netral:  "
                        f"{pct['Netral']:.1f}%"
                    ),
                    (
                        "    Negatif: "
                        f"{pct['Negatif']:.1f}%"
                    ),
                    (
                        "    Total:   "
                        f"{pct['Total']} kalimat"
                    ),
                ]
            )
    report.append(
        "=" * 80
    )
    return "\n".join(report)
# ============================================================================
# PUBLIC API
# ============================================================================
__all__ = [
    "label_review",
    "label_dataframe",
    "label_with_aspects",
    "get_aspect_sentiment_distribution",
    "get_aspect_sentiment_percentage",
    "analyze_labeling_quality",
    "find_edge_cases",
    "save_labeled_data",
    "generate_labeling_report",
]
# =============================================================================
# CLI / ENTRY POINT
# =============================================================================
def main():
    """Menjalankan Stage 3 labeling secara mandiri.
    Input ditetapkan ke:
        <WAYGROUND_PIPELINE_OUTPUT_DIR>/stage2_preprocessed_new.csv
    Labeling selalu menggunakan kolom:
        processed_text
    Tidak ada preprocessing ulang pada tahap labeling.
    """
    import argparse
    import traceback
    parser = argparse.ArgumentParser(
        description=(
            "Stage 3 - Labeling sentimen menggunakan lexicon aktif "
            "dengan input tetap stage2_preprocessed_new.csv."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default=str(PIPELINE_OUTPUT_DIR / "stage2_preprocessed_new.csv"),
        help="File hasil preprocessing Stage 2.",
    )
    parser.add_argument(
        "--output",
        default=str(PIPELINE_OUTPUT_DIR / "stage3_labeled.csv"),
        help="File hasil labeling Stage 3.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Path laporan TXT. Jika kosong, mengikuti nama file output.",
    )
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    report_path = (
        Path(args.report)
        if args.report
        else output_path.with_name(f"{output_path.stem}_report.txt")
    )
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path
    print("=" * 78)
    print("STAGE 3 - LABELING SENTIMEN")
    print("=" * 78)
    print(f"[INFO] Project root : {PROJECT_ROOT}")
    print(f"[INFO] Input        : {input_path}")
    print(f"[INFO] Output       : {output_path}")
    print(f"[INFO] Report       : {report_path}")
    print("[INFO] Kolom label  : processed_text")
    print("[INFO] Metode       : InSet Lexicon")
    print()
    if not input_path.exists():
        print(f"[ERROR] File input tidak ditemukan: {input_path}")
        return 1
    try:
        # ------------------------------------------------------------------
        # 1. Baca hasil preprocessing
        # ------------------------------------------------------------------
        print("[1/5] Membaca data hasil preprocessing...")
        try:
            df = pd.read_csv(input_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(input_path, encoding="utf-8")
        if df.empty:
            print("[ERROR] File input kosong.")
            return 1
        print(f"[INFO] Jumlah data : {len(df):,}")
        print(f"[INFO] Kolom       : {list(df.columns)}")
        # ------------------------------------------------------------------
        # 2. Validasi kolom canonical
        # ------------------------------------------------------------------
        print()
        print("[2/5] Memvalidasi hasil preprocessing...")
        if "processed_text" not in df.columns:
            print(
                "[ERROR] Kolom 'processed_text' tidak ditemukan. "
                "Labeling wajib menggunakan hasil preprocessing Stage 2."
            )
            return 1
        # Pastikan tidak ada proses preprocessing ulang.
        processed = df["processed_text"].fillna("").astype(str).str.strip()
        empty_count = int((processed == "").sum())
        print("[OK] Kolom canonical ditemukan: processed_text")
        print(f"[INFO] Teks kosong          : {empty_count:,}")
        print("[OK] Preprocessing ulang   : TIDAK dilakukan")
        # ------------------------------------------------------------------
        # 3. Labeling menggunakan processed_text
        # ------------------------------------------------------------------
        print()
        print("[3/5] Melakukan labeling sentimen...")
        labeled_df = label_dataframe(
            df.copy(),
            text_column="processed_text",
            original_column="content",
            preprocess_if_needed=False,
        )
        required_columns = {
            "sentiment_label",
            "sentiment_score",
            "sentiment_normalized",
        }
        missing = required_columns - set(labeled_df.columns)
        if missing:
            print(
                "[ERROR] Kolom hasil labeling tidak lengkap:",
                sorted(missing),
            )
            return 1
        print("[OK] Labeling selesai.")
        # ------------------------------------------------------------------
        # 4. Analisis hasil labeling
        # ------------------------------------------------------------------
        print()
        print("[4/5] Menganalisis hasil labeling...")
        distribution = get_label_distribution(
            labeled_df,
            "sentiment_label",
        )
        print("-" * 78)
        print("DISTRIBUSI LABEL SENTIMEN")
        print("-" * 78)
        for label in SENTIMENT_CLASSES:
            value = distribution.get(label, 0)
            if isinstance(value, dict):
                count = int(value.get("count", 0))
                pct = float(value.get("percentage", 0.0))
            else:
                count = int(value)
                pct = count / len(labeled_df) * 100
            print(f"{str(label):<12}: {count:>6,} ({pct:>6.2f}%)")
        known = set(SENTIMENT_CLASSES)
        for label, value in distribution.items():
            if label not in known:
                if isinstance(value, dict):
                    count = int(value.get("count", 0))
                    pct = float(value.get("percentage", 0.0))
                else:
                    count = int(value)
                    pct = count / len(labeled_df) * 100
                print(f"{str(label):<12}: {count:>6,} ({pct:>6.2f}%)")
        quality = analyze_labeling_quality(labeled_df)
        # ------------------------------------------------------------------
        # 5. Simpan hasil dan laporan
        # ------------------------------------------------------------------
        print()
        print("[5/5] Menyimpan hasil labeling...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        save_labeled_data(
            labeled_df,
            output_path,
        )
        # generate_labeling_report() menerima DataFrame dan optional
        # labeled_aspects_df, bukan parameter quality_analysis.
        report_text = generate_labeling_report(labeled_df)
        report_path.write_text(
            report_text,
            encoding="utf-8",
        )
        print()
        print("=" * 78)
        print("STAGE 3 - LABELING SELESAI")
        print("=" * 78)
        print(f"Total data    : {len(labeled_df):,}")
        print(f"Output CSV    : {output_path}")
        print(f"Output report : {report_path}")
        print(f"Threshold     : {NEUTRAL_THRESHOLD}")
        print("=" * 78)
        return 0
    except KeyboardInterrupt:
        print("\n[WARNING] Proses dihentikan oleh pengguna.")
        return 130
    except Exception as exc:
        print()
        print("=" * 78)
        print("STAGE 3 - LABELING GAGAL")
        print("=" * 78)
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
if __name__ == "__main__":
    raise SystemExit(main())
