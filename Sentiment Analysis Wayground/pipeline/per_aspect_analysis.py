"""Analisis sentimen per aspek.

Script ini TIDAK melatih SVM baru.

Sumber sentimen aspek:
    sentiment_label_final

Kolom sentiment_label_final harus berasal dari dataset canonical
stage3_final_labeled.csv setelah integrasi label pakar.

Alur:
    stage3_final_labeled.csv
            ↓
    stage4_aspects.csv
            ↓
    sentiment_label_final
            ↓
    Analisis sentimen per aspek

CLI:
    python pipeline/per_aspect_analysis.py

    python pipeline/per_aspect_analysis.py \
        --input data/pipeline_output/stage4_aspects.csv

Output:
    data/pipeline_output/aspect_sentiment_summary.json
    data/pipeline_output/stage4_aspects_summary.xlsx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from shared.constants import SENTIMENT_CLASSES, ASPECTS  # noqa: E402


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "pipeline_output"
    / "stage4_aspects.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "pipeline_output"
)

# ============================================================
# KONFIGURASI
# ============================================================

# WAJIB menggunakan label final.
# Jangan fallback ke sentiment_label_inset karena penelitian
# menggunakan label final setelah integrasi validasi pakar.
LABEL_COLUMN = "sentiment_label_final"

EXPECTED_SENTIMENTS = {
    "positif": "Positif",
    "netral": "Netral",
    "negatif": "Negatif",
}


# ============================================================
# HELPER
# ============================================================

def _normalize_sentiment(value: object) -> str:
    """Normalisasi label sentimen ke format canonical."""

    if pd.isna(value):
        return ""

    value = str(value).strip().casefold()

    mapping = {
        "positif": "Positif",
        "positive": "Positif",
        "pos": "Positif",
        "netral": "Netral",
        "neutral": "Netral",
        "neu": "Netral",
        "negatif": "Negatif",
        "negative": "Negatif",
        "neg": "Negatif",
    }

    return mapping.get(value, "")


def _validate_input(df: pd.DataFrame) -> list[str]:
    """Validasi struktur minimum file aspek."""

    errors = []

    required_columns = [
        "aspect",
        LABEL_COLUMN,
    ]

    for column in required_columns:
        if column not in df.columns:
            errors.append(
                f"Kolom wajib '{column}' tidak ditemukan."
            )

    return errors


def _normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Membuat kolom internal label canonical."""

    result = df.copy()

    result["_sentiment_normalized"] = (
        result[LABEL_COLUMN]
        .apply(_normalize_sentiment)
    )

    return result


def _check_invalid_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Mengambil baris dengan label kosong/tidak dikenal."""

    invalid_mask = (
        df["_sentiment_normalized"]
        .eq("")
    )

    return df.loc[invalid_mask].copy()


# ============================================================
# ANALISIS
# ============================================================

def run_per_aspect_analysis(
    df_aspects: pd.DataFrame,
    output_dir: Path | None = None,
) -> dict:
    """Analisis distribusi sentimen untuk setiap aspek."""

    if df_aspects is None or df_aspects.empty:
        return {}

    errors = _validate_input(df_aspects)

    if errors:
        raise ValueError(
            "Validasi input gagal:\n- "
            + "\n- ".join(errors)
        )

    df = _normalize_labels(df_aspects)

    # --------------------------------------------------------
    # Validasi label
    # --------------------------------------------------------

    invalid_df = _check_invalid_labels(df)

    if not invalid_df.empty:
        invalid_values = (
            df.loc[
                df["_sentiment_normalized"].eq(""),
                LABEL_COLUMN,
            ]
            .astype(str)
            .value_counts(dropna=False)
            .to_dict()
        )

        print(
            "[WARNING] Ditemukan label kosong/tidak dikenal:"
        )
        print(invalid_values)

    # Hanya gunakan label yang valid.
    df_valid = df[
        df["_sentiment_normalized"].isin(
            SENTIMENT_CLASSES
        )
    ].copy()

    if df_valid.empty:
        raise ValueError(
            "Tidak ada label sentimen valid yang dapat dianalisis."
        )

    # --------------------------------------------------------
    # Analisis per aspek
    # --------------------------------------------------------

    results = {}

    for aspect in ASPECTS:

        d = df_valid[
            df_valid["aspect"].astype(str).str.strip()
            == str(aspect).strip()
        ].copy()

        if d.empty:
            results[aspect] = {
                "status": "no_data",
                "label_column": LABEL_COLUMN,
                "samples": 0,
                "unique_reviews": 0,
                "distribution": {
                    sentiment: {
                        "count": 0,
                        "percentage": 0.0,
                    }
                    for sentiment in SENTIMENT_CLASSES
                },
            }

            continue

        counts = (
            d["_sentiment_normalized"]
            .value_counts()
            .reindex(
                SENTIMENT_CLASSES,
                fill_value=0,
            )
        )

        total = int(len(d))

        unique_reviews = (
            int(d["reviewId"].nunique())
            if "reviewId" in d.columns
            else None
        )

        results[aspect] = {
            "status": "complete",
            "label_column": LABEL_COLUMN,
            "samples": total,
            "unique_reviews": unique_reviews,
            "distribution": {
                sentiment: {
                    "count": int(counts[sentiment]),
                    "percentage": round(
                        float(counts[sentiment])
                        / total
                        * 100,
                        2,
                    )
                    if total
                    else 0.0,
                }
                for sentiment in SENTIMENT_CLASSES
            },
        }

    # --------------------------------------------------------
    # Simpan JSON
    # --------------------------------------------------------

    if output_dir:
        out = Path(output_dir)
        out.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_path = (
            out
            / "aspect_sentiment_summary.json"
        )

        json_path.write_text(
            json.dumps(
                results,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return results


# ============================================================
# DATAFRAME OUTPUT
# ============================================================

def _to_frame(results: dict) -> pd.DataFrame:
    """Mengubah hasil analisis menjadi tabel."""

    rows = []

    for aspect, payload in results.items():

        distribution = payload.get(
            "distribution",
            {},
        )

        rows.append(
            {
                "Aspek": aspect,
                "Status": payload.get(
                    "status"
                ),
                "Kolom Label": payload.get(
                    "label_column"
                ),
                "Jumlah Baris": payload.get(
                    "samples",
                    0,
                ),
                "Ulasan Unik": payload.get(
                    "unique_reviews"
                ),
                "Positif": distribution.get(
                    "Positif",
                    {},
                ).get(
                    "count",
                    0,
                ),
                "Netral": distribution.get(
                    "Netral",
                    {},
                ).get(
                    "count",
                    0,
                ),
                "Negatif": distribution.get(
                    "Negatif",
                    {},
                ).get(
                    "count",
                    0,
                ),
                "Persentase Positif": distribution.get(
                    "Positif",
                    {},
                ).get(
                    "percentage",
                    0,
                ),
                "Persentase Netral": distribution.get(
                    "Netral",
                    {},
                ).get(
                    "percentage",
                    0,
                ),
                "Persentase Negatif": distribution.get(
                    "Negatif",
                    {},
                ).get(
                    "percentage",
                    0,
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Analisis distribusi sentimen "
            "per aspek pembelajaran."
        ),
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=(
            "File CSV Stage 4 "
            "yang akan dianalisis."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Folder output hasil analisis."
        ),
    )

    args = parser.parse_args()

    source = Path(args.input)

    # --------------------------------------------------------
    # File input
    # --------------------------------------------------------

    if not source.exists():

        print(
            f"ERROR: File aspek tidak ditemukan:\n"
            f"{source}"
        )

        print(
            "\nJalankan pipeline lebih dulu:"
        )

        print(
            "python pipeline/run_pipeline.py"
        )

        return 1

    print("=" * 70)
    print("ANALISIS SENTIMEN PER ASPEK")
    print("=" * 70)

    print(
        f"[INPUT] {source}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    try:

        df = pd.read_csv(
            source,
            sep=None,
            engine="python",
            encoding="utf-8-sig",
        )

    except Exception as exc:

        print(
            f"ERROR: Gagal membaca file:\n{exc}"
        )

        return 1

    print(
        f"[INFO] Jumlah baris Stage 4 : {len(df)}"
    )

    print(
        f"[INFO] Kolom label digunakan: "
        f"{LABEL_COLUMN}"
    )

    # --------------------------------------------------------
    # Validasi struktur
    # --------------------------------------------------------

    errors = _validate_input(df)

    if errors:

        print(
            "\nERROR: Struktur file tidak valid:"
        )

        for error in errors:
            print(
                f"  - {error}"
            )

        return 1

    # --------------------------------------------------------
    # Informasi label awal
    # --------------------------------------------------------

    print()
    print("DISTRIBUSI LABEL INPUT")
    print("-" * 70)

    print(
        df[LABEL_COLUMN]
        .value_counts(dropna=False)
        .to_string()
    )

    # --------------------------------------------------------
    # Analisis
    # --------------------------------------------------------

    output_dir = Path(
        args.output_dir
    )

    try:

        results = run_per_aspect_analysis(
            df,
            output_dir,
        )

    except Exception as exc:

        print(
            f"\nERROR: Analisis gagal:\n{exc}"
        )

        return 1

    if not results:

        print(
            "ERROR: Tidak ada data aspek "
            "untuk dianalisis."
        )

        return 1

    # --------------------------------------------------------
    # DataFrame hasil
    # --------------------------------------------------------

    frame = _to_frame(results)

    # --------------------------------------------------------
    # Validasi hasil
    # --------------------------------------------------------

    print()
    print("VALIDASI HASIL")
    print("-" * 70)

    validation_ok = True

    for _, row in frame.iterrows():

        total = int(
            row["Jumlah Baris"]
        )

        sentiment_total = (
            int(row["Positif"])
            + int(row["Netral"])
            + int(row["Negatif"])
        )

        if total != sentiment_total:

            validation_ok = False

            print(
                f"[ERROR] {row['Aspek']}: "
                f"total={total}, "
                f"sentimen={sentiment_total}"
            )

        else:

            print(
                f"[OK] {row['Aspek']}: "
                f"{total} baris = "
                f"{sentiment_total} sentimen"
            )

    # --------------------------------------------------------
    # Excel
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    excel_path = (
        output_dir
        / "stage4_aspects_summary.xlsx"
    )

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl",
    ) as writer:

        frame.to_excel(
            writer,
            sheet_name="Ringkasan Per Aspek",
            index=False,
        )

        # Tambahkan data agregat
        # agar hasil audit mudah dilakukan.
        overall = pd.DataFrame(
            {
                "Label": SENTIMENT_CLASSES,
                "Jumlah": [
                    int(
                        (
                            df[
                                LABEL_COLUMN
                            ]
                            .apply(
                                _normalize_sentiment
                            )
                            == sentiment
                        ).sum()
                    )
                    for sentiment in SENTIMENT_CLASSES
                ],
            }
        )

        overall["Persentase"] = (
            overall["Jumlah"]
            / overall["Jumlah"].sum()
            * 100
        ).round(2)

        overall.to_excel(
            writer,
            sheet_name="Distribusi Keseluruhan",
            index=False,
        )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ANALISIS PER ASPEK SELESAI")
    print("=" * 70)

    print(
        frame.to_string(
            index=False
        )
    )

    print()

    print(
        f"JSON  : "
        f"{output_dir / 'aspect_sentiment_summary.json'}"
    )

    print(
        f"Excel : "
        f"{excel_path}"
    )

    print()

    if validation_ok:

        print(
            "[PASS] Validasi distribusi sentimen "
            "per aspek berhasil."
        )

    else:

        print(
            "[FAIL] Terdapat ketidaksesuaian "
            "jumlah baris dan sentimen."
        )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())