"""
Upload Service
==============
Validasi file uji (CSV/TSV) secara terpusat.

Modul ini adalah SATU-SATUNYA tempat mendefinisikan:
  * ekstensi file uji yang didukung,
  * batas jumlah baris,
  * deteksi delimiter,
  * deteksi kolom ulasan (wajib) dan kolom label aktual (opsional),
  * validasi nilai label aktual terhadap kelas yang didukung.

Backend/app.py dan backend/api/analysis.py sama-sama memakai modul ini
sehingga validasi file uji identik di seluruh endpoint.

Aturan metodologi (metodologi_final.md Bab 15 & 19):
  - Website hanya menguji model final; tidak ada training di jalur ini.
  - Label aktual bersifat opsional. Jika ada, nilainya harus peta ke salah
    satu dari tiga kelas riset (positif / netral / negatif).
  - Nilai label yang tidak dikenal harus ditolak dengan pesan jelas, bukan
    diam-diam diabaikan.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from shared.labels import CANONICAL_CLASSES, normalize_label

# Kolom ulasan yang dikenali pada file uji pengguna.
POSSIBLE_CONTENT_COLUMNS = [
    "content", "ulasan asli", "text", "review", "ulasan", "review_text",
    "komentar", "teks", "processed_text",
]

# Kolom label aktual yang dikenali pada file uji pengguna.
POSSIBLE_LABEL_COLUMNS = [
    "sentiment_label_final", "sentiment_label", "sentiment", "label",
    "label_sentimen", "kelas", "class", "target", "label_aktual", "actual_label", "sentimen",
]
EXPERT_LABEL_COLUMNS = (
    "label_pakar_final", "label pakar final", "label_final_pakar", "label final pakar", "sentiment_label_expert_final",
    "label pakar", "label_pakar", "sentiment_label_expert", "label_pakar_1", "label_pakar_2",
)

# Ekstensi file uji yang didukung (CSV dan TSV).
ALLOWED_UPLOAD_EXTENSIONS = {"csv", "tsv", "txt"}

# Batas jumlah baris per file uji.
MAX_UPLOAD_ROWS = 20000
NEW_DATA = "new_data"
RESEARCH_DEMO = "research_demo"


class UploadValidationError(ValueError):
    """Kesalahan validasi file uji yang aman untuk ditampilkan ke pengguna."""

    def __init__(self, message: str, **extra):
        super().__init__(message)
        self.message = message
        self.extra = extra


def allowed_file(filename: Optional[str]) -> bool:
    """True jika ekstensi file uji didukung."""
    if not filename or "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in ALLOWED_UPLOAD_EXTENSIONS


def detect_delimiter(path: Path) -> str:
    """Deteksi delimiter file CSV/TSV berdasarkan isi file."""
    sample = Path(path).read_text(encoding="utf-8-sig", errors="replace")[:20000]
    counts = {sep: sample.count(sep) for sep in (",", ";", "\t")}
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        return ","
    return best


def read_uploaded_dataframe(path: Path) -> pd.DataFrame:
    """
    Baca file uji dengan fallback encoding (utf-8-sig lalu latin-1).

    Semua kolom dibaca sebagai string supaya label tidak terkonversi
    menjadi angka (mis. label "1" menjadi integer 1).
    """
    separator = detect_delimiter(path)
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            skiprows = 0
            with path.open(encoding=encoding, newline="") as source:
                reader = csv.reader(source, delimiter=separator)
                first = next(reader, [])
                if first and first[0].strip().lower().startswith("form validasi pakar"):
                    for row in reader:
                        columns = {cell.strip().lower() for cell in row}
                        if "ulasan asli" in columns and "label" in columns:
                            break
                    else:
                        raise ValueError("Header tabel validasi pakar tidak ditemukan.")
                    skiprows = reader.line_num - 1
            frame = pd.read_csv(
                path, sep=separator, encoding=encoding, engine="python", dtype=str,
                keep_default_na=False, skip_blank_lines=False, skiprows=skiprows,
            )
            return frame.rename(columns=str.strip) if skiprows else frame
        except UnicodeDecodeError:
            if encoding == "latin-1":
                raise


def find_content_column(df: pd.DataFrame) -> Optional[str]:
    """Cari kolom yang memuat teks ulasan."""
    columns = {str(col).strip().lower(): col for col in df.columns}
    return next((columns[key] for key in POSSIBLE_CONTENT_COLUMNS if key in columns), None)


def label_candidates(df: pd.DataFrame) -> list[str]:
    aliases = set(POSSIBLE_LABEL_COLUMNS) | set(EXPERT_LABEL_COLUMNS)
    return [col for col in df.columns if str(col).strip().lower() in aliases]


def find_label_column(df: pd.DataFrame) -> Optional[str]:
    candidates = label_candidates(df)
    return candidates[0] if len(candidates) == 1 else None


def label_issue_rows(df, column):
    issues = {"empty": [], "invalid": []}
    for row, raw in enumerate(df[column], start=1):
        value = normalize_label(raw)
        if value is None:
            issues["empty"].append(row)
        elif value not in CANONICAL_CLASSES:
            issues["invalid"].append(row)
    return issues


def has_usable_labels(df: pd.DataFrame, label_col: Optional[str]) -> bool:
    """
    True bila file benar-benar membawa label aktual yang bisa dibandingkan.

    Keberadaan kolom label saja TIDAK cukup: bila semua nilainya kosong,
    file tidak memiliki label aktual dan tidak boleh ikut dievaluasi.
    Definisi ini dipakai oleh ``validate_upload`` dan ``count_rows`` supaya
    tidak ada dua definisi "punya label" yang bisa berbeda.
    """
    if not label_col or label_col not in df.columns:
        return False
    return any(normalize_label(value) is not None for value in df[label_col])


def validate_upload(
    df: pd.DataFrame, content_column=None, label_column=None, *, mode=NEW_DATA, origins_out=None,
) -> Tuple[str, Optional[str], bool]:
    """
    Validasi struktur file uji.

    Returns:
        (content_column, label_column_or_None, has_actual_labels)

    Raises:
        UploadValidationError: bila file tidak memenuhi syarat.
    """
    if df is None or df.empty:
        raise UploadValidationError("File kosong atau tidak memiliki data.")

    if len(df) > MAX_UPLOAD_ROWS:
        raise UploadValidationError(
            f"Jumlah baris melebihi batas ({MAX_UPLOAD_ROWS}). "
            f"File berisi {len(df)} baris."
        )

    content_col = content_column or find_content_column(df)
    if content_column and content_column not in df.columns:
        raise UploadValidationError('Kolom ulasan yang dipilih tidak ditemukan.')
    if not content_col:
        raise UploadValidationError(
            "Kolom ulasan tidak ditemukan. File harus memiliki satu dari kolom: "
            + ", ".join(POSSIBLE_CONTENT_COLUMNS),
            columns=list(df.columns),
        )

    candidates = [col for col in label_candidates(df) if col != content_col]
    if not label_column and len(candidates) > 1:
        raise UploadValidationError(
            "Beberapa kolom label terdeteksi. Pilih kolom label aktual atau opsi tanpa label.",
            label_candidates=candidates,
        )
    label_col = None if label_column == '__none__' else (label_column or next(iter(candidates), None))
    if label_col is not None and (label_col not in df.columns or label_col == content_col):
        raise UploadValidationError('Kolom label yang dipilih tidak valid.')

    if label_col is not None:
        issues = label_issue_rows(df, label_col)
        invalid = issues["invalid"]
        if invalid:
            values = df[label_col].iloc[[row - 1 for row in invalid]]
            unknown = sorted({normalize_label(value) for value in values})
            raise UploadValidationError(
                f"Kolom label berisi nilai yang tidak dikenal atau ambigu pada {len(invalid)} baris. "
                "Label yang didukung adalah negatif, netral, dan positif.",
                label_column=label_col, unknown_labels=unknown, label_issues=issues,
            )

    # Label aktual dianggap ada hanya bila ada minimal satu nilai yang bisa
    # dipetakan ke kelas riset (lihat has_usable_labels).
    has_labels = has_usable_labels(df, label_col)

    if mode not in (NEW_DATA, RESEARCH_DEMO):
        raise UploadValidationError("Mode analisis tidak dikenal.")

    from shared.holdout_guard import canonical_origins

    try:
        origins = canonical_origins(df, content_col)
    except (ValueError, OSError) as exc:
        raise UploadValidationError(str(exc)) from exc
    if mode == NEW_DATA:
        development = sum("development" in origin for origin in origins)
        holdout = sum("holdout" in origin for origin in origins)
        if development or holdout:
            raise UploadValidationError(
                f"Mode Pengujian Data Baru menolak data canonical "
                f"(development: {development}, holdout: {holdout}). "
                "Gunakan Mode Demonstrasi Data Penelitian untuk meninjau ulasan tersebut."
            )
    if origins_out is not None:
        origins_out.extend(origins)

    return content_col, label_col, has_labels


def count_rows(
    df: pd.DataFrame,
    content_col: str,
    label_col: Optional[str] = None,
    has_labels: Optional[bool] = None,
) -> dict:
    """
    Hitung jumlah baris valid / dilewati pada file uji.

    Baris dengan ulasan kosong dilewati. Baris dengan label kosong (bila
    file memang membawa label aktual) TETAP diproses sebagai prediksi tanpa
    pembanding; angka tersebut dilaporkan terpisah, bukan dianggap dilewati.

    Args:
        has_labels: bila diberikan, dipakai apa adanya. Bila tidak, dihitung
            dengan ``has_usable_labels`` supaya definisi "punya label aktual"
            hanya ada di satu tempat.
    """
    if has_labels is None:
        has_labels = has_usable_labels(df, label_col)

    saved = 0
    skipped_empty = 0
    skipped_no_label = 0

    for _, row in df.iterrows():
        content = str(row[content_col]) if pd.notna(row.get(content_col)) else ""
        if not content.strip():
            skipped_empty += 1
            continue
        if has_labels and normalize_label(row.get(label_col)) is None:
            skipped_no_label += 1
        saved += 1

    return {
        "rows_saved": saved,
        "rows_skipped_empty": skipped_empty,
        "rows_without_actual_label": skipped_no_label,
    }


__all__ = [
    "ALLOWED_UPLOAD_EXTENSIONS",
    "MAX_UPLOAD_ROWS",
    "POSSIBLE_CONTENT_COLUMNS",
    "POSSIBLE_LABEL_COLUMNS",
    "UploadValidationError",
    "allowed_file",
    "count_rows",
    "detect_delimiter",
    "find_content_column",
    "find_label_column",
    "has_usable_labels",
    "read_uploaded_dataframe",
    "validate_upload",
]
