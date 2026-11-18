"""
Dataset IO
==========
Pembacaan dataset canonical yang robust.

Dataset canonical project ini
(``data/pipeline_output/stage3_final_labeled.csv``) pernah ditulis dengan
delimiter berbeda oleh script yang berbeda (``;`` oleh run_pipeline, ``,``
oleh integrate_expert_labels). Membaca file dengan delimiter yang salah
menyebabkan baris hilang atau error parse, sehingga identitas dataset
sebenarnya tidak dapat diverifikasi.

Modul ini menyediakan SATU-SATUNYA cara membaca dataset canonical:
mencoba beberapa delimiter + encoding dan mengembalikan DataFrame yang
sudah tervalidasi (kolom teks dan label wajib ada, delimiter konsisten).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

# Delimiter yang dicoba, urut dari yang paling mungkin.
CANDIDATE_DELIMITERS: Tuple[str, ...] = (",", ";", "\t")
CANDIDATE_ENCODINGS: Tuple[str, ...] = ("utf-8-sig", "latin-1")


def detect_delimiter(path: Path, sample_size: int = 20000) -> str:
    """Deteksi delimiter berdasarkan baris header file."""
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")[:sample_size]
    header = text.splitlines()[0] if text else ""
    counts = {sep: header.count(sep) for sep in CANDIDATE_DELIMITERS}
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        return ","
    return best


def read_dataset(path: Path) -> pd.DataFrame:
    """
    Baca file CSV/TSV dengan fallback delimiter dan encoding.

    Raises:
        FileNotFoundError: bila file tidak ada.
        ValueError: bila file tidak dapat dibaca dengan delimiter mana pun.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    delimiter = detect_delimiter(path)
    errors: List[str] = []

    for encoding in CANDIDATE_ENCODINGS:
        try:
            df = pd.read_csv(
                path,
                sep=delimiter,
                encoding=encoding,
                engine="python",
            )
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")
            continue
        if len(df.columns) > 1:
            return df
        # Delimiter mungkin salah; coba delimiter lain dengan encoding ini.
        for alt in CANDIDATE_DELIMITERS:
            if alt == delimiter:
                continue
            try:
                candidate = pd.read_csv(
                    path,
                    sep=alt,
                    encoding=encoding,
                    engine="python",
                )
            except Exception:
                continue
            if len(candidate.columns) > len(df.columns):
                df = candidate
        return df

    raise ValueError(
        f"Gagal membaca dataset: {path}\n" + "\n".join(errors)
    )


def read_canonical_dataset(
    path: Path,
    text_column: str = "processed_text",
    label_column: str = "sentiment_label_final",
) -> pd.DataFrame:
    """
    Baca dataset canonical dan pastikan kolom wajib tersedia.

    Raises:
        FileNotFoundError: file tidak ada.
        ValueError: kolom wajib tidak ditemukan.
    """
    df = read_dataset(path)
    required = {text_column, label_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan pada {path}: {sorted(missing)}. "
            f"Kolom tersedia: {list(df.columns)}"
        )
    return df


__all__ = [
    "CANDIDATE_DELIMITERS",
    "CANDIDATE_ENCODINGS",
    "detect_delimiter",
    "read_dataset",
    "read_canonical_dataset",
]
