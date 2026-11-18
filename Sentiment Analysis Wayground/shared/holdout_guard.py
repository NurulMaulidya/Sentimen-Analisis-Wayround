"""
Holdout Guard
=============

Mencegah baris holdout canonical masuk ke jalur pengujian web.

KONTEKS
=======
Holdout adalah 604 baris yang DIBUKA satu kali untuk laporan akhir. Latar
belakang: pada attempt yang dibatalkan, 527 dari 604 baris holdout sempat
masuk ke training/CV, dan holdout sempat diunggah ke website. Karena itu
holdout harus dijaga agar tidak dapat dipakai lagi sebagai data uji model
di website.

Fakta exposure historis TIDAK dihapus - ia didokumentasikan sebagai temuan
penelitian (lihat docs/DECISIONS/D4-KONFLIK-KONFIGURASI.md). Yang dilakukan
modul ini adalah MENJAGA ke depan:

  * berkas uji yang memuat baris holdout ditolak,
  * guard dapat dimatikan hanya lewat environment variable eksplisit
    (WAYGROUND_ALLOW_HOLDOUT_UPLOAD=1) supaya reproducible audit tetap bisa
    dilakukan, dan setiap penggunaan terekam di log.

Modul ini TIDAK mengubah dataset, label, atau preprocessing.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Set

from shared.constants import PIPELINE_OUTPUT_DIR

CANONICAL_HOLDOUT_PATH = PIPELINE_OUTPUT_DIR / "stage5_split_v2" / "holdout.csv"

ALLOW_ENV_VAR = "WAYGROUND_ALLOW_HOLDOUT_UPLOAD"

_WHITESPACE = re.compile(r"\s+")


class HoldoutLeakError(ValueError):
    """Berkas uji memuat baris holdout canonical."""


def allow_holdout_upload() -> bool:
    """True bila guard sengaja dimatikan lewat environment variable."""
    return os.environ.get(ALLOW_ENV_VAR, "").strip().lower() in {"1", "true", "yes"}


def _normalize_content(value: object) -> str:
    """Normalisasi teks untuk pencocokan holdout (case + spasi)."""
    if value is None:
        return ""
    try:
        import pandas as pd

        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return _WHITESPACE.sub(" ", str(value)).strip().lower()


def _content_digest(value: object) -> str:
    return hashlib.sha256(_normalize_content(value).encode("utf-8")).hexdigest()


def _read_split(path: Path) -> tuple:
    """Read canonical review IDs and text digests from one split."""
    if not path.exists():
        return set(), set()

    import pandas as pd

    frame = pd.read_csv(path, dtype=str)
    review_ids: Set[str] = set()
    if "reviewId" in frame.columns:
        review_ids = {v.strip() for v in frame["reviewId"].dropna() if v.strip()}

    digests: Set[str] = set()
    for column in ("content", "processed_text"):
        if column in frame.columns:
            digests.update(_content_digest(v) for v in frame[column].dropna())
            digests.discard(hashlib.sha256(b"").hexdigest())
    return review_ids, digests


def _read_holdout() -> tuple:
    return _read_split(CANONICAL_HOLDOUT_PATH)


def canonical_origins(df, content_column):
    """Return a split status per source row using review ID and normalized text."""
    paths = {
        "development": PIPELINE_OUTPUT_DIR / "stage5_split_v2" / "development.csv",
        "holdout": CANONICAL_HOLDOUT_PATH,
    }
    df = df.reset_index(drop=True)
    origins = [[] for _ in range(len(df))]
    split_digests = {}
    for split, path in paths.items():
        ids, digests = _read_split(path)
        if not ids and not digests:
            raise ValueError(f"Dataset canonical {split} tidak tersedia untuk pemeriksaan asal data.")
        split_digests[split] = digests
        for index in _matching_rows(df, content_column, ids, digests):
            origins[index].append(split)

    from shared.preprocessing import preprocess_text

    processed = {}
    for index, value in enumerate(df[content_column].fillna('')):
        text = str(value)
        if origins[index] or not text.strip():
            continue
        if text not in processed:
            final = preprocess_text(text)
            digest = _content_digest(final) if final else None
            processed[text] = [split for split, digests in split_digests.items() if digest in digests]
        origins[index] = processed[text]
    return [" + ".join(matches) or "Tidak cocok dengan split canonical" for matches in origins]


def canonical_holdout_review_ids() -> Set[str]:
    return _read_holdout()[0]


def canonical_holdout_content_digests() -> Set[str]:
    return _read_holdout()[1]


def find_holdout_rows(df, content_column: Optional[str] = None) -> List[int]:
    """
    Indeks baris pada ``df`` yang cocok dengan holdout canonical.

    Pencocokan memakai dua sinyal:
      1. reviewId (bila kolom ada), dan
      2. hash teks ulasan (content atau processed_text).
    """
    if df is None or len(df) == 0:
        return []

    review_ids, digests = _read_holdout()
    if not review_ids and not digests:
        return []

    return _matching_rows(df, content_column, review_ids, digests)


def _matching_rows(df, content_column, review_ids, digests):
    matched = set()
    for column in df.columns:
        if str(column).strip().lower() in {"reviewid", "review_id", "review id", "id ulasan"}:
            values = df[column].astype(str).str.strip()
            matched.update(int(index) for index in df.index[values.isin(review_ids)])

    if digests:
        text_aliases = {"content", "processed_text", "teks preprocessing", "ulasan asli"}
        for column in df.columns:
            if column != content_column and str(column).strip().lower() not in text_aliases:
                continue
            hashes = df[column].map(_content_digest)
            mask = hashes.isin(digests)
            matched.update(int(index) for index in df.index[mask])

    return sorted(matched)


def assert_no_holdout_rows(df, content_column: Optional[str] = None) -> None:
    """
    Lempar HoldoutLeakError bila ``df`` memuat baris holdout canonical.

    Dilempar sebagai ValueError agar lapisan API dapat menanganinya sebagai
    kesalahan validasi berkas uji (HTTP 400), bukan kegagalan server.
    """
    if allow_holdout_upload():
        print(
            "[HOLDOUT-GUARD] PERINGATAN: guard dimatikan lewat "
            f"{ALLOW_ENV_VAR}=1. Berkas holdout diterima."
        )
        return

    matched = find_holdout_rows(df, content_column)
    if matched:
        raise HoldoutLeakError(
            f"Berkas uji memuat {len(matched)} baris yang berasal dari holdout "
            "canonical (604 baris data uji akhir). Holdout tidak boleh "
            "diunggah atau dipakai sebagai data uji model di website. "
            "Gunakan data lain yang tidak berasal dari holdout."
        )


__all__ = [
    "ALLOW_ENV_VAR",
    "CANONICAL_HOLDOUT_PATH",
    "HoldoutLeakError",
    "allow_holdout_upload",
    "assert_no_holdout_rows",
    "canonical_holdout_content_digests",
    "canonical_holdout_review_ids",
    "find_holdout_rows",
]
