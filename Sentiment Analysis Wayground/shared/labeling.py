"""
Wayground - Satu-Satunya Pintu Masuk Pelabelan Otomatis
========================================================

Modul ini menjadi SATU-SATUNYA implementasi pelabelan sentimen berbasis
lexicon pada project ini.

Consumer:
- pipeline/run_pipeline.py  (Stage 3 - pelabelan dataset)
- scripts/audit_oov.py      (audit OOV)
- scripts/evaluasi_per_aspek.py
- pipeline/labeling.py      (adapter)

Tujuan modul ini (hasil audit Tahap 1):
- Menghilangkan jalur ganda pelabelan.
- Men'Iran: lexicon yang aktif harus dapat dipilih tanpa mengubah kode.
- Menyediakan ambang netral (neutral threshold) sebagai satu konstanta.

Pemilihan versi lexicon:
- Default: shared.constants.DEFAULT_LEXICON_VERSION ("v2").
- Override: environment variable WAYGROUND_LEXICON_VERSION="v2".

Fungsi utama:
    label_text(text, threshold=None)          -> dict
    label_series(series, threshold=None)      -> pandas.Series of label
    get_active_lexicon()                      -> shared.lexicon.TwoLexicon
"""

from __future__ import annotations

from typing import Iterable, List, Optional

from .constants import (
    resolve_lexicon_version,
)
from .lexicon import (
    TwoLexicon,
    classify_sentiment,
    get_lexicon,
    get_lexicon_statistics,
)
from .research_config import DEPLOYMENT_NEUTRAL_THRESHOLD


# ============================================================================
# THRESHOLD
# ============================================================================

# Ambang netral yang dipakai Stage 3.
# Skor > threshold  -> Positif
# Skor < -threshold -> Negatif
# selain itu        -> Netral
#
# Deployment uses Experiment 2's selected threshold. Preparation snapshots
# explicitly pass BASELINE_NEUTRAL_THRESHOLD to preserve their own recipe.
DEFAULT_NEUTRAL_THRESHOLD = DEPLOYMENT_NEUTRAL_THRESHOLD


# ============================================================================
# LEXICON AKTIF
# ============================================================================

def get_active_lexicon_version() -> str:
    """Versi lexicon aktif: dari environment variable, else default."""
    return resolve_lexicon_version()


def get_active_lexicon() -> TwoLexicon:
    """Objek lexicon yang sedang aktif."""
    return get_lexicon(get_active_lexicon_version())


def lexicon_info() -> dict:
    """Ringkasan lexicon aktif untuk logging dan laporan."""
    version = get_active_lexicon_version()
    stats = get_lexicon_statistics(get_lexicon(version))
    stats["threshold"] = DEFAULT_NEUTRAL_THRESHOLD
    return stats


# ============================================================================
# PELABELAN
# ============================================================================

def label_text(
    text: str,
    threshold: Optional[float] = None,
    lexicon: Optional[TwoLexicon] = None,
) -> dict:
    """
    Melabeli satu teks dan mengembalikan dict hasil.

    Mengembalikan kunci:
        label, raw_score, normalized_score, threshold, method,
        token_count, lexicon_version
    """
    active = lexicon if lexicon is not None else get_active_lexicon()
    limit = DEFAULT_NEUTRAL_THRESHOLD if threshold is None else threshold

    return classify_sentiment(text, threshold=limit, lexicon=active)


def label_texts(
    texts: Iterable[str],
    threshold: Optional[float] = None,
    lexicon: Optional[TwoLexicon] = None,
) -> List[dict]:
    """Melabeli kumpulan teks."""
    return [label_text(text, threshold, lexicon) for text in texts]


def label_series(
    series,
    threshold: Optional[float] = None,
    lexicon: Optional[TwoLexicon] = None,
):
    """
    Melabeli pandas.Series dan mengembalikan Series berisi label.

    Series input harus sudah berupa processed_text (hasil preprocessing),
    karena pelabelan lexicon bekerja pada token stemmed.
    """
    active = lexicon if lexicon is not None else get_active_lexicon()
    limit = DEFAULT_NEUTRAL_THRESHOLD if threshold is None else threshold

    return series.fillna("").astype(str).map(
        lambda text: classify_sentiment(
            text, threshold=limit, lexicon=active
        )["label"]
    )


def label_dataframe(
    df,
    text_column: str = "processed_text",
    threshold: Optional[float] = None,
    lexicon: Optional[TwoLexicon] = None,
    score_column: str = "sentiment_score",
    normalized_column: str = "sentiment_score_normalized",
):
    """
    Melabeli DataFrame dan menambahkan kolom label + skor.

    Kolom yang ditambahkan:
        sentiment_label_inset, sentiment_score, sentiment_score_normalized
    """
    if text_column not in df.columns:
        raise KeyError(
            f"Kolom teks '{text_column}' tidak ditemukan. "
            f"Kolom tersedia: {list(df.columns)}"
        )

    active = lexicon if lexicon is not None else get_active_lexicon()
    limit = DEFAULT_NEUTRAL_THRESHOLD if threshold is None else threshold

    results = (
        df[text_column]
        .fillna("")
        .astype(str)
        .map(
            lambda text: classify_sentiment(
                text, threshold=limit, lexicon=active
            )
        )
    )

    out = df.copy()
    out["sentiment_label_inset"] = results.map(lambda r: r["label"])
    out[score_column] = results.map(lambda r: r["raw_score"])
    out[normalized_column] = results.map(lambda r: r["normalized_score"])

    return out


__all__ = [
    "DEFAULT_NEUTRAL_THRESHOLD",
    "get_active_lexicon",
    "get_active_lexicon_version",
    "label_dataframe",
    "label_series",
    "label_text",
    "label_texts",
    "lexicon_info",
]
