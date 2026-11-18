"""
Aspect Detection Service
========================
Lapisan layanan (service layer) untuk analisis aspek.

ATURAN ARSITEKTUR
=================
Deteksi aspek HANYA memiliki satu implementasi canonical:

    pipeline/aspect_detection.py

Modul ini TIDAK mengimplementasikan pencocokan kata kunci sendiri. Semua
fungsi delegasi ke pipeline/aspect_detection.py sehingga tidak ada
duplikasi yang bisa berbeda antara jalur batch (riset) dan jalur web.

DETEKSI ASPEK VS SENTIMEN ASPEK
===============================
Dua hal yang harus dibedakan:

  1. DETEKSI aspek  -> kalimat mana yang milik aspek mana.
     Metode: pencocokan kata kunci (pipeline/aspect_detection.py).

  2. SENTIMEN aspek -> label sentimen untuk sebuah aspek.
     DEFINISI CANONICAL (dipakai riset DAN web):
     sentimen sebuah aspek dihitung dari KALIMAT-KALIMAT ASPEK ITU SENDIRI,
     bukan disalin dari label sentimen tingkat review.

     Menghitung per aspek dari kalimatnya sendiri adalah definisi ABSA yang
     benar. copied-from-review bukan ABSA. Modul ini memakai satu-satunya
     artefak inferensi canonical (final_pipeline.joblib), sehingga tidak
     ada model kedua.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pipeline.aspect_detection import (
    detect_all_aspects,
    detect_aspect,
    split_sentences,
)
from shared.constants import ASPECTS, ASPECT_KEYWORDS

__all__ = [
    "ASPECTS",
    "ASPECT_KEYWORDS",
    "aggregate_aspect_distribution",
    "aspect_sentiment_labels",
    "detect_all_aspects",
    "detect_aspect_in_sentence",
    "detect_aspects_batch",
    "detect_aspects_for_review",
    "get_aspect_summary",
    "split_sentences",
]

# Skope semantik yang dipakai di respons API dan laporan.
ASPECT_SENTIMENT_SCOPE = "per_aspect_sentiment_from_aspect_sentences"


def detect_aspect_in_sentence(
    sentence: str,
    aspect_keywords: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """Deteksi aspek pada satu kalimat (delegasi ke pipeline canonical)."""
    return detect_aspect(sentence, aspect_keywords)


# =============================================================================
# REVIEW-LEVEL DETECTION
# =============================================================================

def detect_aspects_for_review(
    review_text: str,
    sentiment_label: Optional[str] = None,
    use_clauses: bool = False,
) -> Dict[str, Any]:
    """
    Deteksi aspek untuk satu ulasan.

    Args:
        review_text: Teks ulasan lengkap.
        sentiment_label: Label sentimen tingkat review (opsional). Disimpan
            sebagai referensi; BUKAN dipakai sebagai sentimen per aspek.

    Returns:
        Dict dengan:
        - detected_aspects: daftar nama aspek yang ditemukan
        - aspect_details: {aspek: {sentences, count, sentiment}}
          dengan `sentiment` = label sentimen aspek itu sendiri
        - general_sentences: kalimat tanpa aspek
        - sentiment_label: label tingkat review (referensi)
        - aspect_sentiment_scope: skope definisi sentimen per aspek
        - has_aspects: bool
    """
    if use_clauses:
        from shared.candidate_sentiment import aspect_clauses
        all_detected = aspect_clauses(review_text)
    else:
        all_detected = detect_all_aspects(review_text)

    detected_aspects = [aspect for aspect in ASPECTS if all_detected.get(aspect)]
    aspect_details = {}
    general_sentences = all_detected.get("Umum", [])

    for aspect in detected_aspects:
        sentences = all_detected[aspect]
        aspect_details[aspect] = {
            'sentences': sentences,
            'count': len(sentences),
            # Diisi oleh aspect_sentiment_labels() pada tahap berikutnya.
            'sentiment': None,
        }

    return {
        'detected_aspects': detected_aspects,
        'aspect_details': aspect_details,
        'general_sentences': general_sentences,
        'sentiment_label': sentiment_label,
        'aspect_sentiment_scope': ASPECT_SENTIMENT_SCOPE,
        'has_aspects': len(detected_aspects) > 0,
    }


def aspect_sentiment_labels(
    aspect_details: Dict[str, Dict[str, Any]],
    pipeline=None,
) -> Dict[str, Optional[str]]:
    """
    Hitung sentimen setiap aspek dari kalimat-kalimat aspek itu sendiri.

    Memakai artefak canonical (final_pipeline.joblib) + preprocessing
    canonical (shared/preprocessing.py). Tidak ada classifier kedua.
    """
    if not aspect_details:
        return {}

    if pipeline is None:
        from shared.deployment_integrity import load_verified_pipeline

        pipeline, _ = load_verified_pipeline()

    from shared.preprocessing import preprocess_text

    labels: Dict[str, Optional[str]] = {}
    for aspect, details in aspect_details.items():
        sentences = details.get('sentences') or []
        processed = preprocess_text(' '.join(sentences))
        if not processed.strip():
            labels[aspect] = None
            continue
        labels[aspect] = str(pipeline.predict([processed])[0])
    return labels


def detect_aspects_batch(reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deteksi aspek untuk banyak ulasan.

    Args:
        reviews: List of dicts dengan 'content' dan 'sentiment_label' opsional.

    Returns:
        List of aspect detection results
    """
    results = []
    for review in reviews:
        content = review.get('content') or review.get('original_content') or review.get('text', '')
        sentiment = review.get('sentiment_label') or review.get('label')
        result = detect_aspects_for_review(content, sentiment)
        results.append(result)

    return results


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_aspect_distribution(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """
    Agregasi distribusi sentimen per aspek.

    Args:
        results: List of aspect detection results. Label diambil dari
            'aspect_sentiment' bila ada (sentimen per aspek), jika tidak
            dari 'sentiment_label' (label tingkat review).

    Returns:
        Dict mapping aspect ke hitungan sentimen
    """
    distribution = {
        aspect: {'positif': 0, 'netral': 0, 'negatif': 0, 'total': 0}
        for aspect in ASPECTS
    }

    for result in results:
        sentiment = str(
            result.get('aspect_sentiment')
            or result.get('sentiment_label')
            or ''
        ).strip().lower()

        for aspect in result.get('detected_aspects', []):
            if aspect in distribution:
                distribution[aspect]['total'] += 1
                if sentiment == 'positif':
                    distribution[aspect]['positif'] += 1
                elif sentiment == 'netral':
                    distribution[aspect]['netral'] += 1
                elif sentiment == 'negatif':
                    distribution[aspect]['negatif'] += 1

    return distribution


def get_aspect_summary(
    distribution: Dict[str, Dict[str, int]],
    reviews_with_aspects: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate summary statistics for aspect distribution.

    Args:
        distribution: Aspect distribution dict
        reviews_with_aspects: Number of DISTINCT reviews that have at least
            one aspect detected. Karena satu ulasan dapat masuk ke beberapa
            aspek, jumlah kemunculan aspek (total_aspect_mentions) dapat lebih
            besar daripada jumlah ulasan unik. Nilai ini harus dihitung dari
            data review, bukan dari jumlah aspek.

    Returns:
        Summary with counts and percentages
    """
    total_aspects = sum(d['total'] for d in distribution.values())

    if reviews_with_aspects is None:
        # Fallback konservatif: dipakai hanya bila pemanggil tidak
        # menyediakan jumlah ulasan unik.
        reviews_with_aspects = sum(1 for d in distribution.values() if d['total'] > 0)

    summary = {
        'total_reviews_with_aspects': int(reviews_with_aspects),
        'total_aspect_mentions': total_aspects,
        'aspects': {}
    }

    # Per-aspect stats
    for aspect, counts in distribution.items():
        pct = (counts['total'] / total_aspects * 100) if total_aspects > 0 else 0
        summary['aspects'][aspect] = {
            'count': counts['total'],
            'percentage': round(pct, 2),
            'positif': counts['positif'],
            'netral': counts['netral'],
            'negatif': counts['negatif']
        }

    return summary
