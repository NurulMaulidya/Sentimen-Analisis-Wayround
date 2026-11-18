"""
ML Engine - Pipeline Orchestration Layer
========================================
PURPOSE: Backend ML Engine adalah THIN ORCHESTRATION LAYER.
        Seluruh logika inti WAJIB dipanggil dari shared/ dan pipeline/.
        Tidak boleh ada duplikasi logika preprocessing, labeling, atau training.

Prinsip: Sesuai Metodologi Bagian 2.5 dan 8.1
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# =============================================================================
# IMPORTS FROM SHARED (canonical source of truth)
# =============================================================================
from shared.preprocessing import (
    preprocess_text,
    preprocess_dataframe,
    clean_text,
    case_fold,
    tokenize,
    normalize_slang,
    handle_negation,
    remove_stopwords,
    stem_tokens,
)
from shared.lexicon import (
    classify_sentiment,
)
from shared.constants import (
    ASPECT_KEYWORDS as SHARED_ASPECT_KEYWORDS,
)
from shared.labeling import DEFAULT_NEUTRAL_THRESHOLD, get_active_lexicon
from shared.labels import distribution as label_distribution

from backend.config import (
    ASPECTS,
    NEUTRAL_THRESHOLD,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
)

# =============================================================================
# IMPORTS FROM PIPELINE (canonical source of truth)
# =============================================================================
from pipeline.aspect_detection import (
    detect_all_aspects,
    split_sentences,
)


# =============================================================================
# TEXT PROCESSING ENGINE
# =============================================================================

class TextProcessingEngine:
    """
    Orchestrator untuk preprocessing teks.
    MEMANGGIL shared/preprocessing.py - tidak mengimplementasikan sendiri.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """Tahap 1 (cleaning) saja - untuk preview cepat."""
        return clean_text(text)

    @staticmethod
    def preprocess_full(text: str) -> str:
        """
        Preprocessing LENGKAP 7 tahap.
        WAJIB digunakan sebelum labeling dan training.
        """
        return preprocess_text(text)

    @staticmethod
    def preprocess_with_steps(text: str) -> Tuple[str, Dict]:
        """Preprocessing lengkap + detail setiap tahap."""
        return preprocess_text(text, return_steps=True)

    @staticmethod
    def process_dataframe(df, text_column: str = 'content'):
        """Process seluruh DataFrame dengan 7 tahap preprocessing."""
        return preprocess_dataframe(df, text_column, show_progress=True)

    @staticmethod
    def get_preprocessing_steps() -> List[str]:
        """Return daftar tahap preprocessing untuk dokumentasi UI."""
        return [
            "1. Cleaning - hapus URL, emoji non-sentimen, angka, tanda baca",
            "2. Case folding - huruf kecil",
            "3. Tokenizing - pecah kata",
            "4. Normalisasi slang - gantikan kata gaul",
            "5. Negation handling - gabung tidak+bagus -> tidak_bagus",
            "6. Stopword removal - hapus kata umum",
            "7. Stemming - Sastrawi",
        ]


# =============================================================================
# ASPECT DETECTION ENGINE
# =============================================================================

class AspectDetectionEngine:
    """
    Orchestrator untuk deteksi aspek.
    MEMANGGIL pipeline/aspect_detection.py dan shared/constants.py.
    """
    # GUNAKAN ASPECT_KEYWORDS dari shared/constants.py - BUKAN dari config.py
    # Ini memastikan konsistensi dengan pipeline batch dan metodologi Bagian 4.4c

    def __init__(self):
        self.aspects = ASPECTS
        # Verifikasi: gunakan canonical keywords dari shared
        self.keywords = SHARED_ASPECT_KEYWORDS

    def detect_in_text(self, text: str) -> Dict[str, List[str]]:
        """
        Deteksi aspek dalam teks.
        MEMANGGIL pipeline/aspect_detection.detect_all_aspects().
        """
        return detect_all_aspects(text)

    def detect_sentences(self, text: str) -> List[str]:
        """Pecah teks menjadi kalimat."""
        return split_sentences(text)

    def detect_in_dataframe(self, df, text_column: str = 'content'):
        """Deteksi aspek untuk seluruh DataFrame."""
        from pipeline.aspect_detection import add_aspects_to_dataframe
        return add_aspects_to_dataframe(df, text_column)

    def get_aspects(self) -> List[str]:
        """Return daftar aspek."""
        return self.aspects

    def get_keywords(self) -> Dict[str, List[str]]:
        """Return kata kunci aspek (dari shared/constants.py)."""
        return self.keywords


# =============================================================================
# SENTIMENT LABELING ENGINE
# =============================================================================

class SentimentLabelingEngine:
    """
    Orchestrator untuk pelabelan sentimen lexicon.
    MEMANGGIL shared/lexicon.py - tidak mengimplementasikan sendiri.

    CATATAN PENTING: engine ini hanya untuk jalur RISET (pelabelan dataset).
    Prediksi ulasan baru di website SELALU memakai SVM canonical, bukan
    lexicon (lihat SVMModelEngine).
    """

    def __init__(self, threshold: float = NEUTRAL_THRESHOLD):
        self.threshold = threshold
        # Lexicon singleton dari shared/lexicon.py (versi aktif, default v2)
        self.lexicon = get_active_lexicon()

    def label_text(self, processed_text: str) -> Tuple[str, float, float]:
        """
        Label satu teks (teks sudah harus dipreprocessing).
        MEMANGGIL shared/lexicon.classify_sentiment().

        Returns:
            Tuple of (label, raw_score, normalized_score)
        """
        result = classify_sentiment(
            processed_text, threshold=self.threshold, lexicon=self.lexicon
        )
        return (result['label'], result['raw_score'], result['normalized_score'])

    def label_raw_text(self, raw_text: str) -> Tuple[str, float, float]:
        """
        Label teks mentah - otomatis preprocessing + labeling.
        """
        processed = preprocess_text(raw_text)
        return self.label_text(processed)

    def label_dataframe(self, df, text_column: str = 'processed_text'):
        """Label seluruh DataFrame. MEMANGGIL pipeline/labeling.label_dataframe()."""
        from pipeline.labeling import label_dataframe
        return label_dataframe(df, text_column)

    def label_with_aspects(self, df, aspects_df, text_column: str = 'processed_text',
                           review_id_column: str = 'reviewId'):
        """Label sentimen per aspek. MEMANGGIL pipeline/labeling.label_with_aspects()."""
        from pipeline.labeling import label_with_aspects
        return label_with_aspects(df, aspects_df, text_column, review_id_column)

    def get_distribution(self, df, label_column: str = 'sentiment_label') -> Dict:
        """
        Distribusi label pada satu kolom DataFrame.

        Memakai shared.labels.distribution() sebagai single source of truth
        normalisasi label. Versi lama memanggil shared.lexicon.get_label_distribution()
        dengan argumen yang tidak kompatibel sehingga DataFrame diperlakukan
        sebagai daftar teks dan nama kolom terbaca sebagai threshold.
        """
        if label_column not in df.columns:
            raise KeyError(
                f"Kolom label '{label_column}' tidak ditemukan. "
                f"Kolom tersedia: {list(df.columns)}"
            )
        return label_distribution(df[label_column].tolist())

    def analyze_quality(self, df, label_column: str = 'sentiment_label') -> Dict:
        """Analisis kualitas pelabelan."""
        from pipeline.labeling import analyze_labeling_quality
        return analyze_labeling_quality(df, label_column)

    def get_threshold(self) -> float:
        """Return threshold netral yang digunakan."""
        return self.threshold


# =============================================================================
# SVM MODEL ENGINE
# =============================================================================

class SVMModelEngine:
    """
    Orchestrator untuk prediksi SVM memakai artefak final canonical.

    TUJUAN:
    Modul ini TIDAK melakukan training. Website adalah media pengujian
    model yang sudah dilatih (metodologi_final.md Bab 2 dan Bab 15).
    Training hanya boleh dilakukan lewat:

        python scripts/train_final_model.py

    Seluruh prediksi memakai SATU artefak canonical:

        models/final_v4/final_pipeline.joblib

    yang memuat TF-IDF (sudah di-fit) + OneVsOneClassifier + Linear SVM.
    Dengan demikian tidak ada fit()/fit_transform() pada data uji dan tidak
    ada classifier kedua di project.
    """

    def __init__(self,
                 max_features: int = TFIDF_MAX_FEATURES,
                 ngram_range: Tuple[int, int] = TFIDF_NGRAM_RANGE):
        # Gunakan konstanta dari shared - bukan hardcoded
        self.max_features = max_features
        self.ngram_range = ngram_range

        # Artefak canonical: pipeline TF-IDF + classifier (sudah di-fit).
        self.pipeline = None
        self.is_loaded = False
        self.model_version = None

    def load_canonical_pipeline(self):
        """
        Muat pipeline final canonical (tidak melakukan training).

        Memakai deployment integrity gate: metadata, hash artifact, dan
        runtime fingerprint preprocessing harus cocok, selain itu muatan
        ditolak.
        """
        from shared.deployment_integrity import load_verified_pipeline
        self.pipeline, meta = load_verified_pipeline()
        self.is_loaded = True
        self.model_version = meta["model_version"]
        return self.pipeline

    def train(self, *args, **kwargs):
        """
        DITOLAK. Website tidak boleh melatih ulang model.

        Model final sudah dikunci. Untuk training ulang (mis. setelah
        preprocessing direbuild), jalankan:

            python scripts/train_final_model.py
        """
        raise RuntimeError(
            "Training dari website dinonaktifkan. Model final sudah dikunci. "
            "Gunakan scripts/train_final_model.py untuk training ulang."
        )

    def predict(self, texts: List[str]) -> List[Dict]:
        """
        Prediksi sentimen untuk list teks.

        Memakai preprocessing canonical (shared/preprocessing.py) dan
        pipeline final canonical. Tidak ada fit() pada data ini.
        """
        if not self.is_loaded:
            self.load_canonical_pipeline()

        # Preprocess: gunakan shared/preprocessing.py (7 tahap)
        processed = [preprocess_text(t) for t in texts]

        # Transform + predict di dalam pipeline yang sudah di-fit.
        labels = self.pipeline.predict(processed)

        results = []
        for i, text in enumerate(texts):
            results.append({
                'original_text': text,
                'processed_text': processed[i],
                'predicted_label': str(labels[i]),
                'model_version': self.model_version,
                # SVM tidak menghasilkan probabilitas kalibrasi; margin
                # keputusan bukan confidence.
                'confidence': None,
            })

        return results

    def predict_single(self, text: str) -> Dict:
        """Prediksi satu teks."""
        return self.predict([text])[0]

    def predict_with_aspects(self, text: str) -> Dict:
        """
        Prediksi review + sentimen per aspek.

        DEFINISI SENTIMEN PER ASPEK (dipakai bersama di seluruh project):
        sentimen sebuah aspek dihitung dari KALIMAT-KALIMAT aspek tersebut,
        bukan disalin dari label sentimen tingkat review. Modul ini memakai
        artefak SVM canonical yang sama, sehingga tidak ada model kedua.

        Pipeline/aspect_detection.py memisahkan kalimat per aspek; setiap
        gabungan kalimat aspek dipreprocessing dengan shared/preprocessing.py
        lalu diprediksi dengan pipeline final yang sama.
        """
        prediction = self.predict_single(text)

        aspects = detect_all_aspects(text)

        aspect_sentiments = {}
        for aspect, sentences in aspects.items():
            if aspect == 'Umum' or not sentences:
                continue
            aspect_text = ' '.join(sentences)
            processed = preprocess_text(aspect_text)
            if not processed.strip():
                continue
            label = str(self.pipeline.predict([processed])[0])
            aspect_sentiments[aspect] = {
                'label': label,
                'sentences': sentences[:3],
            }

        prediction['aspects'] = aspect_sentiments
        return prediction


# =============================================================================
# MODULE-LEVEL COMPATIBILITY
# =============================================================================

__all__ = [
    "TextProcessingEngine",
    "AspectDetectionEngine",
    "SentimentLabelingEngine",
    "SVMModelEngine",
]
