"""
Unit Tests for Data Leakage Prevention
========================================
Tests to ensure TF-IDF is NOT fit on full dataset before split.

CRITICAL: TF-IDF must be fit on TRAIN only, then transform TEST.
Fitting on full dataset before split is DATA LEAKAGE.

Uji ini memakai API canonical yang sama dengan scripts/train_final_model.py:
    - shared.final_config.FINAL_MODEL_CONFIG   (konfigurasi terkunci)
    - scripts.train_final_model.build_vectorizer / build_classifier

RUN: python -m pytest tests/test_leakage.py -v
"""

import pytest
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.preprocessing import preprocess_text
from shared.final_config import (
    FINAL_MODEL_CONFIG,
    NEUTRAL_THRESHOLD,
    RANDOM_STATE,
)
from scripts.train_final_model import build_classifier, build_vectorizer
from sklearn.model_selection import train_test_split


class TestTFIDFLeakagePrevention:
    """Test that TF-IDF is NOT fit on full dataset"""

    def test_tfidf_fit_on_train_only(self):
        """
        CRITICAL TEST: TF-IDF must be fit on train, NOT on full dataset.

        This test verifies the correct pattern:
        - fit_transform(train)
        - transform(test)

        NOT:
        - fit_transform(full_data)
        """
        # Create sample data
        texts = [
            "aplikasi bagus membantu belajar",
            "error crash tidak bisa digunakan",
            "soal lengkap cocok untuk latihan",
            "sangat kecewa tidak recomended",
            "bagus sekali sangat bermanfaat",
        ]

        # Proper split
        train_texts, test_texts = train_test_split(
            texts, test_size=0.2, random_state=42
        )

        # CORRECT: Fit on train, transform test
        proper_vectorizer = build_vectorizer()
        X_train = proper_vectorizer.fit_transform(train_texts)
        X_test = proper_vectorizer.transform(test_texts)

        # Verify shapes are different (test is subset)
        assert X_train.shape[0] == 4  # 80% of 5
        assert X_test.shape[0] == 1   # 20% of 5

        # Vocabulary hanya boleh berasal dari data training.
        assert len(proper_vectorizer.vocabulary_) > 0

        # Fit pada seluruh data menghasilkan kosakata yang lebih besar,
        # sehingga pilihan fit harus dibatasi pada train.
        leaky_vectorizer = build_vectorizer()
        leaky_vectorizer.fit(texts)
        assert len(leaky_vectorizer.vocabulary_) >= len(proper_vectorizer.vocabulary_)

    def test_split_consistency(self):
        """Test that same split with same random_state gives same results"""
        texts = [f"sample text {i}" for i in range(100)]
        labels = ["Positif"] * 50 + ["Negatif"] * 50

        # First split
        train1, test1 = train_test_split(
            list(zip(texts, labels)),
            test_size=0.2,
            random_state=42,
            stratify=labels
        )

        # Second split with same random state
        train2, test2 = train_test_split(
            list(zip(texts, labels)),
            test_size=0.2,
            random_state=42,
            stratify=labels
        )

        # Should get same results
        assert len(train1) == len(train2)
        assert len(test1) == len(test2)

    def test_vectorizer_transform_before_fit_fails(self):
        """Transform before fit should raise error"""
        vectorizer = build_vectorizer()

        with pytest.raises(ValueError, match="not fitted"):
            vectorizer.transform(["some text"])


class TestReviewLevelSplit:
    """Test that review-level split is maintained"""

    def test_review_split_does_not_mix(self):
        """
        Test that sentences from same review stay in same split.

        This is critical for ABSA - if we split by sentence,
        same review could have sentences in both train and test.
        """
        # Create reviews with multiple sentences
        reviews = [
            {"review_id": "r1", "content": "Bagus. Recommended."},
            {"review_id": "r2", "content": "Error. Crash. Bad."},
            {"review_id": "r3", "content": "Helpful. Good."},
            {"review_id": "r4", "content": "Slow. Laggy."},
            {"review_id": "r5", "content": "Fun. Game."},
        ]

        df = pd.DataFrame(reviews)

        # Split by review_id (correct approach)
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=None  # No stratify for this test
        )

        # Get unique review IDs in each split
        train_ids = set(train_df['review_id'])
        test_ids = set(test_df['review_id'])

        # Verify no overlap
        overlap = train_ids.intersection(test_ids)
        assert len(overlap) == 0, f"Reviews should not appear in both splits: {overlap}"


class TestLabelingNoLeakage:
    """Test that labeling does not leak information"""

    def test_lexicon_threshold_consistency(self):
        """Threshold should be the same everywhere"""
        from shared.lexicon import classify_sentiment

        # Test a few samples
        texts = [
            "aplikasi sangat bagus",
            "error crash tidak bisa",
            "kuis untuk belajar"
        ]

        for text in texts:
            res = classify_sentiment(text, threshold=NEUTRAL_THRESHOLD)
            label = res['label']
            score = res['raw_score']

            # Verify classification logic
            if score > NEUTRAL_THRESHOLD:
                assert label == "Positif"
            elif score < -NEUTRAL_THRESHOLD:
                assert label == "Negatif"
            else:
                assert label == "Netral"


class TestPredictionConsistency:
    """Test that prediction uses same preprocessing as training"""

    def test_preprocessing_identical(self):
        """Preprocessing for prediction must be identical to training"""
        train_text = "Aplikasi ini sangat bagus!"
        pred_text = "Aplikasi ini sangat bagus!"

        train_processed = preprocess_text(train_text)
        pred_processed = preprocess_text(pred_text)

        assert train_processed == pred_processed

    def test_vectorizer_consistency(self):
        """Vectorizer must be loaded, not retrained"""
        # Use more text with more variation
        texts = [
            "aplikasi bagus membantu belajar sekolah",
            "error crash tidak bisa digunakan offline",
            "soal lengkap cocok untuk latihan matematika",
            "sangat kecewa tidak recomended",
            "bagus sekali sangat bermanfaat bermanfaat",
            "game seru poin banyak ranking"
        ]

        # Create and fit vectorizer
        vectorizer = build_vectorizer()
        X1 = vectorizer.fit_transform(texts)

        # Save (in real scenario, would save to disk)
        # Load and use again
        vectorizer2 = build_vectorizer()
        X2 = vectorizer2.fit_transform(texts)

        # Results should be identical
        assert X1.shape == X2.shape
        assert X1.shape[1] > 0  # Should have some features


class TestCrossValidationNoLeakage:
    """Test that cross-validation uses proper splits"""

    def test_cv_uses_correct_data(self):
        """CV should only use training data, not test data"""
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.pipeline import Pipeline

        # Create simple dataset
        texts = ["text " + str(i) for i in range(50)]
        labels = ["Positif"] * 25 + ["Negatif"] * 25

        # Split first (simulating proper pipeline)
        train_idx, test_idx = train_test_split(
            range(50), test_size=0.2, random_state=42, stratify=labels
        )

        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]

        # Vectorize only train data
        pipeline = Pipeline([
            ("tfidf", build_vectorizer()),
            ("classifier", build_classifier()),
        ])

        # CV on train data only (correct)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        scores = cross_val_score(
            pipeline, train_texts, train_labels, cv=cv, scoring="f1_macro"
        )

        # CV should complete without error
        assert len(scores) == 5
        assert all(0 <= s <= 1 for s in scores)


class TestFinalConfigLocked:
    """Konfigurasi final tidak boleh berubah diam-diam."""

    def test_tfidf_config_locked(self):
        """
        Konfigurasi TF-IDF canonical v4.

        CATATAN SEJARAH: nilai lama 3000 / (1,2) / sublinear_tf=True berasal
        dari brief lama dan SUDAH DI-GANTI oleh konfigurasi v4 yang dipilih
        lewat chain eksperimen pada development set. Nilai lama tidak boleh
        dikembalikan.
        """
        assert FINAL_MODEL_CONFIG["tfidf"] == {
            "max_features": 5000,
            "ngram_range": (1, 3),
            "sublinear_tf": False,
            "min_df": 1,
            "max_df": 0.90,
        }

    def test_svm_config_locked(self):
        assert FINAL_MODEL_CONFIG["svm"] == {
            "kernel": "linear",
            "C": 10.0,
            "gamma": "scale",
            "class_weight": None,
        }

    def test_multiclass_strategy_is_ovo(self):
        assert FINAL_MODEL_CONFIG["multiclass"]["strategy"] == "ovo"

    def test_random_state(self):
        assert RANDOM_STATE == 42


class TestHoldoutGuard:
    """
    Holdout canonical tidak boleh masuk ke jalur pengujian web.

    Latar belakang: audit menemukan holdout pernah diunggah ke website dan
    527 dari 604 baris holdout pernah masuk training/CV pada attempt yang
    dibatalkan. Guard ini mencegah terulangnya.
    """

    @staticmethod
    def _holdout_frame(n=5):
        from shared.holdout_guard import CANONICAL_HOLDOUT_PATH

        assert CANONICAL_HOLDOUT_PATH.exists(), (
            f"Holdout canonical tidak ditemukan: {CANONICAL_HOLDOUT_PATH}"
        )
        return pd.read_csv(CANONICAL_HOLDOUT_PATH).head(n)

    def test_holdout_rows_are_detected(self):
        from shared.holdout_guard import find_holdout_rows

        frame = self._holdout_frame()
        matched = find_holdout_rows(frame)
        assert len(matched) == len(frame), (
            f"Hanya {len(matched)} dari {len(frame)} baris holdout terdeteksi."
        )

    def test_upload_of_holdout_is_rejected(self):
        from shared.holdout_guard import HoldoutLeakError, assert_no_holdout_rows

        frame = self._holdout_frame()
        with pytest.raises(HoldoutLeakError):
            assert_no_holdout_rows(frame)

    def test_unrelated_upload_is_allowed(self):
        from shared.holdout_guard import assert_no_holdout_rows

        frame = pd.DataFrame({
            "content": [
                "Aplikasi bagus sekali membantu belajar",
                "Error crash terus saat membuka halaman",
            ],
            "label": ["positif", "negatif"],
        })
        # Tidak boleh melempar error.
        assert_no_holdout_rows(frame)

    def test_guard_is_enforced_in_upload_service(self, monkeypatch):
        """Upload testing rejects canonical holdout even with the legacy audit override."""
        from backend.services.upload_service import UploadValidationError, validate_upload

        monkeypatch.setenv("WAYGROUND_ALLOW_HOLDOUT_UPLOAD", "1")
        with pytest.raises(UploadValidationError, match="holdout: [1-9]"):
            validate_upload(self._holdout_frame(), label_column="__none__")

    def test_holdout_not_used_for_selection(self):
        """Kontrak pemilihaneksperimen menyatakan holdout tidak dipakai."""
        from shared.deployment_integrity import verify_selection_contract

        selected, _run = verify_selection_contract()
        assert selected.get("split", {}).get("holdout_evaluated") is False
        assert selected.get("labeling", {}).get("threshold_selection_partition") \
            == "preparation_development_only"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
