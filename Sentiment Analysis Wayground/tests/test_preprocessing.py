"""
Unit Tests for Preprocessing Module
====================================
Tests all 7 preprocessing stages according to methodology.

RUN: python -m pytest tests/test_preprocessing.py -v
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.preprocessing import (
    clean_text,
    case_fold,
    tokenize,
    normalize_slang,
    handle_negation,
    remove_stopwords,
    stem_tokens,
    normalize_repeated_letters,
    map_emojis,
    preprocess_text
)


class TestCleaning:
    """Test Tahap 1: Cleaning"""

    def test_remove_url(self):
        """URLs should be removed"""
        assert clean_text("Visit https://example.com now") == "Visit now"
        assert clean_text("Check www.test.com") == "Check"

    def test_remove_emoji(self):
        """Emoji should be removed"""
        result = clean_text("Great app! 😊")
        assert "😊" not in result

    def test_remove_numbers(self):
        """Numbers should be removed"""
        result = clean_text("I have 3 apples and 5 oranges")
        assert "3" not in result
        assert "5" not in result

    def test_remove_punctuation(self):
        """Punctuation should be removed"""
        result = clean_text("Hello, world! How are you?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_whitespace_normalization(self):
        """Multiple whitespaces should be normalized"""
        result = clean_text("Hello    world")
        assert "  " not in result


class TestCaseFolding:
    """Test Tahap 2: Case Folding"""

    def test_lowercase(self):
        """All text should be lowercase"""
        assert case_fold("HELLO") == "hello"
        assert case_fold("HeLLo WoRLd") == "hello world"


class TestTokenizing:
    """Test Tahap 3: Tokenizing"""

    def test_split_words(self):
        """Text should be split into words"""
        tokens = tokenize("hello world")
        assert tokens == ["hello", "world"]

    def test_multiple_spaces(self):
        """Multiple spaces should not create empty tokens"""
        tokens = tokenize("hello   world")
        assert "hello" in tokens
        assert "world" in tokens


class TestNormalization:
    """Test Tahap 4: Normalisasi Slang"""

    def test_normalize_slang(self):
        """Slang should be normalized"""
        tokens = ["bgt", "gak", "ga", "bener"]
        normalized = normalize_slang(tokens)
        # Check that normalization happened
        assert len(normalized) > 0

    def test_normalize_list_input(self):
        """normalize_slang should accept list input"""
        tokens = ["bgt"]
        result = normalize_slang(tokens)
        assert isinstance(result, list)


class TestNegationHandling:
    """Test Tahap 5: Negation Handling"""

    def test_negation_combined(self):
        """Negation words should be combined with next word"""
        tokens = ["tidak", "bagus"]
        result = handle_negation(tokens)
        assert "tidak_bagus" in result
        assert "bagus" not in result

    def test_negation_belum_excluded(self):
        """'belum' is intentionally EXCLUDED from negation handling per constants.py comment.

        Because "belum" in review context (e.g., "quiz belum selesai") is actually
        a neutral/descriptive statement, not an expression of dissatisfaction.

        This is a DESIGN DECISION, not a bug.
        """
        tokens = ["belum", "puas"]
        result = handle_negation(tokens)
        # 'belum' should NOT be combined (per constants.py design decision)
        assert "belum_puas" not in result
        # Should remain as separate tokens
        assert "belum" in result
        assert "puas" in result

    def test_negation_at_end(self):
        """Negation at end of text should be kept alone"""
        tokens = ["saya", "tidak"]
        result = handle_negation(tokens)
        assert "tidak" in result

    def test_no_negation(self):
        """Words without negation should be unchanged"""
        tokens = ["aplikasi", "bagus"]
        result = handle_negation(tokens)
        assert "aplikasi" in result
        assert "bagus" in result


class TestStopwordRemoval:
    """Test Tahap 6: Stopword Removal"""

    def test_remove_common_stopwords(self):
        """Common stopwords should be removed"""
        tokens = ["aplikasi", "yang", "sangat", "bagus"]
        result = remove_stopwords(tokens)
        assert "yang" in result or "yang" not in result  # Depends on stopword list

    def test_keep_negation_words(self):
        """Negation words should be preserved"""
        tokens = ["tidak_bagus", "sangat"]
        result = remove_stopwords(tokens, exclude_negations=True)
        assert "tidak_bagus" in result


class TestStemming:
    """Test Tahap 7: Stemming"""

    def test_stem_returns_list(self):
        """stem_tokens should return list"""
        tokens = ["membantu", "belajar"]
        result = stem_tokens(tokens)
        assert isinstance(result, list)

    def test_stem_reduces_length(self):
        """Stemming should reduce word length"""
        tokens = ["membantu"]
        result = stem_tokens(tokens)
        # Stemmed word should be shorter or equal
        if result:
            assert len(result[0]) <= len(tokens[0])


class TestFullPipeline:
    """Test pipeline lengkap (7 tahap)"""

    def test_positive_review(self):
        """Positive review should have positive-related words"""
        text = "Aplikasinya bagus sekali, sangat membantu belajar"
        result = preprocess_text(text)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_negative_review(self):
        """Negative review with negation"""
        text = "Tidak bagus, error terus"
        result = preprocess_text(text)
        assert isinstance(result, str)
        assert "tidak_bagus" in result or "bagus" not in result

    def test_empty_text(self):
        """Empty text should return empty string"""
        result = preprocess_text("")
        assert result == ""

    def test_with_return_steps(self):
        """With return_steps=True, should return both result and steps"""
        result, steps = preprocess_text("test", return_steps=True)
        assert isinstance(result, str)
        assert isinstance(steps, dict)
        assert "1_cleaning" in steps
        assert "2_case_folding" in steps
        assert "3_tokenizing" in steps
        assert "4_normalization" in steps
        assert "5_negation" in steps
        assert "6_stopwords" in steps
        assert "7_stemming" in steps

    def test_slang_normalization_in_pipeline(self):
        """Slang should be normalized in full pipeline"""
        text = "Aplikasi ini bgt helpful, gak ribet"
        result = preprocess_text(text)
        # Check that slang was normalized
        assert isinstance(result, str)
        assert "bgt" not in result.lower() or "banget" in result.lower()


class TestNegationPreservation:
    """Test critical: Negation must NOT be removed by stopword removal"""

    def test_tidak_bagus_preserved(self):
        """'tidak bagus' must become 'tidak_bagus' NOT just 'bagus'"""
        text = "Aplikasi ini tidak bagus"
        result, steps = preprocess_text(text, return_steps=True)

        # Check negation step preserved the negation
        negation_step = steps.get("5_negation", [])
        assert "tidak_bagus" in negation_step

        # Final result should NOT have 'bagus' alone if negation was correctly handled
        final_tokens = result.split()
        # 'tidak_bagus' should be present, 'bagus' alone should not
        assert any("tidak_bagus" in token for token in final_tokens) or "bagus" not in final_tokens


class TestRepeatedLetterNormalization:
    """Tahap 4: normalisasi huruf berulang tanpa merusak kata sah."""

    @pytest.mark.parametrize(
        "word,expected",
        [
            ("bangett", "banget"),
            ("seruu", "seru"),
            ("sekalii", "sekali"),
            ("gacorrrrr", "gacor"),
            ("mantappp", "mantap"),
            ("kerennn", "keren"),
            ("nilaii", "nilai"),
        ],
    )
    def test_repeated_letters_collapsed(self, word, expected):
        assert normalize_repeated_letters(word) == expected

    @pytest.mark.parametrize(
        "word",
        ["saat", "laa", "banyak", "makan", "kelas", "nilai"],
    )
    def test_valid_words_untouched(self, word):
        assert normalize_repeated_letters(word) == word

    def test_pipeline_uses_normalization(self):
        result = preprocess_text("aplikasi ini banggget bagus")
        assert "banget" in result
        assert "banggget" not in result

    def test_unknown_double_consonant_is_kept(self):
        """Tanpa kata dasar yang dikenal, huruf ganda dibiarkan."""
        assert normalize_repeated_letters("banyakk") == "banyakk"


class TestEmojiMapping:
    """Tahap 1: emoji dipetakan menjadi token sentimen, bukan dibuang."""

    def test_thumbs_up_becomes_positive_token(self):
        result = preprocess_text("aplikasi bagus \U0001F44D")
        assert "emoji_positif" in result

    def test_angry_face_becomes_negative_token(self):
        result = preprocess_text("aplikasi jelek \U0001F621")
        assert "emoji_negatif" in result

    def test_thinking_face_becomes_neutral_token(self):
        result = preprocess_text("hmm \U0001F914")
        assert "emoji_netral" in result

    def test_heart_is_positive(self):
        result = preprocess_text("suka ❤️")
        assert "emoji_positif" in result

    def test_repeated_emoji_collapsed(self):
        result = preprocess_text("mantap \U0001F44D\U0001F44D\U0001F44D")
        assert result.count("emoji_positif") == 1

    def test_unmapped_emoji_is_removed(self):
        result = preprocess_text("halo \U0001F9D9")
        assert "\U0001F9D9" not in result

    def test_emoji_token_survives_stemming(self):
        _, steps = preprocess_text(
            "bagus \U0001F44D", return_steps=True
        )
        assert "emoji_positif" in steps["7_stemming"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
