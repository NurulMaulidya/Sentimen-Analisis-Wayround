"""
Preprocessing Service
====================
Handles the 7-step preprocessing pipeline with step-by-step tracking.
Each step result is stored separately for transparency.

This module is a THIN WRAPPER around the canonical implementation in
``shared/preprocessing.py``. It does NOT re-implement any preprocessing
stage: duplicating the logic previously allowed the website and the
training pipeline to drift apart.
"""

import json
from typing import List, Dict, Any

try:
    from shared.preprocessing import (
        preprocess_text as _canonical_preprocess,
        clean_text as _canonical_clean,
        case_fold as _canonical_case_fold,
        tokenize as _canonical_tokenize,
        normalize_slang as _canonical_normalize,
        handle_negation as _canonical_negation,
        remove_stopwords as _canonical_stopwords,
        stem_tokens as _canonical_stem,
    )
except ImportError:  # pragma: no cover - fallback for direct script execution
    from shared.preprocessing import (
        preprocess_text as _canonical_preprocess,
        clean_text as _canonical_clean,
        case_fold as _canonical_case_fold,
        tokenize as _canonical_tokenize,
        normalize_slang as _canonical_normalize,
        handle_negation as _canonical_negation,
        remove_stopwords as _canonical_stopwords,
        stem_tokens as _canonical_stem,
    )


# =============================================================================
# PREPROCESSING STEPS
# =============================================================================

def step1_cleaning(text: str) -> str:
    """Step 1: Cleaning - hapus URL, emoji, angka, tanda baca."""
    return _canonical_clean(text)


def step2_case_folding(text: str) -> str:
    """Step 2: Case Folding - ubah huruf menjadi kecil."""
    return _canonical_case_fold(text)


def step3_tokenizing(text: str) -> List[str]:
    """Step 3: Tokenizing - pecah teks menjadi token kata."""
    return _canonical_tokenize(text)


def step4_normalization(tokens: List[str]) -> List[str]:
    """Step 4: Normalization - slang token-level + phrase normalization."""
    return _canonical_normalize(tokens)


def step5_negation_handling(tokens: List[str]) -> List[str]:
    """Step 5: Negation Handling - gabung negasi dengan token berikutnya."""
    return _canonical_negation(tokens)


def step6_stopword_removal(tokens: List[str]) -> List[str]:
    """Step 6: Stopword Removal - hapus stopword, pertahankan negasi."""
    return _canonical_stopwords(tokens)


def step7_stemming(tokens: List[str]) -> List[str]:
    """Step 7: Stemming - stemming Sastrawi, termasuk pada token negasi."""
    return _canonical_stem(tokens)


def step8_final(tokens: List[str]) -> str:
    """Step 8: Final - gabungkan token menjadi teks."""
    return " ".join(tokens)


# =============================================================================
# FULL PREPROCESSING PIPELINE
# =============================================================================

def preprocess_full(text: str) -> Dict[str, Any]:
    """
    Run the canonical 7-step preprocessing pipeline and return all steps.

    Delegates to ``shared.preprocessing.preprocess_text`` so the website can
    never drift from the preprocessing used during training.
    """
    original = str(text) if text else ""
    final, steps = _canonical_preprocess(original, return_steps=True)
    return {
        'original': original,
        'cleaned': steps["1_cleaning"],
        'case_folded': steps["2_case_folding"],
        'tokenized': steps["3_tokenizing"],
        'normalized': steps["4_normalization"],
        'negation': steps["5_negation"],
        'stopwords': steps["6_stopwords"],
        'stemmed': steps["7_stemming"],
        'final': final,
    }


def preprocess_batch(texts: List[str]) -> List[Dict[str, Any]]:
    """Process multiple texts."""
    return [preprocess_full(text) for text in texts]


# =============================================================================
# STAGE MAPPING FOR DATABASE STORAGE
# =============================================================================

# Mapping from stage name to field name in ReviewData model
STAGE_TO_FIELD = {
    'original': 'original_content',
    'cleaning': 'cleaned_text',
    'case_folding': 'case_folded_text',
    'tokenizing': 'tokenized_tokens',
    'normalization': 'normalized_tokens',
    'negation': 'negation_tokens',
    'stopwords': 'stopwords_tokens',
    'stemming': 'stemmed_tokens',
    'final': 'final_processed_text'
}

# Display names for stages
STAGE_DISPLAY_NAMES = {
    'original': 'Original',
    'cleaning': 'Cleaning',
    'case_folding': 'Case Folding',
    'tokenizing': 'Tokenizing',
    'normalization': 'Normalization',
    'negation': 'Negation',
    'stopwords': 'Stopword Removal',
    'stemming': 'Stemming',
    'final': 'Final'
}

# Stage order for sequential processing
STAGE_ORDER = [
    'original', 'cleaning', 'case_folding', 'tokenizing',
    'normalization', 'negation', 'stopwords', 'stemming', 'final'
]
