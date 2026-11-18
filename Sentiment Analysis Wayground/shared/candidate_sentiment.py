"""Candidate rules for research runs; the v4 inference pipeline is separate."""

import re

from pipeline.aspect_detection import detect_aspect, split_sentences
from shared.constants import ASPECTS
from shared.lexicon import get_lexicon
from shared import preprocessing as prep


_PRESERVE = re.compile(r"\b(menyenangkan\s+tidak|kirim\s+berulang)\b")
_TOPIC_WORDS = {"aplikasi", "anak", "soal", "guru", "sekolah", "siswa", "nilai", "game"}


def preprocess_candidate(text, original=None):
    cleaned = prep.case_fold(prep.clean_text(text))
    revised = re.sub(r"\btdk\b", "tidak", cleaned)
    revised = re.sub(r"\btidak\s+ada\s+masalah\b", "tidak masalah", revised)
    if original is not None and revised == cleaned and not _PRESERVE.search(cleaned):
        return original
    tokens = []
    for index, chunk in enumerate(_PRESERVE.split(revised)):
        words = prep.tokenize(chunk)
        # These phrases have conflicting entries in the v4 normalization JSON.
        tokens.extend(words if index % 2 else prep.normalize_slang(words))
    return " ".join(prep.stem_tokens(prep.remove_stopwords(prep.handle_negation(tokens))))


def candidate_weak_label(processed):
    lexicon = get_lexicon("v2")
    score = sum(lexicon.classify_token(token) for token in processed.split()
                if token not in _TOPIC_WORDS)
    if score > 1.0:
        return "positif"
    if score < -1.0:
        return "negatif"
    return "netral"


def aspect_clauses(text):
    """Try independent contrast clauses, preserving aspect detection on other text."""
    aspects = {aspect: [] for aspect in ASPECTS}
    for sentence in split_sentences(text):
        for clause in re.split(r"\s*,?\s*\b(?:tetapi|tapi|namun)\b\s*", sentence, flags=re.IGNORECASE):
            clause = clause.strip()
            # The raw keyword detector uses word boundaries.
            keywords_text = re.sub(r"\b(soal|kuis|game)nya\b", r"\1", clause, flags=re.IGNORECASE)
            for aspect in detect_aspect(keywords_text):
                aspects[aspect].append(clause)
    return aspects
