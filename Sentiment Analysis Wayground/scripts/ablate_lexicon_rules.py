"""
Ablasi Aturan Lexicon v2
========================

Mengisolasi kontribusi tiap aturan (R1 topic removal, R3 restoration,
R4 stem forms) terhadap kualitas pelabelan terhadap 288 label pakar.

Tujuan: mencari aturan yang justru menurunkan kualitas, agar lexicon v2
dapat diperbaiki berdasarkan data, bukan berdasarkan dugaan.

Script ini hanya membaca data; tidak menulis ke lexicon.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_lexicon_v2 import (  # noqa: E402
    RESTORED_SENTIMENT,
    TOPIC_BRIEF,
    TOPIC_EXTENSI,
    TOPIC_TOKENS,
)

from shared.constants import (  # noqa: E402
    NEGATION_WORDS,
    CANONICAL_DATASET_PATH,
    LEXICON_AUDIT_DIR,
)
from shared.labeling import label_text  # noqa: E402
from shared.lexicon import TwoLexicon  # noqa: E402  (dipakai sebagai referensi antarmuka)

CANONICAL = CANONICAL_DATASET_PATH
OUTPUT_JSON = (
    LEXICON_AUDIT_DIR / "lexicon_v2_ablation.json"
)

CLASSES = ["Positif", "Negatif", "Netral"]

# Cache hasil stemming supaya tidak dihitung berulang kali.
STEM_CACHE: dict[str, str] = {}


def read_lexicon(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 2:
                continue
            word = row[0].strip().lower()
            if not word or word == "word":
                continue
            try:
                result[word] = float(row[1])
            except ValueError:
                continue
    return result


def build_variant(
    base_pos: dict,
    base_neg: dict,
    remove_topic: bool,
    restore: bool,
    stem_forms: bool,
    stemmer=None,
    topic_set=None,
    min_stem_freq: int = 0,
    corpus_freq: dict | None = None,
    require_kamus: bool = False,
    kamus_words: set | None = None,
) -> tuple[dict, dict]:
    positive = dict(base_pos)
    negative = dict(base_neg)
    topic = topic_set if topic_set is not None else TOPIC_TOKENS

    if remove_topic:
        for word in topic:
            positive.pop(word, None)
            negative.pop(word, None)

    if restore:
        for word, weight, _reason, _category in RESTORED_SENTIMENT:
            if word not in positive and word not in negative:
                positive[word] = weight

    if stem_forms and stemmer is not None:
        current = {**positive, **negative}
        # Iterasi deterministik (alfabetis) atas entri v1, sama seperti
        # scripts/build_lexicon_v2.py, supaya hasil ablation dan lexicon
        # yang dibangun benar-benar identik.
        for word in sorted(current):
            if " " in word:
                continue
            stemmed = STEM_CACHE.get(word)
            if stemmed is None:
                stemmed = stemmer.stem(word)
                STEM_CACHE[word] = stemmed
            if stemmed == word:
                continue
            # Cek terhadap dict langsung, bukan snapshot.
            if stemmed in positive or stemmed in negative:
                continue
            if stemmed in topic:
                continue
            if min_stem_freq > 0:
                freq = (corpus_freq or {}).get(stemmed, 0)
                if freq < min_stem_freq:
                    continue
            if require_kamus and stemmed not in (kamus_words or set()):
                continue
            score = current[word]
            if stemmed in positive and score < 0:
                continue
            if stemmed in negative and score > 0:
                continue
            if score > 0:
                positive[stemmed] = score
            else:
                negative[stemmed] = score

    # Buang frasa (tidak bisa cocok dengan classifier token).
    positive = {k: v for k, v in positive.items() if " " not in k}
    negative = {k: v for k, v in negative.items() if " " not in k}

    return positive, negative


def evaluate(lexicon, texts, y_true) -> dict:
    pred = [label_text(t, lexicon=lexicon)["label"] for t in texts]
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro")),
        "prediction_distribution": {
            label: pred.count(label) for label in CLASSES
        },
    }


class VariantLexicon:
    """Objek lexicon in-memory dengan antarmuka yang sama dengan TwoLexicon."""

    def __init__(self, positive: dict, negative: dict):
        self.positive_lexicon = dict(positive)
        self.negative_lexicon = dict(negative)
        self.version = "ablation"
        self._combined_cache = None

    def get_combined_lexicons(self) -> dict:
        if self._combined_cache is None:
            combined = {}
            combined.update(self.positive_lexicon)
            combined.update(self.negative_lexicon)
            self._combined_cache = combined
        return self._combined_cache

    def _is_negated_token(self, token: str) -> bool:
        token = str(token).lower()
        return any(token.startswith(f"{neg}_") for neg in NEGATION_WORDS)

    def _get_base_word(self, token: str) -> str:
        token = str(token).lower()
        for neg in NEGATION_WORDS:
            prefix = f"{neg}_"
            if token.startswith(prefix):
                return token[len(prefix):]
        return token

    def classify_token(self, token: str) -> float:
        token = str(token).strip().lower()
        if not token:
            return 0.0

        if self._is_negated_token(token):
            base = self._get_base_word(token)
            base_score = self.get_combined_lexicons().get(base, 0.0)
            if base_score != 0:
                return -base_score

        return self.get_combined_lexicons().get(token, 0.0)


def main() -> int:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer = StemmerFactory().create_stemmer()

    base_pos = read_lexicon(PROJECT_ROOT / "data" / "lexicon" / "lexicon_positive.tsv")
    base_neg = read_lexicon(PROJECT_ROOT / "data" / "lexicon" / "lexicon_negative.tsv")

    df = pd.read_csv(CANONICAL, encoding="utf-8-sig")
    expert = df[df["sentiment_label_expert"].notna()].copy()

    texts = expert["processed_text"].fillna("").astype(str).tolist()
    y_true = expert["sentiment_label_expert"].astype(str).tolist()
    all_texts = df["processed_text"].fillna("").astype(str).tolist()

    # Frekuensi token pada corpus (processed_text) untuk membatasi R4.
    corpus_freq: dict[str, int] = {}
    for text in df["processed_text"].fillna("").astype(str):
        for token in text.split():
            corpus_freq[token] = corpus_freq.get(token, 0) + 1

    brief_only = set(TOPIC_BRIEF)

    # Kamus sentimen independen (kamus_positive/kamus_positif/kamus_negative/
    # kamus_negatif) dipakai sebagai dasar obyektif untuk deciding apakah
    # sebuah bentuk stem layak dimasukkan. Ini SUMBER EKSTERNAL, bukan
    # label pakar, sehingga tidak terjadi overfitting ke 288 baris.
    kamus_words: set[str] = set()
    for name in (
        "kamus_positive.tsv", "kamus_positif.tsv",
        "kamus_negative.tsv", "kamus_negatif.tsv",
    ):
        path = PROJECT_ROOT / "dataset" / name
        for row in read_lexicon(path):
            kamus_words.add(row)

    variants = {
        "v1 (baseline, apa adanya)": dict(
            remove_topic=False, restore=False, stem_forms=False,
        ),
        "v1 + R1 (hapus topic, daftar brief)": dict(
            remove_topic=True, restore=False, stem_forms=False,
            topic_set=brief_only,
        ),
        "v1 + R1 (hapus topic, +ekstensi)": dict(
            remove_topic=True, restore=False, stem_forms=False,
        ),
        "v1 + R3 (kembalikan sentimen)": dict(
            remove_topic=False, restore=True, stem_forms=False,
        ),
        "v1 + R4 (bentuk stem, semua)": dict(
            remove_topic=False, restore=False, stem_forms=True,
        ),
        "v1 + R4 (bentuk stem, freq>=5)": dict(
            remove_topic=False, restore=False, stem_forms=True,
            min_stem_freq=5, corpus_freq=corpus_freq,
        ),
        "v1 + R3 + R4(freq>=5)": dict(
            remove_topic=False, restore=True, stem_forms=True,
            min_stem_freq=5, corpus_freq=corpus_freq,
        ),
        "v1 + R1(brief) + R3 + R4(freq>=5)": dict(
            remove_topic=True, restore=True, stem_forms=True,
            topic_set=brief_only, min_stem_freq=5, corpus_freq=corpus_freq,
        ),
        "v1 + R1(brief+ekst) + R3 + R4(freq>=5)": dict(
            remove_topic=True, restore=True, stem_forms=True,
            min_stem_freq=5, corpus_freq=corpus_freq,
        ),
        "v1 + R4(kamus-grounded)": dict(
            remove_topic=False, restore=False, stem_forms=True,
            require_kamus=True, kamus_words=kamus_words,
        ),
        "v1 + R3 + R4(kamus-grounded)": dict(
            remove_topic=False, restore=True, stem_forms=True,
            require_kamus=True, kamus_words=kamus_words,
        ),
        "v1 + R1(brief) + R3 + R4(kamus-grounded)": dict(
            remove_topic=True, restore=True, stem_forms=True,
            topic_set=brief_only, require_kamus=True, kamus_words=kamus_words,
        ),
        "v1 + R1(brief+ekst) + R3 + R4(kamus-grounded)": dict(
            remove_topic=True, restore=True, stem_forms=True,
            require_kamus=True, kamus_words=kamus_words,
        ),
        "v2 (semua aturan, seperti dibangun)": dict(
            remove_topic=True, restore=True, stem_forms=True,
        ),
    }

    results = {}

    print("=" * 96)
    print("ABLASI ATURAN LEXICON (288 label pakar)")
    print("=" * 96)
    print(
        f"{'Varian':<34}{'Akurasi':>10}{'F1-macro':>12}"
        f"{'Pred Pos':>10}{'Pred Neg':>10}{'Pred Net':>10}"
    )
    print("-" * 96)

    for name, options in variants.items():
        options.setdefault("kamus_words", kamus_words)
        positive, negative = build_variant(
            base_pos, base_neg, stemmer=stemmer, **options
        )

        lexicon = VariantLexicon(positive, negative)

        metrics = evaluate(lexicon, texts, y_true)

        all_pred = [label_text(t, lexicon=lexicon)["label"] for t in all_texts]
        metrics["distribusi_seluruh_dataset"] = {
            label: all_pred.count(label) for label in CLASSES
        }
        metrics["positive_entries"] = len(positive)
        metrics["negative_entries"] = len(negative)

        results[name] = metrics

        dist = metrics["prediction_distribution"]
        print(
            f"{name:<34}{metrics['accuracy']:>10.4f}"
            f"{metrics['f1_macro']:>12.4f}"
            f"{dist['Positif']:>10}{dist['Negatif']:>10}{dist['Netral']:>10}"
        )

    print("-" * 96)
    print("Distribusi label pada 3.060 baris (semua varian)")
    print("-" * 96)
    for name, metrics in results.items():
        print(f"  {name:<34} {metrics['distribusi_seluruh_dataset']}")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nTersimpan: {OUTPUT_JSON}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
