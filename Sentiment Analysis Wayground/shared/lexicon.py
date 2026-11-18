"""
Wayground - Lexicon-Based Sentiment Labeling
============================================

Sumber sentimen:
    1. data/lexicon/lexicon_positive.tsv            (versi 1, default)
    2. data/lexicon/lexicon_negative.tsv            (versi 1, default)
    3. data/lexicon/v2/lexicon_positive.tsv         (versi 2, opsional)
    4. data/lexicon/v2/lexicon_negative.tsv         (versi 2, opsional)

Aturan penting:
- Neutral lexicon NONAKTIF.
- InSet expansion NONAKTIF.
- Hard-coded OOV expansion NONAKTIF.
- OOV hanya diaudit, tidak otomatis dimasukkan ke lexicon.
- Jika sebuah kata muncul di kedua lexicon, proses berhenti dengan error.
- Mendukung token hasil preprocessing seperti "tidak_bisa".
- Entri frasa (mengandung spasi) TIDAK dapat dicocokkan oleh classifier
  berbasis token. Pada lexicon v2 entri frasa dipisah ke file phrases_*.

CATATAN PENTING (hasil audit Tahap 1):
Entri lexicon harus dalam bentuk yang sama dengan token hasil
preprocessing (stemmed). Entri seperti "membantu" tidak pernah cocok karena
Sastrawi mengubahnya menjadi "bantu". Lihat docs/audit_lexicon.md.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import csv
import re

from .constants import (
    POSITIVE_LEXICON_PATH,
    NEGATIVE_LEXICON_PATH,
    NEGATION_WORDS,
    POSITIVE_LEXICON_V2_PATH,
    NEGATIVE_LEXICON_V2_PATH,
    POSITIVE_LEXICON_V2_BRIEF_PATH,
    NEGATIVE_LEXICON_V2_BRIEF_PATH,
    POSITIVE_LEXICON_V2_NO_TOPIC_PATH,
    NEGATIVE_LEXICON_V2_NO_TOPIC_PATH,
)

# ---------------------------------------------------------------------------
# Loader TSV yang toleran terhadap BOM, header, dan baris kosong.
# Tidak menggunakan pandas read_csv agar satu baris rusak tidak menyebabkan
# "Expected 1 fields ... saw 2" tanpa pesan yang jelas.
# ---------------------------------------------------------------------------

def _read_lexicon_file(path: Path, polarity: str) -> Dict[str, float]:
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"Lexicon tidak ditemukan: {path}")

    result: Dict[str, float] = {}
    skipped = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            # Format utama: word<TAB>score.
            parts = line.split("\t")

            # Toleransi untuk file yang terlanjur memakai koma/semicolon.
            if len(parts) < 2:
                if "," in line:
                    parts = line.split(",")
                elif ";" in line:
                    parts = line.split(";")

            if len(parts) < 2:
                # Header satu kolom atau baris invalid.
                first = line.lower().strip()
                if first in {"word", "kata", "term", "token"}:
                    continue
                skipped.append((line_no, line, "kurang dari 2 kolom"))
                continue

            word = parts[0].strip().strip('"').strip("'").lower()
            score_text = parts[1].strip().strip('"').strip("'")

            if word.lower() in {"word", "kata", "term", "token"}:
                continue

            try:
                score = float(score_text.replace(",", "."))
            except ValueError:
                skipped.append((line_no, line, f"score tidak numerik: {score_text}"))
                continue

            if not word:
                skipped.append((line_no, line, "kata kosong"))
                continue

            # Pastikan polaritas konsisten.
            # Positive harus > 0, negative harus < 0.
            if polarity == "positive":
                score = abs(score)
                if score == 0:
                    skipped.append((line_no, line, "score positif = 0"))
                    continue
            elif polarity == "negative":
                score = -abs(score)
                if score == 0:
                    skipped.append((line_no, line, "score negatif = 0"))
                    continue

            # Jika duplikat, gunakan baris terakhir.
            result[word] = score

    if skipped:
        print(f"[LEXICON] Warning {path.name}: {len(skipped)} baris dilewati.")
        for item in skipped[:5]:
            print(f"  line {item[0]}: {item[2]}")

    if not result:
        raise ValueError(f"Lexicon kosong setelah parsing: {path}")

    return result


class TwoLexicon:
    """Lexicon positif + negatif saja, tanpa neutral/expansion."""

    def __init__(self, positive_path=None, negative_path=None, version="v1"):
        self.positive_lexicon: Dict[str, float] = {}
        self.negative_lexicon: Dict[str, float] = {}
        self.expanded_positive: Dict[str, float] = {}
        self.expanded_negative: Dict[str, float] = {}
        self.neutral_lexicon: Dict[str, float] = {}
        self.oov_words: Dict[str, int] = {}
        self._combined_cache: Dict[str, float] | None = None

        self.version = version
        self.positive_path = Path(
            positive_path if positive_path is not None
            else POSITIVE_LEXICON_PATH
        )
        self.negative_path = Path(
            negative_path if negative_path is not None
            else NEGATIVE_LEXICON_PATH
        )

        self._load_lexicons()

    def _load_lexicons(self):
        print(f"[LEXICON] Versi lexicon : {self.version}")
        print("[LEXICON] Neutral lexicon: DISABLED")
        print("[LEXICON] InSet expansion: DISABLED")
        print("[LEXICON] OOV auto-expansion: DISABLED")

        self.positive_lexicon = _read_lexicon_file(
            self.positive_path, "positive"
        )
        self.negative_lexicon = _read_lexicon_file(
            self.negative_path, "negative"
        )

        overlap = set(self.positive_lexicon) & set(self.negative_lexicon)
        if overlap:
            sample = sorted(overlap)[:20]
            raise ValueError(
                f"Ditemukan {len(overlap)} kata pada kedua lexicon. "
                f"Perbaiki konflik terlebih dahulu. Contoh: {sample}"
            )

        print(f"[LEXICON] Positive words: {len(self.positive_lexicon)}")
        print(f"[LEXICON] Negative words: {len(self.negative_lexicon)}")
        print(f"[LEXICON] Cross-polarity overlap: {len(overlap)}")

    def get_combined_lexicons(self) -> Dict[str, float]:
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

    def _get_word_score(self, word: str) -> float:
        return self.get_combined_lexicons().get(str(word).lower(), 0.0)

    def classify_token(self, token: str) -> float:
        token = str(token).strip().lower()
        if not token:
            return 0.0

        # Preprocessing project mengubah "tidak bagus" menjadi "tidak_bagus".
        # Negasi membalik polaritas base word bila base word memang ada.
        if self._is_negated_token(token):
            base_word = self._get_base_word(token)
            base_score = self._get_word_score(base_word)
            if base_score != 0:
                return -base_score

        return self._get_word_score(token)


# Backward-compatible class name. Script lain yang melakukan
# "from shared.lexicon import InSetLexicon" tetap dapat berjalan,
# tetapi implementasinya sudah TWO-LEXICON ONLY.
InSetLexicon = TwoLexicon

# Registry lexicon per versi. Dipakai oleh shared.labeling sebagai
# satu-satunya pintu masuk pelabelan.
#
# Registry ini SENGAJA kosong pada import: lexicon hanya dimuat saat
# benar-benar dipakai. Sebelumnya `LEXICON = TwoLexicon()` dibuat langsung
# saat import, sehingga setiap proses (backend, skrip, test) memuat lexicon
# v1 dan mencetak "[LEXICON] Versi lexicon : v1" walau jalur riset sudah
# memakai v2. Perilaku pelabelan tidak berubah: pemanggilan yang benar
# tetap memakai lexicon yang diminta, hanya waktu muat yang digeser.
_LEXICON_REGISTRY: Dict[str, TwoLexicon] = {}


def get_lexicon(version: str = "v1") -> TwoLexicon:
    """
    Mengembalikan objek lexicon untuk versi tertentu.

    Versi yang tersedia:
        "v1" -> data/lexicon/lexicon_positive.tsv + lexicon_negative.tsv
        "v2" -> data/lexicon/v2/... (rekomendasi: R3 + R6)
        "v2_brief_lengkap" -> data/lexicon/v2_brief_lengkap/...
        "v2_tanpa_hapus_topic" -> data/lexicon/v2_tanpa_hapus_topic/...
    """
    key = str(version).strip().lower()

    if key in _LEXICON_REGISTRY:
        return _LEXICON_REGISTRY[key]

    variants = {
        "v1": (
            POSITIVE_LEXICON_PATH,
            NEGATIVE_LEXICON_PATH,
        ),
        "v2": (
            POSITIVE_LEXICON_V2_PATH,
            NEGATIVE_LEXICON_V2_PATH,
        ),
        "v2_brief_lengkap": (
            POSITIVE_LEXICON_V2_BRIEF_PATH,
            NEGATIVE_LEXICON_V2_BRIEF_PATH,
        ),
        "v2_tanpa_hapus_topic": (
            POSITIVE_LEXICON_V2_NO_TOPIC_PATH,
            NEGATIVE_LEXICON_V2_NO_TOPIC_PATH,
        ),
    }

    if key in variants:
        positive_path, negative_path = variants[key]
        lexicon = TwoLexicon(
            positive_path=positive_path,
            negative_path=negative_path,
            version=key,
        )
        _LEXICON_REGISTRY[key] = lexicon
        return lexicon

    raise ValueError(
        f"Versi lexicon tidak dikenal: {version!r}. "
        f"Pilihan yang tersedia: {sorted(variants)}."
    )


def get_default_lexicon() -> TwoLexicon:
    """
    Lexicon yang dipakai ketika pemanggil tidak memilih versi secara
    eksplisit.

    Versi aktif ditentukan oleh shared.constants.resolve_lexicon_version()
    (default "v2", dapat dioverride lewat WAYGROUND_LEXICON_VERSION).

    CATATAN (Tahap 6): sebelum perbaikan ini, jalur default memakai lexicon
    v1 (InSet) melalui global `LEXICON`. Itu bertentangan dengan aturan
    canonical "lexicon v2, bukan InSet" dan membuat pemanggilan tanpa
    argumen lexicon (tests, scripts/evaluasi_per_aspek.py) memakai kamus
    yang berbeda dari jalur pelabelan canonical.
    """
    from .constants import resolve_lexicon_version

    return get_lexicon(resolve_lexicon_version())


def __getattr__(name: str):
    """
    Menyediakan `LEXICON` secara lazy untuk kompatibilitas lama.

    Nilai yang dikembalikan adalah lexicon AKTIF (default v2), bukan
    fallback v1. Jalur pelabelan canonical memakai
    shared.labeling.get_active_lexicon() yang resolusinya sama.
    """
    if name == "LEXICON":
        return get_default_lexicon()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load_lexicon_file(path: Path, polarity: str) -> Dict[str, float]:
    """
    Loader publik untuk file lexicon TSV.

    Mengembalikan dict {kata: bobot} dengan polaritas yang konsisten:
    positive -> > 0, negative -> < 0.

    Dipakai oleh shared.labeling, scripts/audit_oov.py, dan script lain
    yang butuh membaca lexicon tanpa membuat jalur loader baru.
    """
    return _read_lexicon_file(Path(path), polarity)


def calculate_sentiment_score(
    tokens: List[str],
    lexicon: TwoLexicon | None = None,
) -> float:
    active = lexicon if lexicon is not None else get_default_lexicon()
    return float(sum(active.classify_token(token) for token in tokens))


def classify_sentiment(
    text: str,
    threshold: float = 0.0,
    lexicon: TwoLexicon | None = None,
) -> dict:
    """
    Fungsi pelabelan sentimen berbasis lexicon.

    Ini adalah SATU-SATUNYA implementasi pelabelan otomatis pada project.
    Stage 3 (pipeline/run_pipeline.py) dan audit_oov (scripts/audit_oov.py)
    KEDUANYA memanggil fungsi ini, sehingga tidak ada jalur ganda.
    """
    active = lexicon if lexicon is not None else get_default_lexicon()

    if not text or not isinstance(text, str):
        return {
            "label": "Netral",
            "raw_score": 0.0,
            "normalized_score": 0.0,
            "threshold": threshold,
            "method": "two_lexicon_raw_threshold",
            "token_count": 0,
            "lexicon_version": active.version,
        }

    tokens = text.split()
    if not tokens:
        return {
            "label": "Netral",
            "raw_score": 0.0,
            "normalized_score": 0.0,
            "threshold": threshold,
            "method": "two_lexicon_raw_threshold",
            "token_count": 0,
            "lexicon_version": active.version,
        }

    total_score = calculate_sentiment_score(tokens, active)
    normalized = total_score / len(tokens)

    if total_score > threshold:
        label = "Positif"
    elif total_score < -threshold:
        label = "Negatif"
    else:
        label = "Netral"

    return {
        "label": label,
        "raw_score": total_score,
        "normalized_score": normalized,
        "threshold": threshold,
        "method": "two_lexicon_raw_threshold",
        "token_count": len(tokens),
        "lexicon_version": active.version,
    }


def classify_sentiment_tuple(
    text: str, threshold: float = 0.0
) -> Tuple[str, float, float]:
    result = classify_sentiment(text, threshold)
    return (
        result["label"],
        result["raw_score"],
        result["normalized_score"],
    )


def get_oov_tokens(tokens: List[str]) -> List[str]:
    """Mengembalikan OOV untuk audit saja. Tidak mengubah lexicon."""
    vocabulary = set(get_default_lexicon().get_combined_lexicons())
    result = []

    for token in tokens:
        token = str(token).strip().lower()
        if not token:
            continue

        check_word = token
        if "_" in token:
            prefix, base = token.split("_", 1)
            if prefix in NEGATION_WORDS:
                check_word = base

        if check_word not in vocabulary:
            result.append(check_word)

    return result


def get_label_distribution(
    texts,
    threshold: float = 0.0,
    lexicon: TwoLexicon | None = None,
) -> Dict[str, int]:
    """
    Menghitung distribusi label sentimen untuk kumpulan teks.

    Fungsi ini ditambahkan karena beberapa modul (test, backend, script
    eksperimen lama) mengimpornya dari shared.lexicon.
    """
    distribution = {"Positif": 0, "Negatif": 0, "Netral": 0}

    for text in texts:
        result = classify_sentiment(text, threshold, lexicon=lexicon)
        label = result["label"]
        distribution[label] = distribution.get(label, 0) + 1

    return distribution


def get_lexicon_statistics(
    lexicon: TwoLexicon | None = None,
) -> dict:
    if lexicon is None:
        # Tanpa argumen, tampilkan lexicon AKTIF (bukan fallback v1) supaya
        # ringkasan yang dicetak sesuai dengan jalur pelabelan canonical.
        lexicon = get_default_lexicon()
    active = lexicon
    positive = active.positive_lexicon
    negative = active.negative_lexicon
    return {
        "version": active.version,
        "positive_words": len(positive),
        "negative_words": len(negative),
        "neutral_words": 0,
        "expanded_positive": 0,
        "expanded_negative": 0,
        "oov_auto_expansion": False,
        "overlap": len(set(positive) & set(negative)),
    }


if __name__ == "__main__":
    print("=" * 80)
    print("VERIFIKASI TWO-LEXICON ONLY")
    print("=" * 80)

    stats = get_lexicon_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    examples = [
        "aplikasi sangat bagus dan membantu belajar",
        "saya tidak suka aplikasi ini karena error terus",
        "aplikasi ini untuk mengerjakan tugas sekolah",
        "tidak_bisa login sudah seminggu",
        "game ini seru banget dapat poin banyak",
    ]

    print()
    print("=" * 80)
    print("CONTOH KLASIFIKASI SENTIMEN")
    print("=" * 80)

    for text in examples:
        result = classify_sentiment(text)
        print(f"\nTeks: {text}")
        print(f"Skor (raw): {result['raw_score']:.4f}")
        print(f"Skor (normalized): {result['normalized_score']:.4f}")
        print(f"Label: {result['label']}")
        print(f"Tokens: {result['token_count']}")
