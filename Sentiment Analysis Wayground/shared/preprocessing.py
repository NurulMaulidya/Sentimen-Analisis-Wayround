"""
Preprocessing teks final (canonical).

Urutan:
1. Cleaning
2. Case folding
3. Tokenizing
4. Normalisasi slang + phrase
5. Negation handling
6. Stopword removal
7. Stemming

Modul ini adalah SATU-SATUNYA implementasi preprocessing pada project.
Training, prediksi satu ulasan, dan prediksi file WAJIB memakai modul ini
agar teks yang masuk ke model identik dengan teks yang dipakai saat training.

Catatan:
Token negasi seperti "tidak_bagus" dipertahankan sampai akhir agar
polaritas sentimen tetap dapat dipertahankan.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List

import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory,
)

# Root project:
# Sentiment Analysis Wayground/
# ├── shared/
# │   ├── preprocessing.py
# │   └── constants.py
# └── data/

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Memastikan "python shared/preprocessing.py" tetap dapat mengimpor package
# shared dari root project.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.constants import (
    NEGATION_WORDS as CONFIG_NEGATION_WORDS,
    ADDITIONAL_STOPWORDS,
    SENTIMENT_PROTECTED_WORDS,
    SLANG_DICT_PATH,
    NORMALIZATION_DICT_PATH,
    NEGATIVE_LEXICON_PATH,
    POSITIVE_LEXICON_PATH,
)

# ---------------------------------------------------------------------------
# INITIALIZATION
# ---------------------------------------------------------------------------

STEMMER = StemmerFactory().create_stemmer()
_STOPWORDS = set(StopWordRemoverFactory().get_stop_words())
ALL_STOPWORDS = _STOPWORDS | set(ADDITIONAL_STOPWORDS)

NEGATION_WORDS = set(CONFIG_NEGATION_WORDS)
PROTECTED = set(SENTIMENT_PROTECTED_WORDS) | NEGATION_WORDS

# ---------------------------------------------------------------------------
# DICTIONARY LOADING
# ---------------------------------------------------------------------------

def _load_json(path: Path, label: str) -> dict:
    """Load kamus JSON. Kegagalan harus terlihat, bukan menjadi kamus kosong."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Kamus {label} tidak ditemukan: {path}\n"
            "Kamus ini wajib ada karena tahap normalisasi aktif."
        )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Format kamus {label} tidak valid: {path}\n"
            "Format yang diharapkan adalah object/dictionary JSON."
        )

    return {
        str(k).lower(): v
        for k, v in data.items()
        if not str(k).startswith("_")
    }

def load_slang_dict() -> dict:
    p = Path(SLANG_DICT_PATH)

    fixed = p.with_name("slang_dict_fixed.json")

    return _load_json(
        fixed if fixed.exists() else p,
        "slang",
    )

def load_normalization_dict() -> dict:
    p = Path(NORMALIZATION_DICT_PATH)

    fixed = p.with_name("normalization_dict_fixed.json")

    return _load_json(
        fixed if fixed.exists() else p,
        "normalisasi phrase",
    )

SLANG_DICT = load_slang_dict()
NORMALIZATION_DICT = load_normalization_dict()

# Panjang maksimum phrase berdasarkan jumlah token dalam key.
MAX_PHRASE_TOKENS = max(
    (
        key.count("_") + 1
        for key in NORMALIZATION_DICT
        if "_" in key
    ),
    default=1,
)

# ---------------------------------------------------------------------------
# NORMALISASI HURUF BERULANG (Tahap 4)
# ---------------------------------------------------------------------------
# Ulasan Play Store sering memakai huruf berulang untuk menekankan:
#   "bangett" -> "banget", "seruu" -> "seru", "sekalii" -> "sekali",
#   "gacorrrrr" -> "gacor".
#
# Aturan:
#   * Deretan huruf yang sama >= 3 ALWAYS dipipihkan menjadi 1 huruf.
#   * Deretan vokal ganda (2 huruf) dipipihkan menjadi 1, KECUALI kata
#     yang sah berhuruf ganda (lihat KEEP_DOUBLE_LETTER_WORDS), misalnya
#     "saat" dan "laa".
#   * Deretan konsonan ganda (2 huruf) TIDAK disentuh ("banyak" aman).
#
# Kata yang sah berhuruf vokal ganda:
KEEP_DOUBLE_LETTER_WORDS = {
    "saat", "saat2", "saats", "laat", "laats", "taat", "faat", "haa",
    "laa", "saa", "kaa", "maaf", "taaa", "saaa", "laaa", "haaa",
    }

# Known emphatic spelling present in the reference vocabulary, not a valid
# double-letter word. Keep exceptions narrow to protect app/error/allah.
REPETITION_CORRECTIONS = {"bangett": "banget"}

# Deretan >= 3 huruf yang sama selalu dipipihkan.
REPEATED_LETTER_MIN_RUN = 3

# ---------------------------------------------------------------------------
# PEMETAAN EMOJI (Tahap 1)
# ---------------------------------------------------------------------------
# Keputusan: emoji TIDAK dibuang, tetapi dipetakan menjadi token sentimen
# agar informasinya tidak hilang.
#
#   emoji positif -> "emoji_positif"
#   emoji negatif -> "emoji_negatif"
#   emoji netral  -> "emoji_netral"
#
# Token ini bertahan sampai tahap stemming dan menjadi fitur TF-IDF,
# sehingga model bisa belajar dari emoji. Emoji TIDAK memengaruhi label
# lexicon (lexicon v2 tidak memuat token ini) - lihat docs/audit_lexicon.md.
#
# Alasannya: menghapus emoji (perilaku lama) membuang informasi sentimen
# yang eksplisit, sedangkan menambahkannya ke lexicon akan mengotori
# pelabelan otomatis dengan aturan baru di tahap yang sudah disetujui.
EMOJI_SENTIMENT_MAP = {
    # Positif
    "\U0001F44D": "emoji_positif",   # thumbs up
    "\U0001F44C": "emoji_positif",   # ok hand
    "\U0001F44F": "emoji_positif",   # clap
    "\U0001F450": "emoji_positif",   # open hands
    "\U0001F44A": "emoji_positif",   # fist
    "\U0001F4AF": "emoji_positif",   # hundred points
    "\U0001F64C": "emoji_positif",   # raising hands
    "\U0001F44B": "emoji_positif",   # wave
    "\U0001F44A\U0001F3FB": "emoji_positif",
    "\U0001F44D\U0001F3FB": "emoji_positif",
    "\U0001F44D\U0001F3FF": "emoji_positif",
    "\U0001F44E\U0001F3FB": "emoji_negatif",
    "\U0001F44F\U0001F3FB": "emoji_positif",
    "\U0001F60A": "emoji_positif",   # smiling face with smiling eyes
    "\U0001F603": "emoji_positif",   # grinning face
    "\U0001F604": "emoji_positif",   # grinning face with big eyes
    "\U0001F601": "emoji_positif",   # beaming face
    "\U0001F606": "emoji_positif",   # grinning squinting face
    "\U0001F60D": "emoji_positif",   # smiling face with heart eyes
    "\U0001F970": "emoji_positif",   # smiling face with hearts
    "\U0001F929": "emoji_positif",   # star struck
    "\U0001F618": "emoji_positif",   # face blowing a kiss
    "\U0001F617": "emoji_positif",   # kissing face
    "\U0001F31F": "emoji_positif",   # glowing star
    "\U0001F525": "emoji_positif",   # fire
    "\U0001F389": "emoji_positif",   # party popper
    "\U0001F3C6": "emoji_positif",   # trophy
    # Negatif
    "\U0001F621": "emoji_negatif",   # enraged face
    "\U0001F620": "emoji_negatif",   # angry face
    "\U0001F92C": "emoji_negatif",   # cursing face
    "\U0001F44E": "emoji_negatif",   # thumbs down
    "\U0001F44E\U0001F3FD": "emoji_negatif",
    "\U0001F44E\U0001F3FE": "emoji_negatif",
    "\U0001F44E\U0001F3FF": "emoji_negatif",
    "\U0001F4A9": "emoji_negatif",   # pile of poo
    "\U0001F480": "emoji_negatif",   # skull
    "\U0001F6D2": "emoji_negatif",   # bomb
    "\U0001F5D1\U0001F4A3": "emoji_negatif",
    "\U0001F62D": "emoji_negatif",   # loudly crying face
    "\U0001F629": "emoji_negatif",   # weary face
    "\U0001F62B": "emoji_negatif",   # tired face
    "\U0001F440": "emoji_negatif",   # eyes
    "\U0001F648": "emoji_negatif",   # see-no-evil monkey
    # Netral
    "\U0001F914": "emoji_netral",    # thinking face
    "\U0001F937": "emoji_netral",    # shrug
    "\U0001F928": "emoji_netral",    # raised eyebrow
    "\U0001F615": "emoji_netral",    # confused face
    "\U0001F610": "emoji_netral",    # neutral face
    "\U0001F644": "emoji_netral",    # rolling eyes
    "\U0001F4A1": "emoji_netral",    # light bulb
}

# Emoji yang sering dipakai sebagai Heart (U+2764 + variation selector).
EMOJI_HEART = "❤"
EMOJI_HEART_VS = "️"


def map_emojis(text: str) -> str:
    """
    Ganti emoji umum menjadi token sentimen.

    Emoji yang tidak ada di peta TIDAK dihapus di tahap ini; emoji sisanya
    dibuang oleh tahap cleaning (regex karakter Unicode supplementary).
    """
    if text is None:
        return ""

    out = str(text)

    for emoji in sorted(EMOJI_SENTIMENT_MAP, key=len, reverse=True):
        token = EMOJI_SENTIMENT_MAP[emoji]
        if emoji in out:
            out = out.replace(emoji, f" {token} ")

    # Hati: bisa polos atau diikuti variation selector, dan bisa berpasangan
    # dengan emoji lain (mis. "❤️").
    out = re.sub(
        f"{EMOJI_HEART}{EMOJI_HEART_VS}?",
        " emoji_positif ",
        out,
    )

    # Emoji berulang yang identik diringkas menjadi satu token.
    out = re.sub(
        r"\b(emoji_(?:positif|negatif|netral))\b(?:\s+\1\b)+",
        r"\1",
        out,
    )

    return out


# ---------------------------------------------------------------------------
# PREPROCESSING STAGES
# ---------------------------------------------------------------------------

def normalize_repeated_letters(text: str) -> str:
    """
    Pipihkan huruf berulang tanpa merusak kata yang sah.

    Contoh:
        "bangett"   -> "banget"
        "seruu"     -> "seru"
        "sekalii"   -> "sekali"
        "gacorrrrr" -> "gacor"
        "saat"      -> "saat"   (tidak berubah)
        "laa"       -> "laa"    (tidak berubah)
        "banyak"    -> "banyak" (tidak berubah)
    """
    if text is None:
        return ""

    words = []
    for word in str(text).split():
        words.append(_normalize_word_repetition(word))

    return " ".join(words)


def _load_vocabulary() -> set:
    """
    Kosakata acuan untuk memutuskan apakah huruf ganda boleh dipipihkan.

    Sumber (semuanya independen dari label experts):
      - data/slang_dict.json
      - data/normalization_dict.json
      - data/lexicon/lexicon_positive.tsv + lexicon_negative.tsv
    """
    words = set(SLANG_DICT) | set(NORMALIZATION_DICT)

    for path in (POSITIVE_LEXICON_PATH, NEGATIVE_LEXICON_PATH):
        path = Path(path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            next(handle, None)
            for line in handle:
                token = line.split("\t")[0].strip().lower()
                if token:
                    words.add(token)

    return words


VOCABULARY = _load_vocabulary()


def _normalize_word_repetition(word: str) -> str:
    """
    Pipihkan huruf berulang pada satu kata.

    Aturan:
      1. Deretan >= 3 huruf sama -> 1 huruf (selalu aman).
      2. Huruf kembar (2) -> 1 huruf HANYA bila:
           a. bentuk tunggalnya ada di kosakata acuan, DAN
           b. kata ASLI-nya sendiri TIDAK ada di kosakata acuan.

         Syarat (b) mencegah kerusakan kata sah. Contoh nyata:
             "app"    -> "ap"     (jika hanya syarat (a))
             "error"  -> "eror"
             "allah"  -> "alah"
         Kosakata acuan sengaja memuat banyak ejaan salah ("banget",
         "ngak", "eror"), sehingga "bentuk tunggal ada di kamus" saja
         tidak cukup untuk membuktikan kata itu salah ketik.

         Kata sah tetap aman: "banyak" (bukan "bany"), "good" (bukan
         "god"), "bermanfaat" (bukan "bermanfat"), "saat", "laa".
    """
    if len(word) < 3:
        return word

    if word.lower() in KEEP_DOUBLE_LETTER_WORDS:
        return word

    # Explicit spelling corrections override vocabulary membership: reference
    # dictionaries also contain misspellings, which must not become protected.
    mapped = REPETITION_CORRECTIONS.get(word.lower())
    if isinstance(mapped, str) and mapped.lower() != word.lower():
        reduced = re.sub(r"([a-zA-Z])\1+", r"\1", word.lower())
        if reduced == mapped.lower():
            return mapped.upper() if word.isupper() else mapped

    # Deretan >= 3 huruf yang sama -> 1 huruf. Selalu aman: tidak ada kata
    # Indonesia/Inggris yang sah memakai 3 huruf identik beruntun.
    collapsed = re.sub(
        r"([a-zA-Z])\1{%d,}" % (REPEATED_LETTER_MIN_RUN - 1),
        r"\1",
        word,
    )

    if collapsed != word:
        return collapsed

    # Huruf kembar (vokal maupun konsonan) -> 1 huruf.
    lower = word.lower()

    # Syarat (b): kata asli sudah dikenal -> biarkan utuh.
    if lower in VOCABULARY:
        return word

    collapsed = re.sub(
        r"([a-zA-Z])\1",
        r"\1",
        word,
    )

    # Syarat (a): bentuk tunggal harus dikenal.
    if collapsed != word and collapsed.lower() in VOCABULARY:
        return collapsed

    return word


def clean_text(text: str) -> str:
    """
    Tahap 1 - Cleaning.
    Menghapus:
    - URL
    - emoji yang tidak dipetakan menjadi token sentimen
    - angka
    - tanda baca/simbol
    - whitespace berlebih

    Menambahkan:
    - pemetaan emoji menjadi token sentimen (lihat EMOJI_SENTIMENT_MAP)
    - pemipihan huruf berulang (lihat normalize_repeated_letters)
    """
    if text is None:
        return ""

    text = map_emojis(text)
    text = str(text)

    # URL
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Emoji / karakter Unicode supplementary
    text = re.sub(
        r"[\U00010000-\U0010FFFF]",
        " ",
        text,
    )

    # Angka
    text = re.sub(r"\d+", " ", text)

    # Tanda baca dan simbol.
    # \w dipertahankan agar huruf, angka, dan underscore tetap tersedia.
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE,
    )

    text = re.sub(r"\s+", " ", text).strip()

    # Pemipihan huruf berulang (Tahap 4).
    return normalize_repeated_letters(text)

def case_fold(text: str) -> str:
    """Tahap 2 - Case folding."""
    return str(text).lower()

def tokenize(text: str) -> List[str]:
    """Tahap 3 - Tokenizing."""
    return str(text).split()

def _expand_normalized_value(value) -> List[str]:
    """Ubah nilai kamus menjadi daftar token."""
    if isinstance(value, list):
        return [str(x).lower() for x in value]

    value = str(value).lower()

    if " " in value and "_" not in value:
        return value.split()

    return [value]

def _is_negation_safe_phrase(key: str) -> bool:
    """
    Phrase yang diawali kata negasi tidak diterapkan.
    Contoh:
    "tidak_menyenangkan" tidak boleh dinormalisasi menjadi bentuk yang
    menghilangkan kata "tidak", karena polaritas negasi harus diproses
    pada tahap negation handling.
    """
    first_token = key.split("_")[0]
    return first_token not in NEGATION_WORDS

def normalize_slang(tokens: List[str]) -> List[str]:
    """
    Tahap 4 - Normalisasi.
    1. Phrase rules menggunakan greedy longest-match.
    2. Token-level rules menggunakan slang dictionary, kemudian
       normalization dictionary.
    """
    out: List[str] = []
    i = 0
    total = len(tokens)


    while i < total:
        matched = False

        # ---------------------------------------------------------------
        # 1. Phrase rules
        # ---------------------------------------------------------------
        max_span = min(MAX_PHRASE_TOKENS, total - i)

        for span in range(max_span, 1, -1):
            key = "_".join(tokens[i:i + span])

            value = NORMALIZATION_DICT.get(key)

            if value is None:
                continue

            if not _is_negation_safe_phrase(key):
                # Jangan buang kata negasi.
                continue

            out.extend(_expand_normalized_value(value))
            i += span
            matched = True
            break

        if matched:
            continue

        # ---------------------------------------------------------------
        # 2. Token-level rules
        # ---------------------------------------------------------------
        token = tokens[i]
        key = token.lower()

        value = SLANG_DICT.get(
            key,
            NORMALIZATION_DICT.get(key, token),
        )

        out.extend(_expand_normalized_value(value))
        i += 1

    return out

def handle_negation(tokens: List[str]) -> List[str]:
    """
    Tahap 5 - Negation handling.
    Contoh:
    ["tidak", "bagus"] -> ["tidak_bagus"]
    """
    out: List[str] = []
    i = 0

    while i < len(tokens):
        token = tokens[i]

        if token in NEGATION_WORDS and i + 1 < len(tokens):
            out.append(f"{token}_{tokens[i + 1]}")
            i += 2
        else:
            out.append(token)
            i += 1

    return out

def remove_stopwords(
    tokens: List[str],
    exclude_negations: bool = True,
) -> List[str]:
    """
    Tahap 6 - Stopword removal.
    Token gabungan negasi seperti "tidak_bagus" selalu dipertahankan.
    """
    out: List[str] = []

    for token in tokens:
        # Token gabungan negasi harus selalu dipertahankan.
        if (
            exclude_negations
            and "_" in token
            and any(
                token.startswith(f"{negation}_")
                for negation in NEGATION_WORDS
            )
        ):
            out.append(token)
            continue

        if token in PROTECTED or token not in ALL_STOPWORDS:
            out.append(token)

    return out

def stem_tokens(tokens: List[str]) -> List[str]:
    """
    Tahap 7 - Stemming.
    Token gabungan negasi diproses per bagian:
    "tidak_bagus" -> "tidak_bagus"
    dengan stemming pada masing-masing bagian.
    """
    result: List[str] = []

    for token in tokens:
        if "_" in token:
            prefix, base = token.split("_", 1)

            stemmed_prefix = STEMMER.stem(prefix)
            stemmed_base = STEMMER.stem(base)

            result.append(
                f"{stemmed_prefix}_{stemmed_base}"
            )
        else:
            result.append(STEMMER.stem(token))

    return result

# ---------------------------------------------------------------------------
# MAIN PREPROCESSING API
# ---------------------------------------------------------------------------

def preprocess_text(
    text: str,
    return_steps: bool = False,
):
    """
    Jalankan seluruh tahapan preprocessing secara berurutan.
    """
    steps = {}

    # 1. Cleaning
    s = clean_text(text)
    steps["1_cleaning"] = s

    # 2. Case folding
    s = case_fold(s)
    steps["2_case_folding"] = s

    # 3. Tokenizing
    t = tokenize(s)
    steps["3_tokenizing"] = t

    # 4. Normalisasi
    t = normalize_slang(t)
    steps["4_normalization"] = t

    # 5. Negation handling
    t = handle_negation(t)
    steps["5_negation"] = t

    # 6. Stopword removal
    t = remove_stopwords(t)
    steps["6_stopwords"] = t

    # 7. Stemming
    t = stem_tokens(t)
    steps["7_stemming"] = t

    final = " ".join(t)

    if return_steps:
        return final, steps

    return final

def preprocess_dataframe(
    df,
    text_column: str = "content",
    show_progress: bool = True,
):
    """
    Preprocessing seluruh DataFrame.
    Output menambahkan kolom:
    processed_text
    """
    if text_column not in df.columns:
        raise KeyError(
            f"Kolom '{text_column}' tidak ditemukan. "
            f"Kolom tersedia: {list(df.columns)}"
        )

    result = df.copy()

    if show_progress:
        from tqdm import tqdm
        tqdm.pandas(desc="Preprocessing")

        result["processed_text"] = (
            result[text_column]
            .fillna("")
            .progress_apply(preprocess_text)
        )
    else:
        result["processed_text"] = (
            result[text_column]
            .fillna("")
            .apply(preprocess_text)
        )

    # Hapus baris yang menghasilkan processed_text kosong.
    result = result[
        result["processed_text"].str.strip().ne("")
    ].copy()

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_project_path(path_value: str) -> Path:
    """Resolve relative paths against the project root."""
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Preprocessing teks canonical 7 tahap untuk "
            "Sentiment Analysis Wayground."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        default="data/02_data_akhir.csv",
        help="CSV input. Default: data/02_data_akhir.csv",
    )
    parser.add_argument(
        "--output",
        default="data/pipeline_output/stage2_preprocessed.csv",
        help="CSV output hasil preprocessing 7 tahap.",
    )
    parser.add_argument(
        "--text-column",
        default="content",
        help="Kolom teks sumber. Default: content.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Nonaktifkan progress bar.",
    )

    args = parser.parse_args()

    input_path = _resolve_project_path(args.input)
    output_path = _resolve_project_path(args.output)

    print("=" * 80)
    print("STAGE 2 - PREPROCESSING TEKS CANONICAL")
    print("=" * 80)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Input        : {input_path}")
    print(f"Output       : {output_path}")
    print(f"Text column  : {args.text_column}")
    print()
    print("URUTAN 7 TAHAP:")
    print("1. Cleaning")
    print("2. Case Folding")
    print("3. Tokenizing")
    print("4. Normalisasi Slang + Phrase")
    print("5. Negation Handling")
    print("6. Stopword Removal")
    print("7. Stemming")
    print("=" * 80)

    if not input_path.exists():
        raise FileNotFoundError(
            f"File input tidak ditemukan:\n{input_path}"
        )

    if input_path.suffix.lower() != ".csv":
        raise ValueError(f"File input harus CSV: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. LOAD DATA
    # -----------------------------------------------------------------------
    print("\n[1/3] Membaca dataset...")
    try:
        df = pd.read_csv(input_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(input_path, encoding="utf-8")

    print(f"      Data input : {len(df):,}")
    print(f"      Kolom      : {list(df.columns)}")

    if args.text_column not in df.columns:
        raise KeyError(
            f"Kolom '{args.text_column}' tidak ditemukan.\n"
            f"Kolom tersedia: {list(df.columns)}"
        )

    # -----------------------------------------------------------------------
    # 2. RUN ALL 7 STAGES
    # -----------------------------------------------------------------------
    print("\n[2/3] Menjalankan preprocessing 7 tahap...")
    print("      Cleaning -> Case Folding -> Tokenizing ->")
    print("      Normalisasi -> Negation Handling ->")
    print("      Stopword Removal -> Stemming")

    result = preprocess_dataframe(
        df,
        text_column=args.text_column,
        show_progress=not args.no_progress,
    )

    # -----------------------------------------------------------------------
    # 3. SAVE
    # -----------------------------------------------------------------------
    print("\n[3/3] Menyimpan hasil...")
    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 80)
    print("STAGE 2 - PREPROCESSING SELESAI")
    print("=" * 80)
    print(f"Data input       : {len(df):,}")
    print(f"Data output      : {len(result):,}")
    print(f"Data kosong drop : {len(df) - len(result):,}")
    print(f"Output           : {output_path}")
    print()
    print("7 tahap preprocessing yang dijalankan:")
    print("[1] Cleaning")
    print("[2] Case Folding")
    print("[3] Tokenizing")
    print("[4] Normalisasi Slang + Phrase")
    print("[5] Negation Handling")
    print("[6] Stopword Removal")
    print("[7] Stemming")
    print("=" * 80)

if __name__ == "__main__":
    main()
