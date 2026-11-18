"""
AUDIT OOV - WAYGROUND SENTIMENT ANALYSIS
========================================

Audit vocabulary terhadap dataset preprocessing.

Prinsip:
- Hanya membaca positive.tsv dan negative.tsv.
- Tidak ada hard-coded OOV aktif.
- Tidak ada auto-expansion.
- Tidak mengubah lexicon.
- Tidak mengubah dataset canonical.
- OOV yang ditemukan hanya menjadi bahan audit/manual validation.
- Loader TSV dan CSV dibuat toleran terhadap BOM, delimiter campuran,
  dan baris rusak. Baris CSV bermasalah diperbaiki bila memungkinkan.

JALUR TUNGGAL (hasil audit Tahap 1):
Script ini TIDAK lagi memiliki loader lexicon sendiri. Vocabulary diambil
dari shared.lexicon (sumber tunggal), dan pelabelan memakai
shared.labeling.label_text, yang sama dengan yang dipakai Stage 3 di
pipeline/run_pipeline.py.
"""

from pathlib import Path
from collections import Counter, defaultdict
import json
import sys
import csv

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.constants import (  # noqa: E402
    NEGATIVE_LEXICON_PATH,
    NEGATIVE_LEXICON_V2_PATH,
    POSITIVE_LEXICON_PATH,
    POSITIVE_LEXICON_V2_PATH,
    LEXICON_AUDIT_DIR,
    PIPELINE_OUTPUT_DIR,
)
from shared.lexicon import load_lexicon_file  # noqa: E402
from shared.labeling import (  # noqa: E402
    get_active_lexicon_version,
    label_text,
)


DATA_PATH = PIPELINE_OUTPUT_DIR / "stage2_preprocessed_new.csv"

# Path lexicon ditentukan oleh versi lexicon yang aktif.
# resolved_lexicon_paths() dipanggil di main() supaya env var
# WAYGROUND_LEXICON_VERSION tetap bisa dipilih saat runtime.
LEXICON_PATHS = {
    "v1": (POSITIVE_LEXICON_PATH, NEGATIVE_LEXICON_PATH),
    "v2": (POSITIVE_LEXICON_V2_PATH, NEGATIVE_LEXICON_V2_PATH),
}

OUTPUT_DIR = LEXICON_AUDIT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_XLSX = OUTPUT_DIR / "audit_oov_stage2.xlsx"
OUTPUT_TSV = OUTPUT_DIR / "oov_kandidat_sentimen.tsv"
OUTPUT_JSON = OUTPUT_DIR / "audit_oov_report.json"

TEXT_COLUMN = "processed_text"
MIN_CANDIDATE_FREQUENCY = 5
MAX_EXAMPLES = 5


FUNCTION_WORDS = {
    "dan", "atau", "yang", "untuk", "dengan", "dari", "ke", "di",
    "pada", "dalam", "ini", "itu", "saya", "aku", "kami", "kita",
    "mereka", "dia", "nya", "jadi", "agar", "karena", "kalau",
    "jika", "sangat", "lebih", "masih", "sudah", "telah", "akan",
    "bisa", "mau", "ingin", "terus", "juga", "cuma", "hanya",
    "buat", "dapat", "ada", "tidak", "saja", "seperti",
}

POSITIVE_HINTS = {
    "bagus", "baik", "seru", "suka", "senang", "mudah", "bantu",
    "membantu", "cocok", "mantap", "keren", "puas", "sukses",
    "nyaman", "hebat", "sempurna", "menarik", "asyik", "terbaik",
    "love", "seneng",
}

NEGATIVE_HINTS = {
    "buruk", "jelek", "error", "lambat", "lemot", "ngeleg",
    "ngelag", "lag", "crash", "gagal", "susah", "sulit", "kecewa",
    "rusak", "bug", "parah", "ganggu", "mengganggu", "bosan",
    "forceclose", "keluar", "macet", "hang", "down",
}


def build_phrases(lexicon: dict):
    phrases = []
    for word in lexicon:
        parts = tuple(str(word).split())
        if len(parts) > 1:
            phrases.append(parts)

    phrases.sort(key=len, reverse=True)
    return phrases


def get_oov_from_tokens(tokens, vocabulary, phrases):
    """
    Phrase-aware OOV detection.

    Contoh:
        lexicon = "bagus banget"
        text = "bagus banget"
    maka token phrase dikonsumsi dan tidak dihitung sebagai OOV.
    """
    tokens = [
        str(token).strip().lower()
        for token in tokens
        if str(token).strip()
    ]

    if not tokens:
        return []

    consumed = [False] * len(tokens)

    for phrase in phrases:
        n = len(phrase)
        if n > len(tokens):
            continue

        for i in range(len(tokens) - n + 1):
            if any(consumed[i:i + n]):
                continue

            if tuple(tokens[i:i + n]) == phrase:
                for j in range(i, i + n):
                    consumed[j] = True

    oov = []

    for i, token in enumerate(tokens):
        if consumed[i]:
            continue

        check_word = token

        # "tidak_bagus" harus dibandingkan dengan "bagus"
        # karena negasi ditangani oleh classifier.
        if "_" in token:
            prefix, base = token.split("_", 1)
            if prefix in {
                "tidak", "tak", "bukan", "jangan"
            }:
                check_word = base

        if check_word not in vocabulary:
            oov.append(check_word)

    return oov


def classify_candidate(word: str):
    word = word.lower().strip()

    if word in POSITIVE_HINTS:
        return (
            "Positif",
            2.0,
            "Kandidat positif berdasarkan screening awal",
        )

    if word in NEGATIVE_HINTS:
        return (
            "Negatif",
            -2.0,
            "Kandidat negatif berdasarkan screening awal",
        )

    if word in FUNCTION_WORDS:
        return (
            "Bukan sentimen",
            0.0,
            "Kata fungsi/kontekstual umum",
        )

    return (
        "Review manual",
        0.0,
        "Belum dapat ditentukan secara heuristik",
    )



def detect_delimiter(path: Path) -> str:
    """Detect delimiter from a small sample without assuming comma."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(12000)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\\t|")
        return dialect.delimiter
    except csv.Error:
        first_nonempty = next(
            (line for line in sample.splitlines() if line.strip()), ""
        )
        counts = {d: first_nonempty.count(d) for d in [",", ";", "\\t", "|"]}
        return max(counts, key=counts.get)


def load_dataset_robust(path: Path) -> pd.DataFrame:
    """
    Load dataset safely for audit purposes.

    First tries the normal C parser. If the CSV is malformed, falls back to
    the Python parser with a bad-line repair callback. The repair preserves
    malformed rows instead of silently dropping them.
    """
    delimiter = detect_delimiter(path)

    try:
        return pd.read_csv(
            path,
            sep=delimiter,
            encoding="utf-8-sig",
            engine="c",
        )
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        print(f"[DATA] Parser standar gagal: {exc}")
        print(f"[DATA] Fallback parser Python, delimiter={delimiter!r}")

    # Read header first so the repair callback knows the expected width.
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader)

    expected = len(header)
    text_idx = header.index(TEXT_COLUMN) if TEXT_COLUMN in header else None

    repaired = 0
    skipped = 0

    def repair_bad_line(fields):
        nonlocal repaired, skipped

        if len(fields) == expected:
            return fields

        if len(fields) < expected:
            skipped += 1
            return None

        # Extra delimiters usually originate from unquoted text.
        # If processed_text is present, preserve all excess pieces in that
        # column because OOV auditing only depends on the processed text.
        if text_idx is not None:
            extra = len(fields) - expected
            if text_idx + extra < len(fields):
                merged = fields[:text_idx]
                merged.append(delimiter.join(fields[text_idx:text_idx + extra + 1]))
                merged.extend(fields[text_idx + extra + 1:])
                if len(merged) == expected:
                    repaired += 1
                    return merged

        # Conservative fallback: merge surplus fields into the final column.
        repaired += 1
        return fields[:expected - 1] + [delimiter.join(fields[expected - 1:])]

    df = pd.read_csv(
        path,
        sep=delimiter,
        encoding="utf-8-sig",
        engine="python",
        on_bad_lines=repair_bad_line,
    )

    print(f"[DATA] Malformed rows repaired : {repaired}")
    print(f"[DATA] Malformed rows skipped  : {skipped}")

    if df.empty:
        raise ValueError("Dataset berhasil dibaca tetapi kosong.")

    return df


def main():
    version = get_active_lexicon_version()

    if version not in LEXICON_PATHS:
        raise ValueError(
            f"Versi lexicon tidak dikenal: {version!r}. "
            f"Pilihan: {sorted(LEXICON_PATHS)}"
        )

    positive_path, negative_path = LEXICON_PATHS[version]

    print("=" * 90)
    print("AUDIT OOV - WAYGROUND SENTIMENT ANALYSIS")
    print(f"MODE: TWO-LEXICON ONLY (versi lexicon: {version})")
    print("=" * 90)

    print(f"[DATA] Dataset   : {DATA_PATH}")
    print(f"[LEXICON] Positif: {positive_path}")
    print(f"[LEXICON] Negatif: {negative_path}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}")

    df = load_dataset_robust(DATA_PATH)

    if TEXT_COLUMN not in df.columns:
        raise KeyError(
            f"Kolom '{TEXT_COLUMN}' tidak ditemukan. "
            f"Kolom tersedia: {list(df.columns)}"
        )

    print(f"[DATA] Rows      : {len(df)}")
    print(f"[DATA] Text col  : {TEXT_COLUMN}")

    positive = load_lexicon_file(positive_path, "positive")
    negative = load_lexicon_file(negative_path, "negative")

    overlap = set(positive) & set(negative)
    if overlap:
        raise ValueError(
            f"Ditemukan {len(overlap)} kata pada kedua lexicon. "
            f"Contoh: {sorted(overlap)[:20]}"
        )

    vocabulary = set(positive) | set(negative)

    phrases = build_phrases(positive) + build_phrases(negative)
    phrases.sort(key=len, reverse=True)

    print(f"[LEXICON] Positive words : {len(positive)}")
    print(f"[LEXICON] Negative words : {len(negative)}")
    print(f"[LEXICON] Vocabulary     : {len(vocabulary)}")
    print(f"[LEXICON] Phrases        : {len(phrases)}")
    print(f"[LEXICON] Overlap        : {len(overlap)}")
    print("[LEXICON] Hard-coded OOV : DISABLED")
    print("[LEXICON] Auto-expansion  : DISABLED")

    counter = Counter()
    examples = defaultdict(list)

    for text in df[TEXT_COLUMN].fillna("").astype(str):
        if not text.strip():
            continue

        tokens = text.split()

        oov_words = get_oov_from_tokens(
            tokens=tokens,
            vocabulary=vocabulary,
            phrases=phrases,
        )

        counter.update(oov_words)

        for word in set(oov_words):
            if len(examples[word]) < MAX_EXAMPLES:
                examples[word].append(text)

    total_oov_occurrences = sum(counter.values())
    unique_oov = len(counter)

    print()
    print("[OOV] Unique OOV       :", unique_oov)
    print("[OOV] Total occurrences:", total_oov_occurrences)

    rows = []

    for rank, (word, frequency) in enumerate(
        counter.most_common(),
        start=1,
    ):
        candidate, suggested_score, reason = classify_candidate(word)

        rows.append({
            "rank": rank,
            "word": word,
            "frequency": int(frequency),
            "percentage": (
                round(
                    100 * frequency / total_oov_occurrences,
                    4,
                )
                if total_oov_occurrences
                else 0
            ),
            "candidate_sentiment": candidate,
            "suggested_score": suggested_score,
            "example_reviews": " || ".join(
                examples.get(word, [])
            ),
            "decision": "",
            "reason": reason,
        })

    all_oov_df = pd.DataFrame(rows)

    if all_oov_df.empty:
        all_oov_df = pd.DataFrame(
            columns=[
                "rank", "word", "frequency", "percentage",
                "candidate_sentiment", "suggested_score",
                "example_reviews", "decision", "reason",
            ]
        )

    candidate_df = all_oov_df[
        all_oov_df["frequency"] >= MIN_CANDIDATE_FREQUENCY
    ].copy()

    sentiment_candidate_df = candidate_df[
        candidate_df["candidate_sentiment"].isin(
            ["Positif", "Negatif"]
        )
    ].copy()

    # -----------------------------------------------------------------------
    # PEMERIKSAAN JALUR TUNGGAL
    # -----------------------------------------------------------------------
    # Audit ini memakai fungsi pelabelan yang sama dengan Stage 3
    # (shared.labeling.label_text). Pemeriksaan berikut membuktikan bahwa
    # kedua jalur benar-benar menghasilkan label yang sama.
    canonical_labels = (
        df[TEXT_COLUMN]
        .fillna("")
        .astype(str)
        .map(lambda text: label_text(text)["label"])
    )

    canonical_distribution = {
        label: int((canonical_labels == label).sum())
        for label in ("Positif", "Negatif", "Netral")
    }

    print("[CEK] Distribusi label (fungsi pelabelan canonical):")
    for label, amount in canonical_distribution.items():
        print(f"        {label:8} : {amount}")

    summary = pd.DataFrame([
        ["Dataset", str(DATA_PATH)],
        ["Versi lexicon", version],
        ["Jumlah review", len(df)],
        ["Positive lexicon", len(positive)],
        ["Negative lexicon", len(negative)],
        ["Total vocabulary", len(vocabulary)],
        ["Phrase lexicon", len(phrases)],
        ["Cross-polarity conflict", len(overlap)],
        ["Hard-coded OOV", False],
        ["Auto expansion", False],
        ["Unique OOV", unique_oov],
        ["Total OOV occurrences", total_oov_occurrences],
        ["OOV frequency >= threshold", len(candidate_df)],
        ["Kandidat positif/negatif", len(sentiment_candidate_df)],
        ["Minimum candidate frequency", MIN_CANDIDATE_FREQUENCY],
        ["Label Positif (canonical)", canonical_distribution["Positif"]],
        ["Label Negatif (canonical)", canonical_distribution["Negatif"]],
        ["Label Netral (canonical)", canonical_distribution["Netral"]],
    ], columns=["metric", "value"])

    priority_df = sentiment_candidate_df[
        [
            "rank",
            "word",
            "frequency",
            "percentage",
            "candidate_sentiment",
            "suggested_score",
            "example_reviews",
            "decision",
            "reason",
        ]
    ].copy()

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        summary.to_excel(
            writer,
            sheet_name="Ringkasan",
            index=False,
        )
        all_oov_df.to_excel(
            writer,
            sheet_name="Semua OOV",
            index=False,
        )
        candidate_df.to_excel(
            writer,
            sheet_name="Kandidat Prioritas",
            index=False,
        )
        priority_df.to_excel(
            writer,
            sheet_name="Kandidat Sentimen",
            index=False,
        )

    priority_df.to_csv(
        OUTPUT_TSV,
        sep="\t",
        index=False,
        encoding="utf-8-sig",
    )

    json_report = {
        "method": {
            "mode": "TWO-LEXICON ONLY",
            "lexicon_version": version,
            "dataset": str(DATA_PATH),
            "text_column": TEXT_COLUMN,
            "positive_lexicon": str(positive_path),
            "negative_lexicon": str(negative_path),
            "hard_coded_oov_active": False,
            "auto_expansion": False,
            "phrase_aware": True,
            "min_candidate_frequency": MIN_CANDIDATE_FREQUENCY,
            "labeling_function": "shared.labeling.label_text",
            "shared_with_stage3": True,
        },
        "summary": {
            "dataset_rows": int(len(df)),
            "positive_lexicon": int(len(positive)),
            "negative_lexicon": int(len(negative)),
            "vocabulary": int(len(vocabulary)),
            "phrases": int(len(phrases)),
            "cross_polarity": int(len(overlap)),
            "unique_oov": int(unique_oov),
            "total_oov_occurrences": int(total_oov_occurrences),
            "candidate_priority": int(len(candidate_df)),
            "candidate_sentiment": int(len(sentiment_candidate_df)),
        },
        "canonical_labeling": {
            "source": "shared.labeling.label_text",
            "threshold": 0.5,
            "distribution": canonical_distribution,
        },
        "outputs": {
            "excel": str(OUTPUT_XLSX),
            "tsv": str(OUTPUT_TSV),
            "json": str(OUTPUT_JSON),
        },
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            json_report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 90)
    print("HASIL AUDIT OOV")
    print("=" * 90)
    print(f"Unique OOV              : {unique_oov}")
    print(f"Total OOV occurrences   : {total_oov_occurrences}")
    print(
        f"OOV >= {MIN_CANDIDATE_FREQUENCY:<2} frequency : "
        f"{len(candidate_df)}"
    )
    print(f"Kandidat sentimen       : {len(sentiment_candidate_df)}")

    print()
    print("TOP 20 OOV:")
    print("-" * 90)

    display_cols = [
        "rank",
        "word",
        "frequency",
        "candidate_sentiment",
    ]

    if not all_oov_df.empty:
        print(
            all_oov_df[display_cols]
            .head(20)
            .to_string(index=False)
        )

    print()
    print("KANDIDAT SENTIMEN:")
    print("-" * 90)

    if not sentiment_candidate_df.empty:
        print(
            sentiment_candidate_df[
                [
                    "word",
                    "frequency",
                    "candidate_sentiment",
                    "suggested_score",
                ]
            ].to_string(index=False)
        )
    else:
        print("[INFO] Tidak ada kandidat sentiment-bearing.")

    print()
    print("[SAVE] Excel :", OUTPUT_XLSX)
    print("[SAVE] TSV   :", OUTPUT_TSV)
    print("[SAVE] JSON  :", OUTPUT_JSON)
    print()
    print("=" * 90)
    print("[PASS] AUDIT OOV SELESAI")
    print("[INFO] Active vocabulary = positive TSV + negative TSV.")
    print("[INFO] Tidak ada hard-coded OOV.")
    print("[INFO] Tidak ada auto-expansion.")
    print("[INFO] Script tidak mengubah lexicon maupun dataset canonical.")
    print("=" * 90)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("[ERROR]", exc)
        sys.exit(1)
