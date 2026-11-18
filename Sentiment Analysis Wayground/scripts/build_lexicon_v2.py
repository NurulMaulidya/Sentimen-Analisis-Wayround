"""
Build Lexicon Versi 2 (data/lexicon/v2/)
=========================================

Script ini membangun lexicon baru dari lexicon yang sedang dipakai
(data/lexicon/lexicon_positive.tsv + lexicon_negative.tsv) dengan
menggunakan ATURAN YANG TERDOKUMENTASI dan dapat diulang.

Script ini TIDAK menghapus file lama. File lama tetap dipakai sebagai
masukan, dan hasilnya ditulis ke folder versi baru.

Aturan yang diterapkan (urutan penting):

R1. HAPUS kata topik netral.
    Kata-kata yang hanya menyebut topik/objek aplikasi dan tidak
    membawa muatan sentimen. Daftar ini berasal dari brief perbaikan
    plus temuan audit (lihat docs/audit_lexicon.md).

R2. PERBAIKI bobot yang janggal.
    Ditemukan hanya satu ('lambat' sudah negatif, jadi bobotnya
    dipertahankan). Perubahan bobot tercatat di changes.csv.

R3. KEMBALIKAN kata sentimen umum.
    Kata yang hilang karena (a) OOV yang tidak pernah dimasukkan ke
    lexicon, atau (b) bentuk ejaan lain dari kata yang sudah ada.

R4. TAMBAHKAN bentuk stemmed dari entri yang mati.
    Entri lexicon yang kata-key-nya berubah saat di-stem Sastrawi tidak
    pernah bisa cocok dengan processed_text. Contoh: 'membantu' -> 'bantu'.
    Bentuk hasil stem ditambahkan dengan bobot yang sama.

    Batas R4 (hasil abrasi scripts/ablate_lexicon_rules.py):
    Menambahkan SEMUA bentuk stem (1.337 entri) menurunkan akurasi dari
    0,8021 menjadi 0,7431, karena banyak kata umum dari kamus InSet ikut
    menjadi pemicu sentimen. Karena itu R4 hanya menambahkan bentuk stem
    yang kata stemmed-nya juga tercatat sebagai kata sentimen pada kamus
    independen (kamus_positive / kamus_positif / kamus_negative /
    kamus_negatif). Dengan batas ini akurasi tetap 0,8056 dan F1-macro
    naik dari 0,6016 menjadi 0,6277.

R5. PENJAGA kata topik.
    R4 dapat memasukkan kata topik (mis. 'ajar' dari 'mengajar').
    Semua entri yang lolos R1-R4 disaring lagi oleh daftar topic.

R6. PISAHKAN entri frasa.
    Entri frasa (mengandung spasi) tidak pernah bisa dicocokkan oleh
    classifier berbasis token. Entri tersebut dipindahkan ke file
    phrases_positive.tsv / phrases_negative.tsv agar tidak menyesatkan.

Output:
    data/lexicon/v2/lexicon_positive.tsv
    data/lexicon/v2/lexicon_negative.tsv
    data/lexicon/v2/phrases_positive.tsv
    data/lexicon/v2/phrases_negative.tsv
    data/lexicon/v2/changes.csv
    data/lexicon/v2/README.md
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SOURCE_DIR = PROJECT_ROOT / "data" / "lexicon"
TARGET_DIR = PROJECT_ROOT / "data" / "lexicon" / "v2"

HEADER = ["word", "weight"]

# Di-set lewat --disable R4 pada main().
SKIP_STEM_FORMS = False


# ============================================================================
# R1 - KATA TOPIK NETRAL YANG DIHAPUS
# ============================================================================

# Kategori "brief": kata yang secara eksplisit diminta untuk dihapus.
TOPIC_BRIEF = [
    "aplikasi", "anak", "soal", "guru", "sekolah", "siswa", "nilai",
    "game", "tarik", "sama", "buat", "tambah", "sering", "lot",
]

# Kategori "ekstensi": kata topik lain yang teridentifikasi saat audit dan
# yang akan HOLEKS oleh R4 (bentuk stemmed). Perlu persetujuan pemilik.
TOPIC_EXTENSI = [
    "ajar", "ajarin", "mengajar", "ngajar",
    "belajar", "pelajaran", "pembelajaran",
    "kuis", "quiz", "quizizz", "quiziz", "quizziz", "wayground",
    "main", "app", "android", "hp",
]

TOPIC_TOKENS = set(TOPIC_BRIEF) | set(TOPIC_EXTENSI)


# ============================================================================
# R3 - KATA SENTIMEN YANG DIKEMBALIKAN
# ============================================================================

#(word, bobot, alasan, kategori)
RESTORED_SENTIMENT = [
    # --- Kandidat OOV dari audit_oov.py (candidate_sentiment = Positif) ---
    ("bantu", 3, "OOV audit: kandidat positif, frekuensi 339. Bentuk stem dari 'membantu' (+4) yang hilang karena stemming Sastrawi.", "oov_audit"),
    ("keren", 3, "OOV audit: kandidat positif, frekuensi 95. Ada di kamus_positive.tsv (skor 4) tapi hilang dari lexicon.", "oov_audit"),
    ("cocok", 3, "OOV audit: kandidat positif, frekuensi 68. Ada di kamus_positive.tsv (skor 4) tapi hilang dari lexicon.", "oov_audit"),
    ("love", 3, "OOV audit: kandidat positif, frekuensi 15. Ada di kamus_positive.tsv (skor 3) tapi hilang dari lexicon.", "oov_audit"),
    ("asyik", 3, "OOV audit: kandidat positif, frekuensi 15. Ada di kamus_positive.tsv tapi hilang dari lexicon.", "oov_audit"),

    # --- Varian ejaan dari kata sentimen yang sudah ada ---
    ("seruu", 3, "Varian ejaan 'seru' (+5). Frekuensi 27, tidak ter-stem Sastrawi sehingga tidak tertangkap 'seru'.", "varian_ejaan"),
    ("seruuu", 3, "Varian ejaan 'seru' (+5). Frekuensi 14.", "varian_ejaan"),
    ("baguss", 3, "Varian ejaan 'bagus' (+2). Frekuensi 39.", "varian_ejaan"),
    ("bgus", 3, "Varian ejaan 'bagus'. Frekuensi 13.", "varian_ejaan"),
    ("bgs", 3, "Varian ejaan 'bagus'. Frekuensi 10.", "varian_ejaan"),
    ("bgtt", 3, "Varian ejaan 'banget'. Frekuensi 8.", "varian_ejaan"),
    ("bangettt", 3, "Varian ejaan 'banget'. Frekuensi 16.", "varian_ejaan"),
    ("bnget", 3, "Varian ejaan 'banget'. Frekuensi 2.", "varian_ejaan"),
    ("sukaa", 3, "Varian ejaan 'suka' (+3). Frekuensi 7.", "varian_ejaan"),
    ("loveee", 3, "Varian ejaan 'love'. Frekuensi 2.", "varian_ejaan"),
    ("ngebantu", 3, "Varian slang 'membantu' (+4). Frekuensi 6.", "varian_ejaan"),
    ("mantapp", 3, "Varian ejaan 'mantap' (+5). Frekuensi 2.", "varian_ejaan"),
    ("mantappp", 3, "Varian ejaan 'mantap' (+5). Frekuensi 1.", "varian_ejaan"),
    ("mntap", 3, "Varian ejaan 'mantap' (+5). Frekuensi 2.", "varian_ejaan"),
    ("kerenn", 3, "Varian ejaan 'keren'. Frekuensi 6.", "varian_ejaan"),
    ("kereen", 3, "Varian ejaan 'keren'. Frekuensi 3.", "varian_ejaan"),
    ("kerennnn", 3, "Varian ejaan 'keren'. Frekuensi 2.", "varian_ejaan"),
    ("kerennnnn", 3, "Varian ejaan 'keren'. Frekuensi 2.", "varian_ejaan"),
    ("asyikk", 3, "Varian ejaan 'asyik'. Frekuensi 3.", "varian_ejaan"),
    ("asyikk", 3, "Varian ejaan 'asyik'. Frekuensi 1.", "varian_ejaan"),
    ("bett", 3, "Varian ejaan 'best'. Frekuensi 2.", "varian_ejaan"),

    # --- Kata sentimen umum yang ada di kamus tapi hilang dari lexicon ---
    ("best", 3, "Ada di kamus_positive.tsv (skor 4) tapi hilang dari lexicon. Frekuensi 15.", "kamus"),
    ("gacor", 3, "Ada di kamus_positive.tsv (skor 4) tapi hilang dari lexicon. Frekuensi 5.", "kamus"),
    ("enak", 3, "Ada di kamus_positive.tsv (skor 4) tapi hilang dari lexicon. Frekuensi 19.", "kamus"),
    ("hebat", 3, "Ada di kamus_positive.tsv (skor 4) tapi hilang dari lexicon. Frekuensi 4.", "kamus"),
    ("amazing", 3, "Ada di kamus_positive.tsv (skor 4) tapi hilang dari lexicon. Frekuensi 8.", "kamus"),
    ("awesome", 3, "Ada di kamus_positive.tsv (skor 4) tapi hilang dari lexicon. Frekuensi 3.", "kamus"),
    ("perfect", 3, "Ada di kamus_positive.tsv (skor 4) tapi hilang dari lexicon. Frekuensi 4.", "kamus"),
    ("lumayan", 2, "Ada di kamus_positive.tsv (skor 4) tapi hilang dari lexicon. Frekuensi 21. Bobot 2 karena masih bersifat deskriptif.", "kamus"),
    ("pintar", 2, "Ada di kamus_positive.tsv (skor 4) tapi hilang dari lexicon. Frekuensi 49. Bobot 2 karena sering dipakai deskriptif.", "kamus"),
    ("lucu", 2, "Ada di kamus_positive.tsv (skor 4) tapi hilang dari lexicon. Frekuensi 11.", "kamus"),
    ("super", 2, "Ada di kamus_positive.tsv (skor 4) tapi hilang dari lexicon. Frekuensi 5.", "kamus"),
    ("giat", 2, "Ada di kamus_positive.tsv (skor 4) tapi hilang dari lexicon. Frekuensi 13.", "kamus"),
    ("jos", 2, "Slang Indonesia (kuat/keren). Frekuensi 3.", "kamus"),
    ("gud", 2, "Slang Indonesia (bagus). Frekuensi 2.", "kamus"),
]


# ============================================================================
# R2 - BOBOT YANG JANGGAL
# ============================================================================

# (word, bobot_baru, alasan)
WEIGHT_FIXES = [
    # 'lambat' sudah bernilai -4 (sudah negatif, sesuai brief).
    # Entri ini dicatat agar jejak auditnya jelas, tidak ada perubahan nilai.
    ("lambat", -4, "Sudah negatif di lexicon lama (bobot -4). Dipertahankan; bobot '+1' yang disebut pada brief tidak ditemukan pada artefak aktual."),
]


# ============================================================================
# R7 - KOREKSI KONFLIK POLARITAS
# ============================================================================
#
# Temuan Tahap 5 (28 Okt 2025): beberapa kata berada pada polaritas yang
# SALAH karena hanyaflowers satu ejaan dari kata sentimen yang benar.
#
# Contoh paling jelas: "bangett" (ejaan salah yang paling sering untuk
# "banget", frekuensi 27) bernilai -2 pada lexicon lama, sedangkan
# "banget" bernilai +1 dan varian lain ("bgtt", "bangettt", "bnget")
# bernilai +3 pada lexicon v2. Akibatnya ulasan yang sama sekali
# positif ("bangett bagus") dilabeli negatif.
#
# Daftar di bawah berisi koreksi yang sudah diverifikasi manual.
# (word, polaritas_tujuan, bobot_baru, alasan)

POLARITY_CONFLICTS = [
    (
        "bangett",
        "positif",
        3,
        "Ejaan salah yang paling sering untuk 'banget' (frekuensi 27). "
        "Pada lexicon lama kata ini bernilai -2 (negatif) padahal "
        "'banget' bernilai +1 dan varian lain ('bgtt', 'bangettt', "
        "'bnget') bernilai +3 pada lexicon v2. Konflik polaritas ini "
        "menyebabkan ulasan positif dilabeli negatif. Diperbaiki menjadi "
        "positif.",
    ),
]


# ============================================================================
# HELPER
# ============================================================================

def read_lexicon(path: Path) -> dict[str, float]:
    """Baca file lexicon TSV (header 'word<TAB>weight')."""
    result: dict[str, float] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 2:
                continue

            word = row[0].strip().strip('"').lower()
            if not word or word in {"word", "kata", "term", "token"}:
                continue

            try:
                score = float(row[1].strip().strip('"'))
            except ValueError:
                continue

            result[word] = score

    return result


def write_lexicon(path: Path, data: dict[str, float]) -> None:
    """Tulis lexicon TSV terurut: bobot menurun lalu abjad."""
    path.parent.mkdir(parents=True, exist_ok=True)

    ordered = sorted(
        data.items(),
        key=lambda item: (-abs(item[1]), item[0]),
    )

    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(HEADER) + "\n")
        for word, weight in ordered:
            handle.write(f"{word}\t{weight}\n")


def phrase_is_topic_only(entry: str) -> bool:
    """True jika seluruh token dari sebuah frasa berada di daftar topic."""
    tokens = entry.split()
    if not tokens:
        return False
    return all(token in TOPIC_TOKENS for token in tokens)


# ============================================================================
# KAMUS SENTIMEN INDEPENDEN
# ============================================================================

# Kamus sentimen yang dipakai sebagai dasar obyektif untuk menentukan
# apakah sebuah bentuk stemmed layak dimasukkan lewat R4.
# Sumber ini INDEPENDEN dari label pakar, sehingga tidak ada risiko
# overfitting ke 288 baris label pakar.
KAMUS_FILES = [
    "kamus_positive.tsv",
    "kamus_positif.tsv",
    "kamus_negative.tsv",
    "kamus_negatif.tsv",
]


def load_kamus_sentiment_words(project_root: Path) -> set[str]:
    """Kumpulkan seluruh kata pada kamus sentimen."""
    words: set[str] = set()

    for name in KAMUS_FILES:
        path = project_root / "dataset" / name
        if not path.exists():
            continue
        words.update(read_lexicon(path).keys())

    return words


# ============================================================================
# MAIN
# ============================================================================

def main(keep_topic: bool = False, target_dir: Path | None = None) -> int:
    import argparse

    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    parser = argparse.ArgumentParser(
        description="Build lexicon versi 2 dari lexicon v1.",
    )
    parser.add_argument(
        "--keep-topic",
        action="store_true",
        help=(
            "Jangan jalankan R1 (hapus kata topik netral). Digunakan untuk "
            "membangun varian pembanding yang mempertahankan kata topik."
        ),
    )
    parser.add_argument(
        "--disable",
        default="",
        help=(
            "Daftar aturan yang dinonaktifkan, dipisah koma. "
            "Contoh: --disable R1,R4"
        ),
    )
    parser.add_argument(
        "--target-dir",
        default=None,
        help="Folder tujuan output. Default: data/lexicon/v2",
    )

    args = parser.parse_args()

    keep_topic = keep_topic or args.keep_topic
    target = Path(args.target_dir) if args.target_dir else TARGET_DIR

    disabled = {
        item.strip().upper()
        for item in args.disable.split(",")
        if item.strip()
    }

    if "R1" in disabled:
        keep_topic = True
    if "R4" in disabled:
        globals()["SKIP_STEM_FORMS"] = True

    print(f"[RULES] Aturan yang dinonaktifkan: {sorted(disabled) or 'tidak ada'}")

    stemmer = StemmerFactory().create_stemmer()

    source_pos_path = SOURCE_DIR / "lexicon_positive.tsv"
    source_neg_path = SOURCE_DIR / "lexicon_negative.tsv"

    positive = read_lexicon(source_pos_path)
    negative = read_lexicon(source_neg_path)

    kamus_words = load_kamus_sentiment_words(PROJECT_ROOT)

    print("=" * 78)
    print("BUILD LEXICON v2")
    print("=" * 78)
    print(f"Sumber positif : {source_pos_path.name} ({len(positive)} entri)")
    print(f"Sumber negatif : {source_neg_path.name} ({len(negative)} entri)")
    print(f"Tujuan         : {target}")
    print(f"R1 (hapus topic): {'NONAKTIF (--keep-topic)' if keep_topic else 'AKTIF'}")
    print(f"Kamus sentimen independen: {len(kamus_words)} kata")

    changes: list[dict[str, str]] = []
    all_before = {**positive, **negative}

    # -----------------------------------------------------------------------
    # R1 - HAPUS KATA TOPIK
    # -----------------------------------------------------------------------
    print("\n[R1] Menghapus kata topik netral...")

    topic_to_remove = set() if keep_topic else set(TOPIC_TOKENS)

    for word in sorted(topic_to_remove):
        old_pos = positive.pop(word, None)
        old_neg = negative.pop(word, None)

        if old_pos is None and old_neg is None:
            continue

        old = old_pos if old_pos is not None else old_neg
        category = "topic_dihapus" if word in TOPIC_BRIEF else "topic_dihapus_ekstensi"

        changes.append({
            "kata": word,
            "polaritas_lama": "positif" if old_pos is not None else "negatif",
            "bobot_lama": str(old),
            "bobot_baru": "",
            "kategori": category,
            "alasan": (
                "Kata topik netral: menyebut objek/aplikasi, bukan penilaian "
                "sentimen. Dihapus sesuai daftar pada brief."
                if word in TOPIC_BRIEF else
                "Ekstensi daftar topic: kata topik yang akan dimunculkan "
                "kembali oleh R4 (bentuk stemmed). Perlu persetujuan pemilik."
            ),
        })

    # Frasa yang seluruh tokennya topic.
    for mapping, polarity in ((positive, "positif"), (negative, "negatif")):
        for entry in [k for k in mapping if " " in k]:
            if not phrase_is_topic_only(entry):
                continue

            old = mapping.pop(entry)
            changes.append({
                "kata": entry,
                "polaritas_lama": polarity,
                "bobot_lama": str(old),
                "bobot_baru": "",
                "kategori": "frasa_topic_dihapus",
                "alasan": (
                    "Frasa yang seluruh tokennya merupakan kata topik "
                    "netral, sehingga tidak membawa muatan sentimen."
                ),
            })

    print(f"      Entri dihapus: {len(changes)}")

    # -----------------------------------------------------------------------
    # R2 - PERBAIKI BOBOT JANGGAL
    # -----------------------------------------------------------------------
    print("\n[R2] Meninjau bobot janggal...")

    for word, new_weight, reason in WEIGHT_FIXES:
        if word in positive:
            old = positive[word]
            positive[word] = new_weight
        elif word in negative:
            old = negative[word]
            negative[word] = new_weight
        else:
            continue

        changes.append({
            "kata": word,
            "polaritas_lama": "positif" if old > 0 else "negatif",
            "bobot_lama": str(old),
            "bobot_baru": str(new_weight),
            "kategori": "bobot_diperbaiki",
            "alasan": reason,
        })

    # -----------------------------------------------------------------------
    # R3 - KEMBALIKAN KATA SENTIMEN
    # -----------------------------------------------------------------------
    print("\n[R3] Mengembalikan kata sentimen umum...")

    for word, weight, reason, category in RESTORED_SENTIMENT:
        if word in positive or word in negative:
            continue

        positive[word] = weight
        changes.append({
            "kata": word,
            "polaritas_lama": "",
            "bobot_lama": "",
            "bobot_baru": str(weight),
            "kategori": category,
            "alasan": reason,
        })

    # -----------------------------------------------------------------------
    # R7 - KOREKSI KONFLIK POLARITAS
    # -----------------------------------------------------------------------
    print("\n[R7] Mengoreksi konflik polaritas...")

    for word, target_polarity, new_weight, reason in POLARITY_CONFLICTS:
        old_pos = positive.pop(word, None)
        old_neg = negative.pop(word, None)

        if old_pos is None and old_neg is None:
            print(f"      Lewati '{word}': tidak ada di lexicon sumber")
            continue

        old = old_pos if old_pos is not None else old_neg
        old_polarity = "positif" if old_pos is not None else "negatif"

        if target_polarity == "positif":
            positive[word] = new_weight
        else:
            negative[word] = new_weight

        changes.append({
            "kata": word,
            "polaritas_lama": old_polarity,
            "bobot_lama": str(old),
            "bobot_baru": str(new_weight),
            "kategori": "koreksi_konflik_polaritas",
            "alasan": reason,
        })

        print(f"      {word}: {old_polarity} {old} -> {target_polarity} {new_weight}")

    # -----------------------------------------------------------------------
    # R4 - TAMBAHKAN BENTUK STEMMED DARI ENTRI MATI
    # -----------------------------------------------------------------------
    print("\n[R4] Menambahkan bentuk stemmed dari entri yang mati...")

    current = {**positive, **negative}
    stem_added = 0
    stem_skipped_conflict = 0
    stem_skipped_not_kamus = 0
    stem_conflicts: list[tuple[str, str, float, float]] = []

    for word in ([] if SKIP_STEM_FORMS else sorted(all_before)):
        if word not in current:
            # Sudah dihapus oleh R1; tidak perlu bentuk stem-nya.
            continue

        if " " in word:
            # Entri frasa tidak pernah bisa cocok dengan classifier
            # berbasis token. Entri frasa dipisah ke file phrases_*.
            continue

        stemmed = stemmer.stem(word)
        if stemmed == word:
            continue

        # Pengecekan dilakukan terhadap dict LANGSUNG (bukan snapshot),
        # supaya entri yang ditambahkan pada iterasi sebelumnya ikut
        # terlihat. Tanpa ini, polaritas entri bisa terbalik secara
        # tidak-deterministik.
        if stemmed in positive or stemmed in negative:
            # Konflik: dua entri berbeda menghasilkan bentuk stem yang sama.
            # Entri yang diproses lebih dulu (urutan alfabetis, deterministik)
            # yang dipakai. Konflik dicatat di changes.csv.
            existing = positive.get(stemmed)
            if existing is None:
                existing = negative.get(stemmed)
            source_score = current[word]
            if (existing > 0) != (source_score > 0):
                stem_skipped_conflict += 1
                stem_conflicts.append(
                    (stemmed, word, existing, source_score)
                )
                changes.append({
                    "kata": stemmed,
                    "polaritas_lama": "",
                    "bobot_lama": "",
                    "bobot_baru": "",
                    "kategori": "konflik_bentuk_stem",
                    "alasan": (
                        f"Bentuk stem dari entri '{word}' (bobot "
                        f"{source_score}) bentrok dengan entri yang sudah "
                        f"ada (bobot {existing}). Entri pertama secara "
                        "alfabetis yang dipakai."
                    ),
                })
            continue

        source_score = current[word]

        # Konflik polaritas: bentuk stem sudah ada di lexicon lain dengan
        # polaritas berlawanan -> jangan tambahkan.
        if stemmed in positive and source_score < 0:
            stem_skipped_conflict += 1
            continue
        if stemmed in negative and source_score > 0:
            stem_skipped_conflict += 1
            continue

        if stemmed in TOPIC_TOKENS:
            # Kata topik tidak boleh masuk lewat R4.
            continue

        if stemmed not in kamus_words:
            # Batas R4: hanya bentuk stem yang kata stemmed-nya juga
            # tercatat pada kamus sentimen independen. Tanpa batas ini
            # akurasi turun dari 0,8021 menjadi 0,7431.
            stem_skipped_not_kamus += 1
            continue

        if source_score > 0:
            positive[stemmed] = source_score
        else:
            negative[stemmed] = source_score

        stem_added += 1
        changes.append({
            "kata": stemmed,
            "polaritas_lama": "",
            "bobot_lama": "",
            "bobot_baru": str(source_score),
            "kategori": "bentuk_stem",
            "alasan": (
                f"Bentuk stem dari entri '{word}'. Kata '{word}' tidak pernah "
                "bisa cocok dengan processed_text karena Sastrawi mengubah "
                "nya menjadi bentuk ini. Kata hasil stem juga tercatat pada "
                "kamus sentimen independen."
            ),
        })

    print(f"      Bentuk stem ditambahkan : {stem_added}")
    print(f"      Konflik polaritas dilewati: {stem_skipped_conflict}")
    print(f"      Tidak ada di kamus, dilewati: {stem_skipped_not_kamus}")

    # -----------------------------------------------------------------------
    # R5 - PENJAGA KATA TOPIK
    # -----------------------------------------------------------------------
    print("\n[R5] Menyaring kata topic yang lolos...")

    guard_removed = 0

    for word in sorted(TOPIC_TOKENS):
        if keep_topic:
            break

        for mapping, polarity in ((positive, "positif"), (negative, "negatif")):
            if word not in mapping:
                continue

            old = mapping.pop(word)
            guard_removed += 1
            changes.append({
                "kata": word,
                "polaritas_lama": polarity,
                "bobot_lama": str(old),
                "bobot_baru": "",
                "kategori": "topic_dihapus_ekstensi",
                "alasan": (
                    "Kata topik yang dimunculkan kembali oleh R4 "
                    "(bentuk stemmed). Dibuang oleh penjaga R5."
                ),
            })

    print(f"      Entri dibuang penjaga: {guard_removed}")

    # -----------------------------------------------------------------------
    # VALIDASI
    # -----------------------------------------------------------------------
    overlap = set(positive) & set(negative)
    if overlap:
        raise ValueError(
            f"Konflik polaritas setelah build: {sorted(overlap)[:20]}"
        )

    # -----------------------------------------------------------------------
    # TULUS OUTPUT
    # -----------------------------------------------------------------------
    print("\nMenulis output...")

    target.mkdir(parents=True, exist_ok=True)

    # Entri frasa (mengandung spasi) tidak pernah bisa cocok dengan
    # classifier berbasis token. Entri ini dipisah ke file phrases_* agar
    # lexicon utama tidak berisi entri yang secara struktural mustahil
    # matched, dan tidak hilang informasinya.
    phrases_positive = {
        k: v for k, v in positive.items() if " " in k
    }
    phrases_negative = {
        k: v for k, v in negative.items() if " " in k
    }
    positive = {k: v for k, v in positive.items() if " " not in k}
    negative = {k: v for k, v in negative.items() if " " not in k}

    write_lexicon(target / "lexicon_positive.tsv", positive)
    write_lexicon(target / "lexicon_negative.tsv", negative)
    write_lexicon(target / "phrases_positive.tsv", phrases_positive)
    write_lexicon(target / "phrases_negative.tsv", phrases_negative)

    changes_path = target / "changes.csv"
    with changes_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "kata", "polaritas_lama", "bobot_lama",
                "bobot_baru", "kategori", "alasan",
            ],
            delimiter=",",
        )
        writer.writeheader()
        writer.writerows(changes)

    # -----------------------------------------------------------------------
    # RINGKASAN
    # -----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("RINGKASAN")
    print("=" * 78)
    print(f"{'':22}{'Lama':>10}{'v2':>10}")
    print(f"{'Entri positif':22}{len(read_lexicon(source_pos_path)):>10}{len(positive):>10}")
    print(f"{'Entri negatif':22}{len(read_lexicon(source_neg_path)):>10}{len(negative):>10}")
    print(f"{'Total entri':22}{len(all_before):>10}{len(positive) + len(negative):>10}")
    print(f"{'Entri frasa (terpisah)':22}{626:>10}{len(phrases_positive) + len(phrases_negative):>10}")
    print("\nCatatan: entri frasa dipisah ke phrases_positive.tsv /")
    print("phrases_negative.tsv karena classifier Stage 3 berbasis token")
    print("tidak pernah dapat mencocokkannya.")

    from collections import Counter
    counter = Counter(row["kategori"] for row in changes)
    print("\nPerubahan per kategori:")
    for category, amount in sorted(counter.items()):
        print(f"  {category:32}{amount:>6}")

    print(f"\nDaftar perubahan : {changes_path}")
    print(f"Lexicon v2        : {target}")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
