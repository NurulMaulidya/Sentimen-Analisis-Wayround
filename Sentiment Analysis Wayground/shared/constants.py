"""
Konstanta inti penelitian Wayground.

File ini hanya menyimpan konstanta domain:
- preprocessing
- sentiment
- aspect
- path lexicon
- visualisasi

Konfigurasi eksperimen/model TIDAK disimpan di sini.
Konfigurasi final model berada di shared.final_config.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "data" / "lexicon"

# ============================================================================
# ROOT KELUARAN (dapat di-override lewat environment variable)
# ============================================================================
# Tujuannya: hasil eksperimen baru bisa ditulis ke folder versi baru
# (mis. output_v2/, models/final_v2/) TANPA menimpa artefak lama yang
# masih dipakai sebagai pembanding.
#
#   set WAYGROUND_OUTPUT_DIR=output_v2
#   set WAYGROUND_FINAL_MODEL_DIR=models/final_v2
#
# Default menunjuk namespace canonical v4 yang sudah diverifikasi.
# Folder historis output/ dan models/final/ tidak ditimpa.
OUTPUT_ROOT_NAME = os.environ.get("WAYGROUND_OUTPUT_DIR", "output_v4")
OUTPUT_DIR = BASE_DIR / OUTPUT_ROOT_NAME
EXPERIMENTS_DIR = OUTPUT_DIR / "experiments"
AUDIT_DIR = OUTPUT_DIR / "audit"
LEXICON_AUDIT_DIR = OUTPUT_DIR / "lexicon_audit"

FINAL_MODEL_DIR = BASE_DIR / os.environ.get(
    "WAYGROUND_FINAL_MODEL_DIR", "models/final_v4"
)

# Alias lama (dipakai di beberapa script).
MODELS_DIR = BASE_DIR / "models"

# ============================================================================
# ROOT KELUARAN PIPELINE DATA (Tahap 5)
# ============================================================================
# Artefak pipeline (Stage 1-8) juga ditulis ke folder berversi supaya
# dataset lama (data/pipeline_output/) tidak tertimpa sebelum hasil baru
# lolos pemeriksaan.
#
#   set WAYGROUND_PIPELINE_OUTPUT_DIR=data/pipeline_output_v2
#
# Default menunjuk dataset canonical v4 (snapshot threshold-selected).
PIPELINE_OUTPUT_DIR = BASE_DIR / os.environ.get(
    "WAYGROUND_PIPELINE_OUTPUT_DIR", "data/pipeline_output_v4"
)

# ============================================================================
# DATASET CANONICAL
# ============================================================================
# Sumber data tunggal untuk seluruh tahap: Stage 3 (lexicon) + integrasi
# label experts. Path-nya mengikuti PIPELINE_OUTPUT_DIR supaya dataset
# berversi (Tahap 5) dipakai tanpa mengubah kode.
CANONICAL_DATASET_PATH = PIPELINE_OUTPUT_DIR / "stage3_final_labeled.csv"

# Kamus normalisasi slang dan phrase.
# CATATAN PATH: file kamus berada di folder data/. Path sebelumnya menunjuk ke
# root project sehingga kedua kamus TERMUAT KOSONG diam-diam dan tahap
# "Normalisasi" tidak menjalankan apa pun (regresi saat project dirapikan).
SLANG_DICT_PATH = BASE_DIR / "data" / "slang_dict.json"
NORMALIZATION_DICT_PATH = BASE_DIR / "data" / "normalization_dict.json"

# ============================================================================
# PARAMETER PREPROCESSING
# ============================================================================

# Kata negasi yang digabung dengan kata sesudahnya
# (sesuai Metodologi Bagian 4.4b + ekspansi untuk bahasa gaul Indonesia)
# PERBAIKAN: 'belum' dikecualikan dari negation handling
# Karena "belum" dalam konteks ulasan (misal "quiz belum selesai")
# sebenarnya adalah statement netral/deskriptif, bukan ekspresi ketidakpuasan
# FIX: Removed duplicates for cleaner code
NEGATION_WORDS = [
    # Kata negasi dasar (dari metodologi) - BUKAN 'belum'
    'tidak', 'bukan', 'jangan', 'tak', 'kurang',
    # Kata negasi tambahan (ekspansi) - dibersihkan dari duplikat
    'nggak', 'gak', 'ga', 'gk', 'gaks', 'nggk', 'gabisa',
    'engga', 'enggak', 'kagak', 'nda', 'ndak', 'nd',
    'napa', 'engap', 'tiada', 'stop'
]

# Kamus kata kunci untuk deteksi aspek.
ASPECT_KEYWORDS_PATH = BASE_DIR / "data" / "aspect_keywords.json"
with ASPECT_KEYWORDS_PATH.open(encoding="utf-8") as aspect_keywords_file:
    ASPECT_KEYWORDS = json.load(aspect_keywords_file)

# ============================================================================
# PARAMETER LEXICON
# ============================================================================
# CATATAN (Tahap 6): judul bagian ini sebelumnya "PARAMETER LEXICON (InSet)",
# padahal file yang ditunjuk adalah lexicon v1 yang SUDAH TIDAK lagi dipakai
# jalur mana pun. Jalur canonical: shared/labeling.py -> get_active_lexicon()
# -> DEFAULT_LEXICON_VERSION ("v2").

# Path file lexicon v1 (historis).
# File yang tersedia: lexicon_positive.tsv, lexicon_negative.tsv
POSITIVE_LEXICON_PATH = DATASET_DIR / "lexicon_positive.tsv"
NEGATIVE_LEXICON_PATH = DATASET_DIR / "lexicon_negative.tsv"

# Lexicon netral NONAKTIF (lihat docs/audit_lexicon.md). Constant ini
# sengaja tidak menunjuk file yang ada: netral ditentukan oleh ambang skor,
# bukan oleh kamus kata. Dihapus pada Tahap 6 karena sebelumnya menunjuk
# file yang tidak pernah ada (neutral_final.tsv).
NEUTRAL_LEXICON_DISABLED = True

# Lexicon versi 2 (dibangun oleh scripts/build_lexicon_v2.py).
# Berisi HANYA entri satu token, karena classifier Stage 3 berbasis token
# tidak dapat mencocokkan entri frasa.
#
# Tersedia tiga varian:
#   v2               -> REKOMENDASI: v1 + R3 (kembalikan kata sentimen) + R6
#                       (pisahkan entri frasa). Tidak menghapus kata topik dan
#                       tidak menambah bentuk stem massal.
#   v2_brief_lengkap -> mengikuti brief secara literal: R1 (hapus kata topik)
#                       + R3 + R4 (bentuk stem) + R6.
#   v2_tanpa_hapus_topic -> R3 + R4 + R6 (tanpa R1).
LEXICON_V2_DIR = DATASET_DIR / "v2"
POSITIVE_LEXICON_V2_PATH = LEXICON_V2_DIR / "lexicon_positive.tsv"
NEGATIVE_LEXICON_V2_PATH = LEXICON_V2_DIR / "lexicon_negative.tsv"

LEXICON_V2_BRIEF_DIR = DATASET_DIR / "v2_brief_lengkap"
POSITIVE_LEXICON_V2_BRIEF_PATH = (
    LEXICON_V2_BRIEF_DIR / "lexicon_positive.tsv"
)
NEGATIVE_LEXICON_V2_BRIEF_PATH = (
    LEXICON_V2_BRIEF_DIR / "lexicon_negative.tsv"
)

LEXICON_V2_NO_TOPIC_DIR = DATASET_DIR / "v2_tanpa_hapus_topic"
POSITIVE_LEXICON_V2_NO_TOPIC_PATH = (
    LEXICON_V2_NO_TOPIC_DIR / "lexicon_positive.tsv"
)
NEGATIVE_LEXICON_V2_NO_TOPIC_PATH = (
    LEXICON_V2_NO_TOPIC_DIR / "lexicon_negative.tsv"
)

# Entri frasa lexicon v2. Disimpan terpisah, TIDAK dipakai Stage 3.
PHRASES_POSITIVE_V2_PATH = LEXICON_V2_DIR / "phrases_positive.tsv"
PHRASES_NEGATIVE_V2_PATH = LEXICON_V2_DIR / "phrases_negative.tsv"

# Versi lexicon yang dipakai pipeline.
# "v1" = data/lexicon/lexicon_positive.tsv + lexicon_negative.tsv (lama)
# "v2" = data/lexicon/v2/... (baru, hasil perbaikan Tahap 1: R3 + R6)
# Nilai ini dapat dioverride lewat environment variable
# WAYGROUND_LEXICON_VERSION tanpa mengubah kode.
#
# Kepemilikan project (27 Okt 2025): default diganti dari "v1" ke "v2".
# Dasar: akurasi 0,8194 vs 0,8021 dan F1-macro 0,6137 vs 0,6020 pada
# 288 label pakar. Detail di docs/audit_lexicon.md.
DEFAULT_LEXICON_VERSION = "v2"
LEXICON_VERSION_ENV_VAR = "WAYGROUND_LEXICON_VERSION"


def resolve_lexicon_version() -> str:
    """
    Versi lexicon yang sedang aktif.

    Satu-satunya tempat resolusi versi lexicon didefinisikan, supaya
    shared/lexicon.py dan shared/labeling.py tidak membuat dua implementasi
    yang bisa berbeda.

    Urutan:
        1. environment variable WAYGROUND_LEXICON_VERSION
        2. DEFAULT_LEXICON_VERSION ("v2")
    """
    return (
        os.environ.get(LEXICON_VERSION_ENV_VAR)
        or DEFAULT_LEXICON_VERSION
    ).strip().lower()

# Path lexicon PAKAR (kosongkan jika belum ada hasil validasi pakar)
POSITIVE_PAKAR_PATH = DATASET_DIR / "positive_PAKAR.tsv"
NEGATIVE_PAKAR_PATH = DATASET_DIR / "negative_PAKAR.tsv"

# Pilihan lexicon aktif.
#
# CATATAN (D4 §3.5): nilai lama 'INSET' berasal dari state sebelum lexicon v2
# tersedia dan SUDAH TIDAK dipakai jalur mana pun. Pelabelan canonical
# berjalan lewat shared/labeling.py -> get_active_lexicon() ->
# DEFAULT_LEXICON_VERSION ("v2").
#
# Nilai di sini hanya penanda untuk pembacaan kode dan TIDAK memengaruhi
# perilaku pelabelan. File lexicon PAKAR juga belum tersedia.
ACTIVE_LEXICON = 'v2'  # v1 | v2 | v2_brief_lengkap | v2_tanpa_hapus_topic

# ============================================================================
# PARAMETER NEUTRAL THRESHOLD
# ============================================================================
# PERBAIKAN: NEUTRAL_THRESHOLD diimpor dari shared/final_config.py
# (DEPLOYMENT_NEUTRAL_THRESHOLD = 1.0, hasil Experiment 2 pada development).
# Aturan: score > threshold -> Positif, score < -threshold -> Negatif,
# selain itu -> Netral.

# ============================================================================
# PARAMETER STOPWORD
# ============================================================================

# Stopword dasar (Sastrawi) + tambahan
# PERBAIKAN v2: Hapus kata yang memiliki makna sentimen dari stopword

# Stopword tambahan untuk penelitian ini
# CATATAN: Hapus kata sentimen dari daftar ini!
# - 'cocok' = sentimen positif untuk "suitable"
# - 'buat' = preposisi penting untuk konteks
# - 'tolong' = kata request yang penting
ADDITIONAL_STOPWORDS = [
    # Stopword dasar (kata-kata umum tanpa makna sentimen)
    'yang', 'dan', 'di', 'ke', 'dari', 'dengan', 'pada', 'adalah',
    'ini', 'itu', 'akan', 'sudah', 'ada', 'saya', 'kamu', 'anda', 'nya',
    'tersebut', 'bisa', 'hanya', 'juga', 'serta', 'atau', 'agar', 'supaya',
    'jika', 'bila', 'maka', 'sebab', 'oleh', 'sehingga',
    'tetapi', 'namun', 'meski', 'walau', 'ingin', 'hendak',
    'wajib', 'patut', 'layak',
    # Stopword tambahan (slang/chat - tidak punya makna sentimen)
    'lagi', 'dong', 'sih', 'deh', 'doang', 'dorang', 'tuh', 'mah', 'atuh', 'adja',
    'kalo', 'gitu', 'begitu', 'gtu',
    'dgn', 'gak', 'nggak', 'engga', 'si', 'gimana', 'pgn', 'pingin', 'sm',
    'tp', 'krn', 'jd', 'utk', 'pd', 'dlm', 'smua',
    'bbrp', 'beberapa', 'lm', 'lama', 'blom', 'blm', 'sdh', 'udah',
    'bkn', 'bukan', 'bsa', 'bs', 'jg', 'byk', 'brp', 'spt', 'sprti', 'nanti',
    'skrng', 'skr', 'sini', 'sitrs', 'sana', 'sndr', 'sendiri',
    'ngak', 'gt', 'dmna', 'kmn', 'kemana', 'napa', 'tpi', 'bbr', 'dll'
]

# ============================================================================
# KATA SENTIMEN YANG TIDAK BOLEH DIHAPUS
# ============================================================================
# Kata-kata ini memiliki makna sentimen dan TIDAK boleh dihapus
SENTIMENT_PROTECTED_WORDS = {
    # Preposisi dengan konteks
    'untuk', 'buat',  # "cocok untuk", "bagus buat"
    # Kata request
    'tolong', 'mohon', 'diminta', 'seharusnya', 'harus',
    # Kata sentimen/kontekstual
    'cocok', 'pas', 'tepat', 'lancar', 'stabil',
    # Kata negatif yang konteksnya penting
    'parah', 'masih', 'udah', 'belum', 'mau',
    # Kata intensitas
    'banget', 'bgt', 'bangett', 'bangettt', 'sangat', 'sekali', 'abis',
}

# ============================================================================
# PARAMETER ASPECT DETECTION
# ============================================================================

# Pola regex untuk memecah kalimat
SENTENCE_DELIMITER = r'[.!?]+'

# ============================================================================
# PARAMETER VISUALISASI
# ============================================================================

# Warna untuk kelas sentimen
SENTIMENT_COLORS = {
    'Positif': '#22c55e',   # hijau
    'Netral': '#3b82f6',    # biru
    'Negatif': '#ef4444'    # merah
}

# Ukuran default figure
FIGURE_SIZE = (10, 6)

# ============================================================================
# PARAMETER DATASET
# ============================================================================

# Kolom yang diharapkan dari hasil scraping
EXPECTED_COLUMNS = [
    'reviewId', 'userName', 'content', 'rating', 'thumbsUpCount',
    'reviewCreatedVersion', 'at', 'appVersion'
]

# ============================================================================
# KONTRAK KOLOM DATA AKHIR
# ============================================================================
# Hasil akhir tahap scraping hanya memuat empat kolom data yang relevan untuk
# tahap preprocessing, ditambah satu kunci teknis (reviewId).
#
#   at         -> tanggal ulasan
#   content    -> konten/ulasan (SATU-SATUNYA kolom yang diproses)
#   rating     -> rating bintang 1..5
#   appVersion -> versi aplikasi
#
# Kolomika seperti userName, thumbsUpCount, replyContent, repliedAt, platform,
# packageName, language_source, scraped_at, dan kolom audit bahasa TIDAK
# diteruskan ke tahap berikutnya karena tidak dipakai di mana pun.
#
# reviewId dipertahankan sebagai kunci teknis untuk penelusuran data dan
# sebagai kunci pencocokan sekunder saat integrasi label pakar. reviewId
# BUKAN kolom data yang dianalisis.
FINAL_DATA_KEY_COLUMN = "reviewId"

FINAL_DATA_COLUMNS = [
    "at",          # tanggal
    "content",     # konten / ulasan
    "rating",      # rating
    "appVersion",  # versi aplikasi
]

# Nama kolom sumber pada data mentah scraping.
RAW_DATE_COLUMN = "at"
RAW_CONTENT_COLUMN = "content"
RAW_RATING_COLUMN = "rating"
RAW_APP_VERSION_COLUMN = "appVersion"

# Label ramah-manusia untuk ditampilkan di laporan dan web.
FINAL_DATA_COLUMN_LABELS = {
    "at": "Tanggal",
    "content": "Konten Ulasan",
    "rating": "Rating",
    "appVersion": "Versi Aplikasi",
}

# Kolom yang boleh dibawa ke tahap berikutnya (kunci teknis + 4 kolom data).
FINAL_OUTPUT_COLUMNS = [FINAL_DATA_KEY_COLUMN] + FINAL_DATA_COLUMNS

# ============================================================================
# KELAS SENTIMEN
# ============================================================================

SENTIMENT_CLASSES = ['Positif', 'Netral', 'Negatif']

SENTIMENT_DEFINITIONS = {
    'Positif': 'Menyatakan kepuasan, pujian, manfaat yang dirasakan, atau dukungan',
    'Netral': 'Bersifat deskriptif, informatif, pertanyaan, atau saran tanpa muatan emosi',
    'Negatif': 'Menyatakan keluhan, ketidakpuasan, kekecewaan, atau kendala'
}

# ============================================================================
# ASPEK PEMBELAJARAN
# ============================================================================

ASPECTS = list(ASPECT_KEYWORDS.keys())

ASPECT_DEFINITIONS = {
    'Kemudahan Akses': 'Penilaian terhadap kemudahan akses dan kecepatan pemuatan',
    'Gamifikasi': 'Penilaian terhadap unsur permainan sebagai pendorong motivasi belajar',
    'Materi/Konten': 'Penilaian terhadap kualitas dan kelengkapan konten pembelajaran',
    'Performa Teknis': 'Penilaian terhadap stabilitas dan keandalan aplikasi'
}

ASPECT_MAPPING = {
    'Kemudahan Akses': 'Aksesibilitas dan kesiapan teknologi',
    'Gamifikasi': 'Motivasi dan keterlibatan peserta didik',
    'Materi/Konten': 'Kualitas materi dan penyampaian',
    'Performa Teknis': 'Stabilitas media pembelajaran'
}

# ============================================================================
# FUNGSI HELPER
# ============================================================================

def get_all_stopwords():
    """
    Mengambil semua stopword (Sastrawi + tambahan).
    Ini akan di-load saat runtime di preprocessing.py
    """
    return ADDITIONAL_STOPWORDS

def get_aspect_keywords_for_display():
    """Format kata kunci aspek untuk tampilan."""
    result = {}
    for aspect, keywords in ASPECT_KEYWORDS.items():
        result[aspect] = {
            'keywords': keywords,
            'definition': ASPECT_DEFINITIONS.get(aspect, ''),
            'mapping': ASPECT_MAPPING.get(aspect, '')
        }
    return result

# ============================================================================
# PREPROCESSING STAGES (Untuk dokumentasi UI)
# ============================================================================
PREPROCESSING_STAGES = [
    "Cleaning",
    "Case Folding",
    "Tokenizing",
    "Normalisasi",
    "Negation Handling",
    "Stopword Removal",
    "Stemming"
]

# ============================================================================
# GRID SEARCH PARAMETERS (Untuk referensi)
# ============================================================================
GRID_SEARCH_C = [0.01, 0.1, 0.5, 1.0, 10, 100]  # Updated with 100 for comprehensive search
GRID_SEARCH_GAMMA = ['scale', 'auto', 0.01, 0.1, 1]
