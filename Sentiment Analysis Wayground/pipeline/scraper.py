"""
Google Play Review Scraper - Wayground / Quizizz
Alur:
1. Scrape review Google Play dengan locale Indonesia.
2. Validasi bahasa dan pertahankan review berbahasa Indonesia.
3. Simpan seluruh review Indonesia hasil scraping -> 01_semua_data.csv
4. Filter periode penelitian -> 02_data_akhir_dedup.csv
5. Hapus review dengan konten yang sama -> 02_data_akhir.csv
6. Simpan audit scraping dan audit proses.
Periode penelitian:
01-Januari-2024 s.d. 30-Juni-2026, inklusif.
Catatan:
- Parameter lang="id" dan country="id" digunakan saat scraping.
- Validasi bahasa tambahan dilakukan setelah review diterima.
- Filter tanggal dilakukan SETELAH seluruh data hasil scraping dikumpulkan.
- Deduplicate menggunakan reviewId dan konten ulasan.
"""
from __future__ import annotations
import argparse
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import pandas as pd
from tqdm import tqdm
# ============================================================================
# PROJECT PATH
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from shared.research_config import RESEARCH_START, RESEARCH_END_EXCLUSIVE
from shared.data_contract import normalize_review_columns
from shared.constants import FINAL_OUTPUT_COLUMNS
# ============================================================================
# OUTPUT
# ============================================================================
OUTPUT_RAW = DATA_DIR / "01_semua_data.csv"
OUTPUT_PERIOD = DATA_DIR / "02_data_akhir_dedup.csv"
OUTPUT_FINAL = DATA_DIR / "02_data_akhir.csv"
OUTPUT_AUDIT = DATA_DIR / "03_data_audit_scraping.csv"
OUTPUT_PASS_AUDIT = DATA_DIR / "03_data_audit_scraping_passes.csv"
# ============================================================================
# RESEARCH CONFIGURATION
# ============================================================================
APP_NAME = "Wayground (Quizizz)"
PACKAGE_NAME = "com.quizizz_mobile"
LANGUAGE = "id"
COUNTRY = "id"
# 01 Januari 2024 sampai 30 Juni 2026, inklusif.
START_DATE = pd.Timestamp(RESEARCH_START)
END_DATE_EXCLUSIVE = pd.Timestamp(RESEARCH_END_EXCLUSIVE)
END_DATE_INCLUSIVE = END_DATE_EXCLUSIVE - pd.Timedelta(seconds=1)
BATCH_SIZE = 200
# None = teruskan pagination sampai continuation_token habis.
# Nilai integer hanya sebagai safety limit jika memang diperlukan.
MAX_BATCHES_PER_PASS: Optional[int] = None
REQUEST_DELAY_SECONDS = 1.0
MAX_RETRIES = 5
RETRY_BASE_DELAY_SECONDS = 2.0
ENABLE_RATING_FALLBACK = True
RATINGS = (1, 2, 3, 4, 5)
LANGUAGE_MIN_WORDS = 5
ENGLISH_REJECT_PROBABILITY = 0.90
# Jika aktif, final tidak dianggap valid apabila scraper belum mencapai
# tanggal awal penelitian.
REQUIRE_START_DATE_COVERAGE = True
# ============================================================================
# DEPENDENCIES
# ============================================================================
try:
    from google_play_scraper import Sort, reviews
except ImportError as exc:
    print("ERROR: google-play-scraper belum terpasang.")
    print("Install dengan:")
    print("    pip install google-play-scraper")
    print(f"\nDetail: {exc}")
    raise SystemExit(1)
try:
    # pyrefly: ignore[missing-import]
    from langdetect import detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    detect_langs = None
    LANGDETECT_AVAILABLE = False
# ============================================================================
# INDONESIAN LANGUAGE VALIDATION
# ============================================================================
INDONESIAN_HINTS = {
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan",
    "tidak", "bisa", "sangat", "bagus", "aplikasi", "belajar",
    "saya", "aku", "ini", "itu", "nya", "karena", "kalau",
    "sudah", "belum", "lagi", "ada", "lebih", "kurang", "tolong",
    "terima", "kasih", "main", "game", "kuis", "soal", "guru",
    "siswa", "sekolah", "aja", "juga", "buat", "dalam",
    "pakai", "pake", "bikin", "mau", "suka", "jelek", "error",
    "login", "masuk", "keluar", "akun", "kelas", "tugas",
}
ENGLISH_HINTS = {
    "the", "this", "that", "is", "are", "was", "were", "and",
    "or", "with", "for", "from", "very", "good", "great", "bad",
    "app", "application", "please", "love", "like", "hate", "easy",
    "hard", "helpful", "useful", "works", "working", "download",
    "update", "quiz", "teacher", "student", "class", "login",
}
# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def normalize_datetime(value: Any) -> Optional[pd.Timestamp]:
    """Konversi nilai tanggal menjadi Timestamp tanpa timezone."""
    if value is None or pd.isna(value):
        return None
    try:
        timestamp = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(timestamp):
            return None
        return timestamp.tz_convert(None)
    except Exception:
        return None
def normalize_content(text: Any) -> str:
    """Normalisasi whitespace pada konten ulasan."""
    if text is None:
        return ""
    return " ".join(
        str(text)
        .replace("\r", " ")
        .replace("\n", " ")
        .split()
    )
def content_key(text: Any) -> str:
    """
    Membuat kunci konten untuk mendeteksi ulasan yang sama.
    Casefold digunakan agar perbedaan huruf besar/kecil tidak dianggap
    sebagai ulasan yang berbeda.
    """
    return (
        normalize_content(text)
        .casefold()
        .replace("\u200b", "")
        .strip()
    )
def stable_review_id(review: dict[str, Any]) -> str:
    """
    Mengambil reviewId asli.
    Jika reviewId tidak tersedia, digunakan id alternatif atau hash
    sebagai fallback.
    """
    review_id = str(review.get("reviewId") or "").strip()
    if review_id:
        return review_id
    alternative_id = str(review.get("id") or "").strip()
    if alternative_id:
        return alternative_id
    payload = "|".join(
        [
            str(review.get("userName") or ""),
            str(review.get("content") or ""),
            str(review.get("at") or ""),
            str(review.get("score", review.get("rating", "")) or ""),
        ]
    )
    digest = hashlib.sha256(
        payload.encode("utf-8", errors="ignore")
    ).hexdigest()
    return f"fallback_{digest}"
# ============================================================================
# LANGUAGE DETECTION
# ============================================================================
def detect_indonesian(text: str) -> tuple[bool, str, float]:
    """
    Menentukan apakah review layak dianggap berbahasa Indonesia.
    Review campuran tidak langsung dibuang kecuali cukup kuat terdeteksi
    sebagai bahasa Inggris.
    """
    text = normalize_content(text).lower()
    if not text:
        return False, "empty", 0.0
    words = text.split()
    # Deteksi probabilistik untuk teks yang cukup panjang.
    if LANGDETECT_AVAILABLE and len(words) >= LANGUAGE_MIN_WORDS:
        try:
            results = detect_langs(text)
            if results:
                top = results[0]
                language = str(top.lang)
                probability = float(top.prob)
                if language == "id":
                    return True, "id", probability
                if (
                    language == "en"
                    and probability >= ENGLISH_REJECT_PROBABILITY
                ):
                    return False, "en", probability
        except Exception:
            pass
    # Fallback berbasis kosakata.
    word_set = set(words)
    id_score = len(word_set & INDONESIAN_HINTS)
    en_score = len(word_set & ENGLISH_HINTS)
    if en_score >= 2 and en_score > id_score:
        return False, "en_heuristic", 0.80
    return True, "id_or_mixed", 0.0
# ============================================================================
# GOOGLE PLAY API
# ============================================================================
def fetch_batch(
    continuation_token=None,
    filter_score_with: Optional[int] = None,
):
    """Mengambil satu batch review dengan retry dan exponential backoff."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = {
                "lang": LANGUAGE,
                "country": COUNTRY,
                "sort": Sort.NEWEST,
                "count": BATCH_SIZE,
                "continuation_token": continuation_token,
            }
            if filter_score_with is not None:
                kwargs["filter_score_with"] = filter_score_with
            result, token = reviews(
                PACKAGE_NAME,
                **kwargs,
            )
            return result or [], token
        except Exception as exc:
            last_error = exc
            print(
                f"    [RETRY {attempt}/{MAX_RETRIES}] "
                f"{type(exc).__name__}: {exc}"
            )
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                time.sleep(delay)
    raise RuntimeError(
        f"Gagal mengambil batch setelah {MAX_RETRIES} percobaan: "
        f"{last_error}"
    )
def collect_pass(
    filter_score_with: Optional[int] = None,
    pass_name: str = "GLOBAL",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Menjalankan satu pass pagination Google Play.
    Pagination dihentikan hanya jika:
    - batch kosong,
    - continuation token habis, atau
    - safety limit batch tercapai jika MAX_BATCHES_PER_PASS diaktifkan.
    PENTING:
    Filter tanggal TIDAK dilakukan di dalam pagination. Seluruh review
    yang tersedia dikumpulkan terlebih dahulu, kemudian validasi bahasa
    Indonesia dan filter periode dilakukan pada tahap berikutnya.
    """
    collected: list[dict[str, Any]] = []
    seen: set[str] = set()
    token = None
    batch_count = 0
    stop_reason = "max_batches"
    progress = tqdm(
        total=MAX_BATCHES_PER_PASS,
        desc=f"{pass_name:>12}",
        unit=" batch",
        leave=True,
    )
    try:
        while batch_count < MAX_BATCHES_PER_PASS:
            batch_count += 1
            batch, next_token = fetch_batch(
                continuation_token=token,
                filter_score_with=filter_score_with,
            )
            if not batch:
                stop_reason = "empty_batch"
                break
            batch_dates = []
            for review in batch:
                review_id = stable_review_id(review)
                if review_id in seen:
                    continue
                seen.add(review_id)
                collected.append(review)
                review_date = normalize_datetime(review.get("at"))
                if review_date is not None:
                    batch_dates.append(review_date)
            progress.update(1)
            progress.set_postfix(
                unique=f"{len(collected):,}",
                oldest=(
                    min(batch_dates).strftime("%Y-%m-%d")
                    if batch_dates
                    else "N/A"
                ),
            )
            # JANGAN menghentikan pagination berdasarkan tanggal.
            # Seluruh data dikumpulkan terlebih dahulu. Filter tanggal
            # dilakukan setelah validasi bahasa Indonesia selesai.
            if not next_token:
                stop_reason = "no_continuation_token"
                break
            token = next_token
            time.sleep(REQUEST_DELAY_SECONDS)
    finally:
        progress.close()
    dates = [
        normalize_datetime(review.get("at"))
        for review in collected
    ]
    dates = [date for date in dates if date is not None]
    metadata = {
        "pass_name": pass_name,
        "filter_score_with": filter_score_with,
        "batches": batch_count,
        "reviews": len(collected),
        "stop_reason": stop_reason,
        "oldest_date": min(dates).strftime("%Y-%m-%d") if dates else None,
        "newest_date": max(dates).strftime("%Y-%m-%d") if dates else None,
    }
    return collected, metadata
# ============================================================================
# REVIEW MERGE
# ============================================================================
def merge_reviews(
    existing: dict[str, dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> int:
    """Menggabungkan review berdasarkan reviewId."""
    added = 0
    for review in incoming:
        review_id = stable_review_id(review)
        if review_id not in existing:
            existing[review_id] = review
            added += 1
    return added
# ============================================================================
# DATAFRAME CONVERSION
# ============================================================================
def reviews_to_dataframe(
    review_map: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Mengubah hasil scraping menjadi DataFrame."""
    rows = []
    scraped_at = datetime.now(timezone.utc).replace(tzinfo=None)
    for review_id, review in review_map.items():
        score = review.get(
            "score",
            review.get("rating", 0),
        )
        rows.append(
            {
                "reviewId": review_id,
                "userName": review.get(
                    "userName",
                    review.get("reviewerName", ""),
                ),
                "content": normalize_content(
                    review.get("content", "")
                ),
                "rating": score,
                "thumbsUpCount": review.get(
                    "thumbsUpCount",
                    0,
                ),
                "at": normalize_datetime(review.get("at")),
                "appVersion": review.get(
                    "appVersion",
                    review.get("reviewCreatedVersion", ""),
                ),
                "replyContent": review.get(
                    "replyContent",
                    review.get("replyText", ""),
                ),
                "repliedAt": normalize_datetime(
                    review.get("repliedAt")
                ),
                "platform": "Google Play Store",
                "packageName": PACKAGE_NAME,
                "language_source": LANGUAGE,
                "scraped_at": scraped_at,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["at"] = pd.to_datetime(
        df["at"],
        errors="coerce",
    )
    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce",
    )
    # Rating Google Play harus 1-5.
    df = df[df["rating"].isin([1, 2, 3, 4, 5])].copy()
    return df.reset_index(drop=True)
# ============================================================================
# LANGUAGE FILTER
# ============================================================================
def filter_indonesian_reviews(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mempertahankan review yang terdeteksi sebagai bahasa Indonesia.
    Audit bahasa dikembalikan terpisah.
    """
    if df.empty:
        return df.copy(), pd.DataFrame()
    audit_rows = []
    for index, content in df["content"].items():
        keep, detected, confidence = detect_indonesian(content)
        audit_rows.append(
            {
                "_index": index,
                "language_detected": detected,
                "language_confidence": confidence,
                "language_keep": keep,
            }
        )
    audit = pd.DataFrame(audit_rows).set_index("_index")
    # Raw exports already contain the previous language audit.
    result = df.drop(columns=audit.columns, errors="ignore").join(audit, how="left")
    valid = result[result["language_keep"]].copy()
    invalid = result[~result["language_keep"]].copy()
    return valid, invalid
# ============================================================================
# PERIOD FILTER
# ============================================================================
def filter_research_period(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Filter tanggal penelitian.
    Periode:
    RESEARCH_START <= tanggal < RESEARCH_END_EXCLUSIVE
    """
    if df.empty:
        return df.copy()
    dates = pd.to_datetime(
        df["at"],
        errors="coerce",
    )
    mask = (
        dates.notna()
        & (dates >= START_DATE)
        & (dates < END_DATE_EXCLUSIVE)
    )
    return df.loc[mask].copy()
# ============================================================================
# DUPLICATE REMOVAL
# ============================================================================
def remove_duplicate_reviews(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Menghapus:
    1. reviewId yang sama
    2. konten/ulasan yang sama
    Duplicate content dibandingkan setelah normalisasi whitespace dan
    casefold, sehingga perbedaan huruf besar/kecil tidak dianggap sebagai
    ulasan berbeda.
    """
    if df.empty:
        return df.copy(), {
            "duplicate_review_id_removed": 0,
            "duplicate_content_removed": 0,
            "empty_content_removed": 0,
        }
    result = df.copy()
    before = len(result)
    # Hapus konten kosong.
    content = result["content"].fillna("").map(normalize_content)
    empty_mask = content.eq("")
    empty_removed = int(empty_mask.sum())
    result = result.loc[~empty_mask].copy()
    # Hapus duplicate reviewId.
    before_review_id = len(result)
    result = result.drop_duplicates(
        subset=["reviewId"],
        keep="first",
    ).copy()
    duplicate_review_id_removed = (
        before_review_id - len(result)
    )
    # Hapus duplicate content.
    result["_content_key"] = result["content"].map(content_key)
    before_content = len(result)
    result = result.drop_duplicates(
        subset=["_content_key"],
        keep="first",
    ).copy()
    duplicate_content_removed = (
        before_content - len(result)
    )
    result.drop(
        columns=["_content_key"],
        inplace=True,
    )
    result.reset_index(drop=True, inplace=True)
    # Validasi internal.
    assert len(result) <= before
    return result, {
        "duplicate_review_id_removed": duplicate_review_id_removed,
        "duplicate_content_removed": duplicate_content_removed,
        "empty_content_removed": empty_removed,
    }
# ============================================================================
# FINAL COLUMN SELECTION
# ============================================================================
def select_final_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Menyiapkan struktur data akhir penelitian.
    reviewId tetap disimpan sebagai kunci teknis.
    """
    if df.empty:
        return pd.DataFrame(columns=FINAL_OUTPUT_COLUMNS)
    result = normalize_review_columns(df)
    result["at"] = result["at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return result[FINAL_OUTPUT_COLUMNS].reset_index(drop=True)
# ============================================================================
# AUDIT
# ============================================================================
def create_audit(
    df_all: pd.DataFrame,
    df_period: pd.DataFrame,
    df_language: pd.DataFrame,
    df_final: pd.DataFrame,
) -> pd.DataFrame:
    """Membuat audit status setiap review hasil scraping."""
    final_ids = set(df_final["reviewId"]) if not df_final.empty else set()
    period_ids = (
        set(df_period["reviewId"])
        if not df_period.empty
        else set()
    )
    language_ids = (
        set(df_language["reviewId"])
        if not df_language.empty
        else set()
    )
    rows = []
    for _, row in df_all.iterrows():
        review_id = row["reviewId"]
        if review_id in final_ids:
            status = "KEEP_FINAL"
            reason = (
                "Dalam periode + bahasa Indonesia + "
                "lolos deduplicate"
            )
        elif review_id not in period_ids:
            status = "DROP_PERIOD"
            reason = "Di luar periode penelitian"
        elif review_id not in language_ids:
            status = "DROP_LANGUAGE"
            reason = "Tidak terdeteksi sebagai bahasa Indonesia"
        else:
            status = "DROP_DUPLICATE"
            reason = "Duplikat reviewId atau content"
        rows.append(
            {
                "reviewId": review_id,
                "at": row["at"],
                "rating": row["rating"],
                "content": row["content"],
                "appVersion": row["appVersion"],
                "status": status,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)
# ============================================================================
# STATISTICS
# ============================================================================
def print_stats(
    df: pd.DataFrame,
    title: str,
) -> None:
    """Menampilkan statistik dataset."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    if df.empty:
        print("Tidak ada data.")
        return
    print(f"Total review : {len(df):,}")
    if "at" in df.columns:
        dates = pd.to_datetime(
            df["at"],
            errors="coerce",
        ).dropna()
        if not dates.empty:
            print(
                "Rentang tanggal : "
                f"{dates.min().date()} s/d {dates.max().date()}"
            )
    if "content" in df.columns:
        words = (
            df["content"]
            .fillna("")
            .astype(str)
            .str.split()
            .str.len()
        )
        print(f"Rata-rata kata : {words.mean():.1f}")
        print(f"Median kata    : {words.median():.0f}")
    if "rating" in df.columns:
        print("\nDistribusi rating:")
        print(
            df["rating"]
            .value_counts()
            .sort_index()
            .to_string()
        )
    if "appVersion" in df.columns:
        print(
            "\nVersi aplikasi unik: "
            f"{df['appVersion'].replace('', pd.NA).nunique()}"
        )
# ============================================================================
# REBUILD FROM RAW
# ============================================================================
def rebuild_from_raw() -> int:
    """
    Membuat ulang file turunan dari 01_semua_data.csv tanpa jaringan.
    Urutan:
    01_semua_data
        -> filter periode
        -> 02_data_akhir_dedup
        -> remove duplicate content
        -> 02_data_akhir
    """
    if not OUTPUT_RAW.exists():
        print(
            f"ERROR: File raw tidak ditemukan:\n"
            f"{OUTPUT_RAW}"
        )
        return 1
    print("=" * 80)
    print("REBUILD DATA TURUNAN DARI RAW")
    print("=" * 80)
    df_all = pd.read_csv(
        OUTPUT_RAW,
        encoding="utf-8-sig",
    )
    df_all["at"] = pd.to_datetime(
        df_all["at"],
        errors="coerce",
    )
    # Pastikan hanya review Indonesia yang digunakan.
    df_language, _ = filter_indonesian_reviews(df_all)
    # Filter periode.
    df_period = filter_research_period(df_language)
    # Simpan hasil filter periode SEBELUM dedup.
    df_period.to_csv(
        OUTPUT_PERIOD,
        index=False,
        encoding="utf-8-sig",
    )
    # Remove duplicate.
    df_final_raw, duplicate_audit = remove_duplicate_reviews(
        df_period
    )
    df_final = select_final_columns(df_final_raw)
    df_final.to_csv(
        OUTPUT_FINAL,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Raw                  : {len(df_all):,}")
    print(f"Bahasa Indonesia     : {len(df_language):,}")
    print(f"Filter periode       : {len(df_period):,}")
    print(f"Setelah dedup        : {len(df_final):,}")
    print(
        "Duplicate reviewId   : "
        f"{duplicate_audit['duplicate_review_id_removed']:,}"
    )
    print(
        "Duplicate content    : "
        f"{duplicate_audit['duplicate_content_removed']:,}"
    )
    print()
    print(f"Output periode : {OUTPUT_PERIOD}")
    print(f"Output final   : {OUTPUT_FINAL}")
    return 0
# ============================================================================
# MAIN SCRAPING PIPELINE
# ============================================================================
def main() -> int:
    """Menjalankan seluruh pipeline scraping."""
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    print("=" * 80)
    print("WAYGROUND / QUIZIZZ GOOGLE PLAY REVIEW SCRAPER")
    print("=" * 80)
    print(f"App              : {APP_NAME}")
    print(f"Package          : {PACKAGE_NAME}")
    print("Source           : Google Play Store")
    print(f"Language         : {LANGUAGE}")
    print(f"Country          : {COUNTRY}")
    print(
        "Research period  : "
        f"{START_DATE.date()} s/d {END_DATE_INCLUSIVE.date()}"
    )
    print(f"Batch size       : {BATCH_SIZE}")
    print(f"Max batch/pass   : {MAX_BATCHES_PER_PASS}")
    print(
        "Langdetect       : "
        f"{'AKTIF' if LANGDETECT_AVAILABLE else 'TIDAK AKTIF'}"
    )
    print(
        "Rating fallback  : "
        f"{'AKTIF' if ENABLE_RATING_FALLBACK else 'NONAKTIF'}"
    )
    print("=" * 80)
    # ------------------------------------------------------------------------
    # PASS 1 - GLOBAL
    # ------------------------------------------------------------------------
    print("\n[PASS 1] SCRAPING GLOBAL")
    global_reviews, global_meta = collect_pass(
        filter_score_with=None,
        pass_name="GLOBAL",
    )
    review_map: dict[str, dict[str, Any]] = {}
    merge_reviews(
        review_map,
        global_reviews,
    )
    pass_metadata = [global_meta]
    print(
        f"\nReview global terkumpul : "
        f"{len(global_reviews):,}"
    )
    print(
        f"Stop reason             : "
        f"{global_meta['stop_reason']}"
    )
    print(
        f"Tanggal tertua          : "
        f"{global_meta['oldest_date']}"
    )
    # ------------------------------------------------------------------------
    # PASS 2 - RATING FALLBACK
    # ------------------------------------------------------------------------
    global_dates = [
        normalize_datetime(review.get("at"))
        for review in global_reviews
    ]
    global_dates = [
        date for date in global_dates if date is not None
    ]
    global_oldest = min(global_dates) if global_dates else None
    need_fallback = (
        ENABLE_RATING_FALLBACK
        and (
            global_oldest is None
            or global_oldest > START_DATE
        )
    )
    if need_fallback:
        print(
            "\n[PASS 2] FALLBACK PER-RATING "
            "(coverage belum mencapai 01/01/2024)"
        )
        for rating in RATINGS:
            print(f"\n[RATING {rating}]")
            rating_reviews, rating_meta = collect_pass(
                filter_score_with=rating,
                pass_name=f"RATING {rating}",
            )
            added = merge_reviews(
                review_map,
                rating_reviews,
            )
            rating_meta["new_unique_added"] = added
            pass_metadata.append(rating_meta)
            print(
                f"    Diperoleh      : "
                f"{len(rating_reviews):,}"
            )
            print(
                f"    Unique baru    : "
                f"{added:,}"
            )
            print(
                f"    Total gabungan : "
                f"{len(review_map):,}"
            )
    # ------------------------------------------------------------------------
    # BUILD RAW DATA
    # ------------------------------------------------------------------------
    df_scraped = reviews_to_dataframe(review_map)
    if df_scraped.empty:
        print("\nERROR: Tidak ada review yang berhasil dikumpulkan.")
        return 1
    print_stats(
        df_scraped,
        "DATA HASIL SCRAPING",
    )
    # ------------------------------------------------------------------------
    # LANGUAGE FILTER
    # ------------------------------------------------------------------------
    print("\n[STAGE 1] VALIDASI BAHASA INDONESIA")
    df_indonesian, df_non_indonesian = filter_indonesian_reviews(
        df_scraped
    )
    print(
        f"Review Indonesia : {len(df_indonesian):,}"
    )
    print(
        f"Review non-Indonesia : "
        f"{len(df_non_indonesian):,}"
    )
    # 01_semua_data = seluruh hasil scraping yang sudah divalidasi
    # sebagai review berbahasa Indonesia.
    df_indonesian.to_csv(
        OUTPUT_RAW,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"\n[SAVED] {OUTPUT_RAW}")
    # ------------------------------------------------------------------------
    # COVERAGE VALIDATION
    # ------------------------------------------------------------------------
    raw_dates = pd.to_datetime(
        df_indonesian["at"],
        errors="coerce",
    ).dropna()
    raw_min = raw_dates.min() if not raw_dates.empty else None
    raw_max = raw_dates.max() if not raw_dates.empty else None
    print("\n" + "=" * 80)
    print("VALIDASI COVERAGE")
    print("=" * 80)
    print(
        f"Target tanggal minimum : "
        f"{START_DATE.date()}"
    )
    print(
        f"Tanggal minimum raw   : "
        f"{raw_min.date() if raw_min is not None else 'N/A'}"
    )
    print(
        f"Tanggal maksimum raw  : "
        f"{raw_max.date() if raw_max is not None else 'N/A'}"
    )
    coverage_ok = (
        raw_min is not None
        and raw_min <= START_DATE
    )
    print(
        "Coverage status        : "
        f"{'OK' if coverage_ok else 'TIDAK LENGKAP'}"
    )
    # ------------------------------------------------------------------------
    # FILTER PERIOD
    # ------------------------------------------------------------------------
    print("\n[STAGE 2] FILTER PERIODE PENELITIAN")
    df_period = filter_research_period(
        df_indonesian
    )
    print_stats(
        df_period,
        "DATA SETELAH FILTER PERIODE",
    )
    # 02_data_akhir_dedup = hasil filter periode, BELUM deduplicate.
    df_period.to_csv(
        OUTPUT_PERIOD,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"\n[SAVED] {OUTPUT_PERIOD}")
    # ------------------------------------------------------------------------
    # REMOVE DUPLICATE
    # ------------------------------------------------------------------------
    print("\n[STAGE 3] REMOVE DUPLICATE")
    df_dedup, duplicate_audit = remove_duplicate_reviews(
        df_period
    )
    print(
        f"Duplicate reviewId : "
        f"{duplicate_audit['duplicate_review_id_removed']:,}"
    )
    print(
        f"Duplicate content  : "
        f"{duplicate_audit['duplicate_content_removed']:,}"
    )
    print(
        f"Konten kosong      : "
        f"{duplicate_audit['empty_content_removed']:,}"
    )
    print_stats(
        df_dedup,
        "DATA SETELAH REMOVE DUPLICATE",
    )
    # ------------------------------------------------------------------------
    # FINAL DATA
    # ------------------------------------------------------------------------
    print("\n[STAGE 4] MENYIAPKAN DATA AKHIR")
    df_final = select_final_columns(
        df_dedup
    )
    df_final.to_csv(
        OUTPUT_FINAL,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"\n[SAVED] {OUTPUT_FINAL}")
    # ------------------------------------------------------------------------
    # AUDIT
    # ------------------------------------------------------------------------
    audit = create_audit(
        df_all=df_scraped,
        df_period=df_period,
        df_language=df_indonesian,
        df_final=df_final,
    )
    audit.to_csv(
        OUTPUT_AUDIT,
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(pass_metadata).to_csv(
        OUTPUT_PASS_AUDIT,
        index=False,
        encoding="utf-8-sig",
    )
    # ------------------------------------------------------------------------
    # FAIL-SAFE
    # ------------------------------------------------------------------------
    if REQUIRE_START_DATE_COVERAGE and not coverage_ok:
        print("\n" + "=" * 80)
        print("WARNING: COVERAGE SCRAPING BELUM LENGKAP")
        print("=" * 80)
        print(
            "Scraper belum berhasil membuktikan data sampai "
            f"{START_DATE.date()}."
        )
        print(
            "01_semua_data.csv dan file audit tetap disimpan "
            "untuk diagnosis."
        )
        print(
            "02_data_akhir.csv TIDAK dianggap valid untuk penelitian."
        )
        print(
            "Jangan lanjut ke preprocessing/model sebelum coverage "
            "terpenuhi."
        )
        return 2
    # ------------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCRAPING SELESAI")
    print("=" * 80)
    print(
        f"01_semua_data.csv       : "
        f"{len(df_indonesian):,} review"
    )
    print(
        f"02_data_akhir_dedup.csv : "
        f"{len(df_period):,} review"
    )
    print(
        f"02_data_akhir.csv       : "
        f"{len(df_final):,} review"
    )
    print()
    print(f"Raw      : {OUTPUT_RAW}")
    print(f"Periode  : {OUTPUT_PERIOD}")
    print(f"Final    : {OUTPUT_FINAL}")
    print(f"Audit    : {OUTPUT_AUDIT}")
    print(f"Passes   : {OUTPUT_PASS_AUDIT}")
    print("=" * 80)
    return 0
# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Scraper review Google Play Wayground/Quizizz."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rebuild-from-raw",
        action="store_true",
        help=(
            "Bangun ulang 02_data_akhir_dedup.csv dan "
            "02_data_akhir.csv dari 01_semua_data.csv "
            "tanpa memanggil Google Play."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="Folder untuk raw, dataset periode, dataset final dan audit.",
    )
    args = parser.parse_args()
    DATA_DIR = args.output_dir.resolve()
    OUTPUT_RAW = DATA_DIR / OUTPUT_RAW.name
    OUTPUT_PERIOD = DATA_DIR / OUTPUT_PERIOD.name
    OUTPUT_FINAL = DATA_DIR / OUTPUT_FINAL.name
    OUTPUT_AUDIT = DATA_DIR / OUTPUT_AUDIT.name
    OUTPUT_PASS_AUDIT = DATA_DIR / OUTPUT_PASS_AUDIT.name
    if args.rebuild_from_raw:
        raise SystemExit(
            rebuild_from_raw()
        )
    raise SystemExit(
        main()
    )
