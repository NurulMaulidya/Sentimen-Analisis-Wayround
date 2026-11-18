"""
Pembersihan database produksi dari artefak test dan unggahan holdout.

KONTEKS RISET
=============
Audit menemukan:
  1. Sesi test pernah menulis ke database produksi (fixture test tidak
     mengarahkan engine ke database test).
  2. File holdout pernah diunggah ke website, sehingga 604 baris holdout
     pernah berada di jalur prediksi web.

Fakta historis ini TIDAK dihapus dari dokumentasi (lihat
docs/DECISIONS/D4-KONFLIK-KONFIGURASI.md dan docs/remediation/), tetapi
artefak yang tidak sah harus dibersihkan dari database produksi:

  * sesi analisis yang isinya berasal dari holdout canonical, dan
  * sesi analisis yang dibuat oleh test suite.

Sesi analisis MANUAL yang sah (mis. analisis peneliti atas dataset penuh)
TIDAK disentuh, kecuali isinya terdeteksi sebagai baris holdout.

SCRIPT INI AMAN (idempotent + backup):
  * selalu membuat backup SQLite + manifest sha256 sebelum menghapus,
  * dapat dijalankan ulang tanpa efek samping,
  * menolak jalan bila target bukan database produksi.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.constants import CANONICAL_DATASET_PATH, PIPELINE_OUTPUT_DIR

DEFAULT_DB = PROJECT_ROOT / "data" / "sentiment_analysis.db"
CANONICAL_HOLDOUT_PATH = PIPELINE_OUTPUT_DIR / "stage5_split_v2" / "holdout.csv"

# Nama/berkas yang jelas berasal dari test suite.
TEST_SESSION_NAMES = {"Test Analysis", "isolated audit", "batch memory audit"}
TEST_SESSION_FILENAMES = {"memory.csv", "test.csv"}

# Teks fixture yang dipakai test suite (tests/test_web_flow.py,
# tests/test_analysis_lifecycle.py). Prediksi tunggal dengan teks ini
# adalah artefak test, bukan riwayat pengguna.
TEST_PREDICTION_TEXTS = {
    "Aplikasi error tidak bisa login",
    "aplikasi apa ini aneh sering keluar terus",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_holdout_review_ids() -> set:
    """reviewId baris holdout canonical."""
    if not CANONICAL_HOLDOUT_PATH.exists():
        return set()
    import pandas as pd

    frame = pd.read_csv(CANONICAL_HOLDOUT_PATH)
    if "reviewId" not in frame.columns:
        return set()
    return {str(value) for value in frame["reviewId"].dropna()}


def session_review_ids(connection, analysis_id: int) -> set:
    """reviewId yang tercatat pada satu sesi analisis."""
    ids = set()
    rows = connection.execute(
        "SELECT raw_data FROM upload_rows WHERE analysis_id = ?", (analysis_id,)
    ).fetchall()
    for (raw,) in rows:
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("reviewId"):
            ids.add(str(payload["reviewId"]))
    return ids


def classify_sessions(connection, holdout_ids: set) -> tuple:
    """
    Pisahkan sesi menjadi (holdout, test, sah).

    Sesi disebut "holdout" bila reviewId yang tercatat beririsan dengan
    holdout canonical. Sesi disebut "test" bila nama/berkasnya berasal dari
    test suite.
    """
    holdout, test, keep = [], [], []

    sessions = connection.execute(
        "SELECT id, name, filename, filepath, original_count FROM analysis_sessions"
    ).fetchall()

    for session_id, name, filename, filepath, count in sessions:
        ids = session_review_ids(connection, session_id)
        if holdout_ids and (ids & holdout_ids):
            holdout.append((session_id, name, filename, filepath, count))
            continue

        if (name in TEST_SESSION_NAMES) or (filename in TEST_SESSION_FILENAMES):
            test.append((session_id, name, filename, filepath, count))
            continue

        keep.append((session_id, name, filename, filepath, count))

    return holdout, test, keep


def delete_sessions(connection, session_ids: list) -> None:
    """Hapus sesi beserta seluruh baris anak (cascade di level FK)."""
    for session_id in session_ids:
        connection.execute(
            "DELETE FROM review_features WHERE analysis_id = ?", (session_id,)
        )
        connection.execute(
            "DELETE FROM aspect_results WHERE analysis_id = ?", (session_id,)
        )
        connection.execute(
            "DELETE FROM evaluation_results WHERE analysis_id = ?", (session_id,)
        )
        connection.execute(
            "DELETE FROM analysis_executions WHERE analysis_id = ?", (session_id,)
        )
        connection.execute(
            "DELETE FROM analysis_run_snapshots WHERE analysis_id = ?", (session_id,)
        )
        connection.execute(
            "DELETE FROM upload_rows WHERE analysis_id = ?", (session_id,)
        )
        connection.execute(
            "DELETE FROM review_data WHERE analysis_id = ?", (session_id,)
        )
        connection.execute(
            "DELETE FROM analysis_sessions WHERE id = ?", (session_id,)
        )


def delete_test_predictions(connection) -> int:
    """Hapus riwayat prediksi tunggal yang berasal dari fixture test."""
    removed = 0
    rows = connection.execute("SELECT id, payload FROM single_predictions").fetchall()
    for prediction_id, payload in rows:
        try:
            data = json.loads(payload)
        except (TypeError, ValueError):
            continue
        original = str(data.get("original", "")).strip()
        if original in TEST_PREDICTION_TEXTS:
            connection.execute(
                "DELETE FROM single_predictions WHERE id = ?", (prediction_id,)
            )
            removed += 1
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bersihkan database produksi dari sesi test dan unggahan holdout."
    )
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path database produksi.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Hanya tampilkan rencana, jangan hapus."
    )
    parser.add_argument(
        "--no-backup", action="store_true", help="Lewati pembuatan backup (tidak disarankan)."
    )
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    if db_path == DEFAULT_DB.resolve() and "pytest" in sys.modules:
        print("ERROR: script ini tidak boleh jalan di lingkungan test.")
        return 2
    if not db_path.exists():
        print(f"Database tidak ditemukan: {db_path}")
        return 1

    holdout_ids = canonical_holdout_review_ids()
    if not holdout_ids:
        print("WARNING: holdout canonical tidak terbaca; deteksi holdout dilewati.")

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        holdout, test, keep = classify_sessions(connection, holdout_ids)

        print("=" * 78)
        print("RENCANA PEMBERSIHAN DATABASE PRODUKSI")
        print("=" * 78)
        print(f"Database : {db_path}")
        print(f"Holdout canonical: {len(holdout_ids)} reviewId")
        print()
        print(f"[HOLDOUT] {len(holdout)} sesi akan dihapus:")
        for row in holdout:
            print(f"   id={row[0]:<4} {row[1]!r} ({row[2]}, {row[4]} baris)")
        print(f"[TEST]    {len(test)} sesi akan dihapus:")
        for row in test:
            print(f"   id={row[0]:<4} {row[1]!r} ({row[2]}, {row[4]} baris)")
        print(f"[SAH]     {len(keep)} sesi dipertahankan:")
        for row in keep:
            print(f"   id={row[0]:<4} {row[1]!r} ({row[2]}, {row[4]} baris)")
        print()

        if args.dry_run:
            print("DRY RUN: tidak ada perubahan yang dilakukan.")
            return 0

        backup_path = None
        if not args.no_backup:
            backup_dir = db_path.parent / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
            backup_path = backup_dir / f"{db_path.stem}_before_cleanup_{stamp}.sqlite"
            with sqlite3.connect(db_path) as source, sqlite3.connect(backup_path) as target:
                source.backup(target)
            manifest = {
                "source": str(db_path),
                "backup": str(backup_path),
                "reason": "Hapus sesi test dan unggahan holdout dari DB produksi",
                "sha256": sha256(backup_path),
                "removed_sessions": [row[0] for row in holdout + test],
                "kept_sessions": [row[0] for row in keep],
            }
            backup_path.with_suffix(".json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            print(f"Backup   : {backup_path}")
            print(f"SHA256   : {manifest['sha256']}")
            print()

        removed_ids = [row[0] for row in holdout + test]
        delete_sessions(connection, removed_ids)
        removed_predictions = delete_test_predictions(connection)
        connection.commit()

        # Hapus berkas unggahan milik sesi yang dihapus.
        removed_files = []
        for session_id, name, filename, filepath, count in holdout + test:
            if not filepath:
                continue
            candidate = Path(filepath)
            if not candidate.is_absolute():
                candidate = PROJECT_ROOT / candidate
            if candidate.exists() and candidate.is_file():
                candidate.unlink()
                removed_files.append(str(candidate))

        remaining = connection.execute(
            "SELECT COUNT(*) FROM analysis_sessions"
        ).fetchone()[0]

        print(f"Sesi dihapus        : {len(removed_ids)}")
        print(f"Prediksi test dihapus: {removed_predictions}")
        print(f"Berkas unggahan dihapus: {len(removed_files)}")
        for path in removed_files:
            print(f"   - {path}")
        print(f"Sesi tersisa        : {remaining}")
        print("=" * 78)
        return 0
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
