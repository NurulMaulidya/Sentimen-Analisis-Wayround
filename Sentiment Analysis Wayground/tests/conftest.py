"""
Konfigurasi pytest bersama.

TUJUAN UTAMA
============
Seluruh test suite WAJIB berjalan terhadap database dan folder upload TEST,
tidak pernah terhadap database produksi (``data/sentiment_analysis.db``).

Sebelum perbaikan ini, fixture test men-set ``backend.config.DATABASE_URI``
SETELAH ``backend.app`` ter-import. Karena Flask-SQLAlchemy mengunci engine
pada saat import, override itu tidak berefek: test menulis ke database
produksi. Akibatnya database produksi berisi sesi test dan satu unggahan
holdout (604 baris).

Mekanisme sekarang:
  1. Modul ini di-import pytest SEBELUM modul test, jadi environment
     variable di-set lebih dulu.
  2. ``backend.config`` membaca environment variable tersebut SATU KALI saat
     import, sehingga engine DB diarahkan ke file database test.
  3. Fixture ``_guard_production_database`` memverifikasi bahwa file
     database produksi dan folder upload produksi tidak berubah selama
     seluruh sesi test.

Semua test yang butuh fixture ``api``/``client`` mewarisi jaminan ini.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# PATH
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PRODUCTION_DB = PROJECT_ROOT / "data" / "sentiment_analysis.db"
PRODUCTION_UPLOADS = PROJECT_ROOT / "data" / "uploads"

# Direktori sementara untuk database dan upload test. Di luar repo supaya
# tidak ikut ter-commit dan tidak mengotori working tree.
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="wayground-pytest-"))
TEST_DB_PATH = _TEST_ROOT / "test_sentiment_analysis.db"
TEST_UPLOADS = _TEST_ROOT / "uploads"


# ---------------------------------------------------------------------------
# ENVIRONMENT (WAJIB SEBELUM backend di-import)
# ---------------------------------------------------------------------------

def _configure_test_environment() -> None:
    """Arahkan backend ke database test dan folder upload test."""
    TEST_UPLOADS.mkdir(parents=True, exist_ok=True)

    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()

    os.environ["WAYGROUND_DATABASE_URI"] = f"sqlite:///{TEST_DB_PATH}"
    os.environ["WAYGROUND_DATABASE_PATH"] = str(TEST_DB_PATH)
    os.environ["WAYGROUND_UPLOAD_DIR"] = str(TEST_UPLOADS)


_configure_test_environment()


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_fingerprint(path: Path) -> list:
    """Daftar (nama, ukuran, sha256) seluruh file di dalam satu folder."""
    if not path.exists():
        return []
    entries = []
    for item in sorted(path.rglob("*")):
        if item.is_file():
            entries.append((str(item.relative_to(path)), item.stat().st_size, _sha256(item)))
    return entries


def is_production_database(database_path: object) -> bool:
    """True bila path tersebut menunjuk ke database produksi."""
    if not database_path:
        return False
    return Path(str(database_path)).resolve() == PRODUCTION_DB.resolve()


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _guard_production_database():
    """
    Pastikan test suite TIDAK mengubah database produksi.

    Guard ini gagal keras (assertion) bila file database produksi atau folder
    upload produksi berubah selama test berjalan. Ini yang membuat
    "test tidak menulis database produksi" dapat diverifikasi secara
    otomatis, bukan sekadar diklaim.
    """
    before_db = _sha256(PRODUCTION_DB)
    before_uploads = _directory_fingerprint(PRODUCTION_UPLOADS)

    yield

    after_db = _sha256(PRODUCTION_DB)
    after_uploads = _directory_fingerprint(PRODUCTION_UPLOADS)

    assert after_db == before_db, (
        "Database produksi berubah selama test suite berjalan: "
        f"{PRODUCTION_DB}. Test WAJIB memakai database test."
    )
    assert after_uploads == before_uploads, (
        "Folder upload produksi berubah selama test suite berjalan: "
        f"{PRODUCTION_UPLOADS}. Test WAJIB memakai folder upload test."
    )


@pytest.fixture(scope="session", autouse=True)
def _clean_test_environment():
    """Hapus artefak database test saat sesi test selesai."""
    yield
    shutil.rmtree(_TEST_ROOT, ignore_errors=True)


@pytest.fixture
def test_database_path() -> Path:
    """Path file database yang dipakai test suite."""
    return TEST_DB_PATH


@pytest.fixture
def production_database_path() -> Path:
    """Path database produksi (hanya untuk assertion, bukan untuk ditulis)."""
    return PRODUCTION_DB
