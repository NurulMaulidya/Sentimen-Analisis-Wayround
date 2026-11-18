"""
Pemeriksaan environment reproducibility.

Model final v4 dilatih pada environment terkunci:

    Python 3.10.6 / numpy 1.26.2 / pandas 2.1.4 / scikit-learn 1.7.2

Environment itu didefinisikan di ``requirements-lock.txt``. Environment
yang dipakai untuk menjalankan website dan test boleh berbeda, karena
artefak canonical sudah di-fit dan hanya dipakai untuk ``transform()`` dan
``predict()``. Yang TIDAK boleh dilakukan diam-diam adalah Melatih ulang
model di environment yang berbeda: hasilnya tidak persis sama dengan
angka canonical sehingga laporan menjadi tidak reproducible.

Modul ini dipakai oleh:

  * ``scripts/train_final_model.py``  -> gerbang sebelum training,
  * ``scripts/check_environment.py``  -> laporan manual,
  * ``tests/test_environment_contract.py`` -> regression test.
"""

from __future__ import annotations

import platform
from typing import Any

__all__ = [
    "LOCKED_VERSIONS",
    "EnvironmentDriftError",
    "active_versions",
    "environment_drift",
    "require_locked_environment",
]

# Versi environment tempat model final v4 dilatih.
# Sumber: models/final_v4/metadata.json -> "versions", dan
# requirements-lock.txt.
LOCKED_VERSIONS: dict[str, str] = {
    "python": "3.10.6",
    "numpy": "1.26.2",
    "pandas": "2.1.4",
    "scikit-learn": "1.7.2",
}

# Pustaka yang perbedaannya mengubah angka hasil training.
NUMERIC_LIBRARIES = ("numpy", "pandas", "scikit-learn")

# Environment variable untuk mengizinkan training di environment yang tidak
# terkunci. Pesan error selalu menyebut environment yang benar.
OVERRIDE_ENV_VAR = "WAYGROUND_ALLOW_ENV_DRIFT"


class EnvironmentDriftError(RuntimeError):
    """Environment aktif tidak cocok dengan environment penelitian terkunci."""


def active_versions() -> dict[str, str]:
    """Versi pustaka pada environment yang sedang berjalan."""
    versions: dict[str, str] = {
        "python": platform.python_version(),
    }

    for module_name, key in (
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
    ):
        try:
            module = __import__(module_name)
        except Exception:  # pragma: no cover - dependency wajib backend
            versions[key] = "tidak terpasang"
            continue
        versions[key] = str(getattr(module, "__version__", "tidak diketahui"))

    return versions


def _python_matches(locked: str, active: str) -> bool:
    """
    Bandingkan major.minor Python.

    Patch level dilaporkan sebagai informasi, bukan sebagai penentu,
    karena perbedaan patch pada Python tidak mengubah hasil numerik
    training. Sebaliknya, perbedaan versi numpy/pandas/scikit-learn bisa
    mengubah hasil, sehingga pustaka itu dibandingkan secara persis.
    """
    try:
        locked_parts = locked.split(".")[:2]
        active_parts = active.split(".")[:2]
    except AttributeError:  # pragma: no cover - bentuk tak terduga
        return False
    return locked_parts == active_parts


def environment_drift() -> list[dict[str, Any]]:
    """
    Daftar selisih antara environment aktif dan environment terkunci.

    Dikembalikan sebagai daftar dict supaya bisa ditampilkan apa adanya
    (tanpa membiarkan pembaca mengira daftar kosong berarti "terverifikasi"
    saat yang sebenarnya adalah "tidak diperiksa").
    """
    active = active_versions()
    drift: list[dict[str, Any]] = []

    for library in ("python", *NUMERIC_LIBRARIES):
        locked = LOCKED_VERSIONS[library]
        current = active.get(library, "tidak diketahui")

        if current == locked:
            matches, severity = True, "ok"
        elif library == "python" and _python_matches(locked, current):
            matches, severity = True, "informational"
        else:
            matches, severity = False, "blocking"

        drift.append(
            {
                "library": library,
                "locked": locked,
                "active": current,
                "matches": matches,
                "severity": severity,
            }
        )

    return drift


def require_locked_environment(
    override_env_var: str = OVERRIDE_ENV_VAR,
) -> dict[str, str]:
    """
    Pastikan environment aktif cocok dengan environment penelitian.

    Mengembalikan dict versi aktif bila cocok. Bila tidak cocok,Raises
    ``EnvironmentDriftError`` kecuali environment variable override diset.

    Override TIDAK menghapus drift: hasilnya tetap dikembalikan agar
    laporan bisa mencatat bahwa training dijalankan di luar environment
    terkunci.
    """
    import os

    drift = environment_drift()
    blocking = [item for item in drift if item["severity"] == "blocking"]

    if not blocking:
        return active_versions()

    details = "\n".join(
        f"  - {item['library']}: locked {item['locked']}, aktif {item['active']}"
        for item in blocking
    )
    message = (
        "Environment aktif tidak cocok dengan environment penelitian "
        "terkunci.\n"
        f"{details}\n"
        "\nTraining di environment lain menghasilkan angka yang tidak "
        "persis sama dengan artefak canonical, sehingga laporan menjadi "
        "tidak reproducible.\n"
        "\nGunakan environment terkunci:\n"
        "  python -m venv .venv310\n"
        "  .venv310\\Scripts\\python -m pip install -r requirements-lock.txt\n"
        "  .venv310\\Scripts\\python scripts\\train_final_model.py\n"
        "\nBila training di luar environment terkunci memang disengaja, "
        f"set environment variable {override_env_var}=1 dan catat "
        "penyimpangan ini pada laporan."
    )

    if os.getenv(override_env_var, "").strip().lower() in {"1", "true", "yes"}:
        return active_versions()

    raise EnvironmentDriftError(message)
