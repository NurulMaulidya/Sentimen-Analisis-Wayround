"""
Laporan environment reproducibility.

Menampilkan versi pustaka yang sedang aktif dibandingkan dengan environment
penelitian terkunci (requirements-lock.txt), yang dipakai saat model final
v4 dilatih.

Perintah:
    python scripts/check_environment.py

Kode keluar:
    0 -> environment cocok dengan environment terkunci
    1 -> ada selisih yang memblokir training ulang

Script ini TIDAK mengubah berkas apa pun dan tidak menyentuh model.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.environment import (  # noqa: E402
    LOCKED_VERSIONS,
    active_versions,
    environment_drift,
)


def main() -> int:
    drift = environment_drift()
    active = active_versions()

    print("=" * 80)
    print("WAYGROUND - LAPORAN ENVIRONMENT REPRODUCIBILITY")
    print("=" * 80)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Python       : {active['python']}")
    print()
    print(f"{'Pustaka':<16}{'Terkunci':<14}{'Aktif':<14}Status")
    print("-" * 80)

    blocking = 0
    for item in drift:
        if item["severity"] == "blocking":
            status = "TIDAK COCOK"
            blocking += 1
        elif item["severity"] == "informational":
            status = "beda patch level"
        else:
            status = "cocok"
        print(
            f"{item['library']:<16}{item['locked']:<14}{item['active']:<14}{status}"
        )

    print("-" * 80)
    print()

    if blocking:
        print("HASIL: environment aktif DI LUAR environment penelitian terkunci.")
        print()
        print("Dampak:")
        print("  - Backend, test, dan website TIDAK terpengaruh: artefak canonical")
        print("    sudah di-fit dan hanya dipakai untuk transform()/predict().")
        print("  - Training ulang DI LARANG di environment ini: hasilnya tidak")
        print("    persis sama dengan angka canonical, sehingga laporan menjadi")
        print("    tidak reproducible.")
        print()
        print("Cara memakai environment terkunci:")
        print("  python -m venv .venv310")
        print("  .venv310\\Scripts\\python -m pip install -r requirements-lock.txt")
        print("  .venv310\\Scripts\\python scripts\\train_final_model.py")
        print()
        print("Rujukan: docs/remediation/ENVIRONMENT-REPRODUCIBILITY.md")
        return 1

    print("HASIL: environment aktif sama dengan environment penelitian terkunci.")
    print("Training ulang di environment ini mereproduksi angka canonical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
