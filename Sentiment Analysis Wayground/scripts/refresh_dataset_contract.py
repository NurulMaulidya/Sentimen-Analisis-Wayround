"""
Refresh Kontrak Dataset Canonical (Tahap 5)
=========================================

`shared/final_config.py` memuat "kontrak dataset": jumlah baris dan
distribusi kelas yang HARUS cocok dengan dataset canonical saat model
final dilatih. Nilai-nilai itu sebelumnya ditulis manual (3060 baris,
distribusi kelas lama), sehingga setiap kali dataset berubah (lexicon
v2 + preprocessing baru + integrasi label pakar) kontrak itu basi dan
pelatihan final gagal dengan pesan yang membingungkan.

Script ini memperbarui kontrak tersebut secara eksplisit dan auditable
dari artefak yang benar-benar dipakai:

    data/pipeline_output_v2/stage3_final_labeled.csv
    data/pipeline_output_v2/stage5_split_v2/split_metadata.json

Yang diperbarui (blok "CANONICAL DATASET CONTRACT" di shared/final_config.py):
    EXPECTED_DATASET_ROWS
    EXPECTED_CLASS_DISTRIBUTION
    EXPECTED_DEVELOPMENT_ROWS
    EXPECTED_HOLDOUT_ROWS
    EXPECTED_EXPERT_VALIDATED_ROWS

Pemakaian:
    python scripts/refresh_dataset_contract.py

CATATAN: Menjalankan script ini berarti menyatakan ulang persetujuan atas
kontrak dataset, dan harus dilakukan setelah hasil Tahap 5 dipresentasikan
kepada pemilik.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.constants import (  # noqa: E402
    CANONICAL_DATASET_PATH,
    PIPELINE_OUTPUT_DIR,
)

FINAL_CONFIG_PATH = PROJECT_ROOT / "shared" / "final_config.py"
SPLIT_DIRNAME = "stage5_split_v2"


def read_dataset(path: Path) -> tuple[int, dict[str, int], int]:
    """Baca jumlah baris, distribusi kelas, dan jumlah label expert."""
    import pandas as pd

    df = pd.read_csv(path, encoding="utf-8-sig")

    label_column = "sentiment_label_final"
    if label_column not in df.columns:
        raise KeyError(
            f"Kolom {label_column!r} tidak ada di {path}. "
            f"Kolom: {list(df.columns)}"
        )

    labels = df[label_column].astype(str).str.strip().str.lower()
    counts = {
        label: int((labels == label).sum())
        for label in sorted(labels.unique())
    }

    expert_rows = 0
    if "label_source" in df.columns:
        sources = df["label_source"].astype(str).str.strip().str.lower()
        expert_rows = int(sources.str.contains("expert").sum())

    return int(len(df)), counts, expert_rows


def read_split_meta() -> dict:
    """Baca metadata split canonical."""
    meta_path = PIPELINE_OUTPUT_DIR / SPLIT_DIRNAME / "split_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata split tidak ditemukan: {meta_path}\n"
            "Jalankan pipeline Stage 5 terlebih dahulu."
        )
    return json.loads(meta_path.read_text(encoding="utf-8"))


def rewrite_contract(
    path: Path,
    rows: int,
    distribution: dict[str, int],
    dev_rows: int,
    holdout_rows: int,
    expert_rows: int,
) -> None:
    """Tulis ulang blok kontrak dataset di shared/final_config.py."""
    text = path.read_text(encoding="utf-8")

    dist_lines = "\n".join(
        f'    "{label}": {value},'
        for label, value in sorted(distribution.items())
    )

    new_block = (
        "# Dataset final yang menjadi dasar locking.\n"
        "# (Diperbarui oleh scripts/refresh_dataset_contract.py)\n"
        f"EXPECTED_DATASET_ROWS = {rows}\n"
        "\n"
        "# Distribusi kelas harus sama persis dengan dataset canonical yang digunakan\n"
        "# untuk final model. Jika dataset aktual berbeda, locking HARUS berhenti.\n"
        "EXPECTED_CLASS_DISTRIBUTION = {\n"
        f"{dist_lines}\n"
        "}\n"
        "\n"
        f"EXPECTED_DEVELOPMENT_ROWS = {dev_rows}\n"
        f"EXPECTED_HOLDOUT_ROWS = {holdout_rows}\n"
        "\n"
        "# Berdasarkan validasi pakar yang tercatat pada konfigurasi final.\n"
        f"EXPECTED_EXPERT_VALIDATED_ROWS = {expert_rows}\n"
    )

    pattern = re.compile(
        r"# Dataset final yang menjadi dasar locking\.\n"
        r".*?"
        r"EXPECTED_EXPERT_VALIDATED_ROWS = \d+\n",
        re.DOTALL,
    )

    if not pattern.search(text):
        raise ValueError(
            "Blok kontrak dataset tidak ditemukan di "
            f"{path}. Struktur file mungkin berubah."
        )

    text = pattern.sub(new_block, text, count=1)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Perbarui kontrak dataset canonical di shared/final_config.py",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Tampilkan nilai baru tanpa menulis file.",
    )
    args = parser.parse_args()

    print("=" * 74)
    print("REFRESH KONTRAK DATASET CANONICAL")
    print("=" * 74)
    print(f"Dataset : {CANONICAL_DATASET_PATH}")

    rows, distribution, expert_rows = read_dataset(CANONICAL_DATASET_PATH)
    meta = read_split_meta()

    dev_rows = int(meta.get("development_rows", 0))
    holdout_rows = int(meta.get("holdout_rows", 0))

    print(f"Baris                : {rows}")
    print(f"Distribusi kelas     : {distribution}")
    print(f"Baris berlabel expert: {expert_rows}")
    print(f"Development / holdout: {dev_rows} / {holdout_rows}")
    print(f"Grup tumpang tindih  : {meta.get('grup_tumpang_tindih', '?')}")

    if dev_rows + holdout_rows != rows:
        raise SystemExit(
            f"Jumlah development + holdout ({dev_rows + holdout_rows}) "
            f"tidak sama dengan jumlah dataset ({rows})."
        )

    if args.dry_run:
        print("\n[DRY RUN] Tidak ada file yang ditulis.")
        return 0

    rewrite_contract(
        FINAL_CONFIG_PATH,
        rows,
        distribution,
        dev_rows,
        holdout_rows,
        expert_rows,
    )

    print(f"\n[OK] Kontrak dataset diperbarui: {FINAL_CONFIG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
