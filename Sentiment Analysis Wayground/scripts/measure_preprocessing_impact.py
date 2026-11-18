"""
Ukur dampak perubahan preprocessing (Tahap 4) terhadap dataset canonical.

Membandingkan `processed_text` yang TERSIMPAN dengan hasil preprocessing
baru (huruf berulang + emoji), lalu melaporkan:
  - jumlah baris yang berubah
  - contoh perubahan
  - perubahan pada label lexicon v2 (sebelum vs sesudah)

Menulis: output/lexicon_audit/preprocessing_impact.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from shared.constants import (  # noqa: E402
    CANONICAL_DATASET_PATH,
    LEXICON_AUDIT_DIR,
)

from shared.labeling import label_text  # noqa: E402
from shared.preprocessing import preprocess_text  # noqa: E402

CANONICAL = CANONICAL_DATASET_PATH
OUT_JSON = LEXICON_AUDIT_DIR / "preprocessing_impact.json"


def main() -> int:
    df = pd.read_csv(CANONICAL, encoding="utf-8-sig", dtype=str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    lama = df["processed_text"].fillna("").astype(str)
    baru = df["content"].fillna("").astype(str).map(preprocess_text)

    berubah = lama != baru

    label_lama = [label_text(t)["label"] for t in lama]
    label_baru = [label_text(t)["label"] for t in baru]

    berubah_label = 0
    contoh = []
    for i in range(len(df)):
        if label_lama[i] != label_baru[i]:
            berubah_label += 1
            if len(contoh) < 15:
                contoh.append({
                    "content": str(df.at[i, "content"])[:120],
                    "processed_lama": lama.iloc[i][:120],
                    "processed_baru": baru.iloc[i][:120],
                    "label_lama": label_lama[i],
                    "label_baru": label_baru[i],
                })

    hasil = {
        "total_baris": int(len(df)),
        "baris_berubah": int(berubah.sum()),
        "persentase_berubah": round(float(berubah.mean()) * 100, 2),
        "baris_label_berubah": int(berubah_label),
        "distribusi_label_lama": pd.Series(label_lama).value_counts().to_dict(),
        "distribusi_label_baru": pd.Series(label_baru).value_counts().to_dict(),
        "contoh_perubahan": contoh,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 78)
    print("DAMPAK PERUBAHAN PREPROCESSING (TAHAP 4)")
    print("=" * 78)
    print(f"Total baris            : {hasil['total_baris']}")
    print(f"processed_text berubah : {hasil['baris_berubah']} "
          f"({hasil['persentase_berubah']}%)")
    print(f"label lexicon berubah  : {hasil['baris_label_berubah']}")
    print()
    print("Distribusi label sebelum :", hasil['distribusi_label_lama'])
    print("Distribusi label sesudah :", hasil['distribusi_label_baru'])
    print()
    print(f"Detail: {OUT_JSON}")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
