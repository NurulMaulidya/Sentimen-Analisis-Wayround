"""
Build Worksheet Anotasi Tambahan (Tahap 2)
========================================

Tujuan:
- Menyiapkan worksheet untuk validasi manual tambahan (~500 ulasan) yang
  menggabungkan label experts lama (288 baris) dengan label baru.
- Semua baris diberi `reviewId` sehingga integrasi label WAJIB lewat
  reviewId, bukan pencocokan teks.

Strata sampling (WAJIB, sesuai brief):
  A. netral_rating_tinggi : SEMUA ulasan berlabel Netral (lexicon v2)
                             yang berating >= 4.
  B. konflik_rating_label  : ulasan yang label lexicon v2-nya bertentangan
                             dengan label berbasis rating.
  C. acak_per_kelas       : sampel acak per kelas label lexicon v2.
  D. revalidasi_lama       : sampel acak dari 288 baris yang sudah pernah
                             dilabeli pakar (untuk mengukur kesesuaian
                             antar-annotator / Cohen's kappa).

Prioritas strata: A > D > B > C. Baris yang sudah pernah dilabeli experts
tidak boleh masuk strata B/C, supaya tidak terannotasi dua kali.

Kolom keluaran:
  reviewId, content, processed_text, rating, label_lexicon_v2,
  label_pakar_1, label_pakar_2, catatan

`label_pakar_1` untuk strata D diisi dari label experts lama (bukan
direkayasa); untuk strata lain dikosongkan.

Tidak ada file canonical yang ditimpa oleh script ini.
"""

from __future__ import annotations

import argparse
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

CANONICAL = CANONICAL_DATASET_PATH
OUT_XLSX = PROJECT_ROOT / "dataset" / "anotasi_tambahan.xlsx"
OUT_TSV = PROJECT_ROOT / "dataset" / "anotasi_tambahan.tsv"
OUT_META = LEXICON_AUDIT_DIR / "anotasi_tambahan_meta.json"

CLASSES = ["Positif", "Negatif", "Netral"]

# Kuota sampling (dapat diubah lewat argumen CLI).
QUOTA_A = None          # None = semua (tidak dibatasi)
QUOTA_REVALIDASI = 40
QUOTA_KONFLIK = 70
QUOTA_ACAK_PER_KELAS = 16


def rating_to_label(rating) -> str:
    """Baseline rating: >=4 Positif, <=2 Negatif, 3 Netral."""
    if rating is None or pd.isna(rating):
        return "Netral"
    value = int(rating)
    if value >= 4:
        return "Positif"
    if value <= 2:
        return "Negatif"
    return "Netral"


def load_canonical() -> pd.DataFrame:
    df = pd.read_csv(CANONICAL, encoding="utf-8-sig", dtype=str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["_label_v2"] = [
        label_text(t if isinstance(t, str) else "")["label"]
        for t in df["processed_text"].fillna("")
    ]
    df["_label_rating"] = df["rating"].map(rating_to_label)
    df["_sudah_pakar"] = df["sentiment_label_expert"].notna()
    return df


def stratified_random(
    df: pd.DataFrame,
    n: int,
    random_state: int,
) -> pd.DataFrame:
    """Ambil n baris acak (stratified per kelas label v2 bila memungkinkan)."""
    if n <= 0 or df.empty:
        return df.head(0)

    per_class = max(1, n // max(1, df["_label_v2"].nunique()))
    picked = []

    for cls in CLASSES:
        part = df[df["_label_v2"] == cls]
        take = min(per_class, len(part))
        if take:
            picked.append(part.sample(n=take, random_state=random_state))

    out = pd.concat(picked) if picked else df.head(0)
    out = out.drop_duplicates(subset="reviewId")

    # Sisa kuota diisi dari sisa baris yang belum terambil.
    if len(out) < n:
        rest = df[~df["reviewId"].isin(out["reviewId"])]
        need = n - len(out)
        if len(rest):
            out = pd.concat(
                [out, rest.sample(n=min(need, len(rest)), random_state=random_state)]
            )

    return out


def build(
    target_total: int = 500,
    random_state: int = 42,
    quota_revalidasi: int = QUOTA_REVALIDASI,
    quota_konflik: int = QUOTA_KONFLIK,
    quota_acak_per_kelas: int = QUOTA_ACAK_PER_KELAS,
) -> pd.DataFrame:
    df = load_canonical()
    df = df[df["reviewId"].notna() & (df["reviewId"].astype(str).str.strip() != "")]
    df = df.drop_duplicates(subset="reviewId").copy()

    frames = []

    def add(frame: pd.DataFrame, stratum: str, alasan: str) -> None:
        if frame.empty:
            return
        part = frame.copy()
        part["stratum"] = stratum
        part["alasan_sampling"] = alasan
        frames.append(part)

    # ------------------------------------------------------------------
    # Stratum A - WAJIB semua Netral (lexicon v2) dengan rating >= 4.
    # ------------------------------------------------------------------
    a_mask = (df["_label_v2"] == "Netral") & (df["rating"] >= 4)
    a_df = df[a_mask]
    if QUOTA_A is not None:
        a_df = stratified_random(a_df, QUOTA_A, random_state)
    add(
        a_df,
        "A_netral_rating_tinggi",
        "Netral menurut lexicon v2 tetapi rating >= 4 (klasifikasi perlu "
        "dipertimbangkan ulang oleh pakar)",
    )
    used = set(a_df["reviewId"])

    # ------------------------------------------------------------------
    # Stratum D - revalidasi dari 288 baris berlabel pakar lama.
    # ------------------------------------------------------------------
    d_pool = df[df["_sudah_pakar"] & ~df["reviewId"].isin(used)]
    d_df = stratified_random(d_pool, quota_revalidasi, random_state)
    add(
        d_df,
        "D_revalidasi_lama",
        "Pernah dilabeli oleh pakar lama; dipakai untuk "
        "mengukur kesesuaian antar-annotator",
    )
    used |= set(d_df["reviewId"])

    # ------------------------------------------------------------------
    # Stratum B - konflik antara label lexicon v2 dan label rating.
    # ------------------------------------------------------------------
    b_pool = df[
        (df["_label_v2"] != df["_label_rating"])
        & ~df["reviewId"].isin(used)
    ]
    b_df = stratified_random(b_pool, quota_konflik, random_state)
    add(
        b_df,
        "B_konflik_rating_label",
        "Label lexicon v2 bertentangan dengan label berbasis rating",
    )
    used |= set(b_df["reviewId"])

    # ------------------------------------------------------------------
    # Stratum C - sampel acak per kelas (sisa dari seluruh kelas).
    # ------------------------------------------------------------------
    c_pool = df[~df["reviewId"].isin(used)]
    c_parts = []
    for cls in CLASSES:
        part = c_pool[c_pool["_label_v2"] == cls]
        take = min(quota_acak_per_kelas, len(part))
        if take:
            c_parts.append(part.sample(n=take, random_state=random_state))
    c_df = pd.concat(c_parts) if c_parts else c_pool.head(0)
    add(
        c_df,
        "C_acak_per_kelas",
        "Sampel acak per kelas label lexicon v2",
    )

    sheet = pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Isi label_pakar_1 untuk strata D dari label pakar lama.
    # ------------------------------------------------------------------
    sheet["label_pakar_1"] = ""
    sheet["label_pakar_2"] = ""
    sheet["catatan"] = ""
    sheet["sumber_label_pakar_1"] = ""

    lama = sheet["stratum"].eq("D_revalidasi_lama")
    sheet.loc[lama, "label_pakar_1"] = sheet.loc[lama, "sentiment_label_expert"]
    sheet.loc[lama, "sumber_label_pakar_1"] = "pakar lama (288 label)"
    sheet.loc[~lama, "sumber_label_pakar_1"] = "kosong (menunggu annotator)"

    # Susun kolom keluaran.
    out = sheet[
        [
            "reviewId",
            "content",
            "processed_text",
            "rating",
            "_label_v2",
            "label_pakar_1",
            "label_pakar_2",
            "catatan",
            "sumber_label_pakar_1",
        ]
    ].rename(columns={"_label_v2": "label_lexicon_v2"})

    out["stratum"] = sheet["stratum"].values
    out["alasan_sampling"] = sheet["alasan_sampling"].values

    # Urutkan agar mudah dibaca: stratum -> kelas -> rating.
    out = out.sort_values(
        by=["stratum", "label_lexicon_v2", "rating"],
        ascending=[True, True, False],
        kind="stable",
    ).reset_index(drop=True)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Membangun worksheet anotasi tambahan."
    )
    parser.add_argument("--total", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--revalidasi", type=int, default=QUOTA_REVALIDASI)
    parser.add_argument("--konflik", type=int, default=QUOTA_KONFLIK)
    parser.add_argument("--acak-per-kelas", type=int, default=QUOTA_ACAK_PER_KELAS)
    args = parser.parse_args()

    out = build(
        target_total=args.total,
        random_state=args.random_state,
        quota_revalidasi=args.revalidasi,
        quota_konflik=args.konflik,
        quota_acak_per_kelas=args.acak_per_kelas,
    )

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    OUT_META.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="anotasi", index=False)
        meta = (
            out.groupby(["stratum", "label_lexicon_v2"])
            .size()
            .reset_index(name="jumlah")
        )
        meta.to_excel(writer, sheet_name="ringkasan_stratum", index=False)

    out.to_csv(
        OUT_TSV,
        sep="\t",
        index=False,
        encoding="utf-8-sig",
    )

    meta = {
        "canonical": str(CANONICAL),
        "total_baris": int(len(out)),
        "random_state": args.random_state,
        "kuota": {
            "A_netral_rating_tinggi": "semua",
            "D_revalidasi_lama": args.revalidasi,
            "B_konflik_rating_label": args.konflik,
            "C_acak_per_kelas_per_kelas": args.acak_per_kelas,
        },
        "komposisi_stratum": out["stratum"].value_counts().to_dict(),
        "komposisi_kelas": out["label_lexicon_v2"].value_counts().to_dict(),
        "sudah_terisi_label_pakar_1": int(
            (out["label_pakar_1"].fillna("").str.strip() != "").sum()
        ),
        "perlu_diisi_annotator": int(
            (out["label_pakar_2"].fillna("").str.strip() == "").sum()
        ),
    }
    OUT_META.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 78)
    print("WORKSHEET ANOTASI TAMBAHAN")
    print("=" * 78)
    print(f"Total baris : {len(out)}")
    print(f"XLSX        : {OUT_XLSX}")
    print(f"TSV         : {OUT_TSV}")
    print(f"Metadata    : {OUT_META}")
    print()
    print("Komposisi stratum:")
    for key, value in out["stratum"].value_counts().items():
        print(f"  {key:<28} {value}")
    print()
    print("Komposisi kelas (lexicon v2):")
    for key, value in out["label_lexicon_v2"].value_counts().items():
        print(f"  {key:<28} {value}")
    print()
    print(
        f"label_pakar_1 sudah terisi : "
        f"{meta['sudah_terisi_label_pakar_1']} (dari label pakar lama)"
    )
    print(
        f"label_pakar_2 masih kosong  : "
        f"{meta['perlu_diisi_annotator']} (menunggu anotator kedua)"
    )
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
