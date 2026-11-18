"""
Integrasi Label Pakar Wayground - Versi Baru (Tahap 2)
=====================================================

Perbedaan mendasar dari versi lama:

1. **Gabung lewat reviewId.** Label experts lama (288 baris, form TSV tanpa
   reviewId) dan label experts baru (worksheet anotasi tambahan) digabung
   ke baris dataset canonical berdasarkan `reviewId`. Label dari file lama
   dipetakan ke reviewId dengan urutan:
       a. content exact normalized
       b. content prefix (canonical diawali teks form; berguna
          untuk kasus teks form terpotong ellipsis saat scraping)
       c. fuzzy token (Jaccard >= 0,85 DAN selisih >= 0,10 terhadap kandidat
          kedua)
       Prefix/fuzzy hanya kandidat audit, tidak diterapkan otomatis.
       Setiap metode dicatat di audit, tidak ada positional matching.
2. **Prioritas label: experts > lexicon v2.** Baris tanpa label experts
   memakai label lexicon v2 (bukan lexicon v1 lagi).
3. **Kolom `label_source`.** Nilai: `Expert` atau `Lexicon v2`.
   Ditambah `label_source_detail` untuk rincian asal label.
4. **Cohen's kappa.** Dihitung bila ada dua annotator pada baris yang sama
   (label_pakar_1 dan label_pakar_2 terisi). Bila worksheet baru belum
   terisi, kappa antar-annotator dihitung dari label experts lama vs
   `label_pakar_1` pada baris yang sama (stratum revalidasi).
5. **Pengaman.** Integrasi **ditolak** bila worksheet anotasi baru ada
   tetapi belum berisi satu pun label terisi. Archivo canonical yang lama
   tidak ditimpa: keluaran default ditulis ke file versi baru.

Default:
  stage3      : data/pipeline_output/stage3_final_labeled.csv
  expert lama : dataset/Hasil_Validasi_Pakar.tsv
  expert baru : dataset/anotasi_tambahan.tsv
  output      : data/pipeline_output/stage3_final_labeled_v2.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.constants import (  # noqa: E402
    AUDIT_DIR,
    PIPELINE_OUTPUT_DIR,
)
from shared.labeling import get_active_lexicon, label_text  # noqa: E402
from shared.dataset_io import read_dataset

DEFAULT_STAGE3 = PIPELINE_OUTPUT_DIR / "stage3_final_labeled.csv"
DEFAULT_EXPERT_LAMA = PROJECT_ROOT / "dataset" / "Hasil_Validasi_Pakar.tsv"
DEFAULT_EXPERT_BARU = PROJECT_ROOT / "dataset" / "anotasi_tambahan.tsv"
DEFAULT_OUTPUT = PIPELINE_OUTPUT_DIR / "stage3_final_labeled_v2.csv"

AUDIT_FILE = AUDIT_DIR / "expert_label_integration_v2.csv"
MAPPING_FILE = AUDIT_DIR / "expert_review_id_mapping.csv"
SUMMARY_FILE = AUDIT_DIR / "expert_label_summary_v2.json"

CLASSES = ["Positif", "Negatif", "Netral"]
SEP = "=" * 78
VALID_LABELS = set(CLASSES)

FUZZY_MIN_JACCARD = 0.85
FUZZY_MIN_GAP = 0.10


# ==========================================================================
# UTILITAS NORMALISASI
# ==========================================================================

def normalize_text(value: object) -> str:
    """Normalisasi teks untuk pencocokan: lowercase + spasi rapat."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("﻿", "").replace(" ", " ")
    text = re.sub(r"[​-‍]", " ", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text))


def normalize_label(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    key = normalize_text(value)
    mapping = {
        "positif": "Positif",
        "positive": "Positif",
        "pos": "Positif",
        "1": "Positif",
        "netral": "Netral",
        "neutral": "Netral",
        "neu": "Netral",
        "0": "Netral",
        "negatif": "Negatif",
        "negative": "Negatif",
        "neg": "Negatif",
        "-1": "Negatif",
    }
    return mapping.get(key)


def detect_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:10000]
    return "\t" if sample.count("\t") >= sample.count(",") else ","


def read_table(path: Path) -> pd.DataFrame:
    """Baca tabel anotasi experts (TSV/CSV/XLSX) dengan deteksi header."""
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    suffix = path.suffix.lower()

    if suffix in {".tsv", ".txt"}:
        sep = "\t" if suffix == ".tsv" else detect_delimiter(path)
        raw = pd.read_csv(
            path, sep=sep, header=None, dtype=str,
            keep_default_na=False, encoding="utf-8-sig",
        )
    elif suffix == ".csv":
        raw = pd.read_csv(
            path, header=None, dtype=str,
            keep_default_na=False, encoding="utf-8-sig",
        )
    elif suffix in {".xlsx", ".xls"}:
        raw = pd.read_excel(path, header=None, dtype=str).fillna("")
    else:
        raise ValueError(f"Format tidak didukung: {suffix}")

    header_row = find_header_row(raw)
    header = [str(c).strip() for c in raw.iloc[header_row].tolist()]
    data = raw.iloc[header_row + 1:].copy()
    data.columns = header
    data = data.reset_index(drop=True)

    keep = [
        c for c in data.columns
        if not data[c].astype(str).str.strip().eq("").all()
    ]
    return data[keep]


def find_header_row(raw: pd.DataFrame) -> int:
    """
    Deteksi baris header form experts.

    Form lama: No | Ulasan Asli | Teks Preprocessing | Label | ...
    Form baru: reviewId | content | processed_text | rating |
               label_lexicon_v2 | label_pakar_1 | label_pakar_2 | ...
    """
    for i in range(min(len(raw), 80)):
        values = [normalize_text(x) for x in raw.iloc[i].tolist()]
        joined = " | ".join(values)

        has_id = any(
            v in {"review id", "review_id", "reviewid", "id ulasan"}
            for v in values
        )
        has_no = any(v in {"no", "nomor", "no."} for v in values)
        has_text = any(
            v in {"ulasan asli", "ulasan", "content", "review", "teks asli"}
            for v in values
        )
        has_label = any(
            "label" in v or v.startswith("label_pakar") for v in values
        )

        if has_id and has_text and has_label:
            return i
        if (has_no or has_id) and has_text and has_label:
            return i

    raise ValueError(
        "Header tabel experts tidak ditemukan. Form harus memuat "
        "kolom reviewId (atau No) + kolom teks + kolom label."
    )


def pick_column(
    df: pd.DataFrame,
    exact: List[str],
    contains: Optional[List[str]] = None,
) -> Optional[str]:
    contains = contains or []
    mapping = {normalize_text(c): c for c in df.columns}

    for name in exact:
        key = normalize_text(name)
        if key in mapping:
            return mapping[key]

    for c in df.columns:
        key = normalize_text(c)
        if any(term in key for term in contains):
            return c

    return None


# ==========================================================================
# PEMBACAAN FORM EXPERTS LAMA
# ==========================================================================

def read_expert_lama(path: Path) -> pd.DataFrame:
    df = read_table(path)

    text_col = pick_column(
        df, ["ulasan asli", "content", "review", "teks asli", "ulasan"]
    )
    if text_col is None:
        raise ValueError("Kolom 'Ulasan Asli' tidak ditemukan pada form lama.")

    label_col = pick_column(
        df,
        ["label pakar", "label expert", "expert label", "sentiment_label_expert"],
    )
    if label_col is None:
        generic = pick_column(df, ["label"])
        if generic is None:
            raise ValueError("Kolom label experts tidak ditemukan.")
        label_col = generic

    no_col = pick_column(df, ["no", "nomor", "no."])

    out = pd.DataFrame({
        "expert_no": df[no_col].astype(str).str.strip() if no_col else "",
        "text": df[text_col].map(normalize_text),
        "label": df[label_col].map(normalize_label),
    })
    out = out[(out["text"] != "") & out["label"].notna()].reset_index(drop=True)
    return out


# ==========================================================================
# PEMBACAAN WORKSHEET ANOTASI BARU
# ==========================================================================

def read_expert_baru(path: Path) -> pd.DataFrame:
    df = read_table(path)

    id_col = pick_column(
        df, ["reviewid", "review_id", "review id", "id ulasan"]
    )
    if id_col is None:
        raise ValueError(
            "Worksheet anotasi baru WAJIB memuat kolom reviewId. "
            "Integrasi label lewat pencocokan teks tidak diperbolehkan "
            "untuk file ini."
        )

    text_col = pick_column(
        df, ["content", "ulasan asli", "review", "teks asli", "ulasan"]
    )
    label1_col = pick_column(
        df, ["label_pakar_1", "label_pakar1", "label_pakar_1"]
    )
    label2_col = pick_column(
        df, ["label_pakar_2", "label_pakar2", "label_pakar_2"]
    )
    notes_col = pick_column(df, ["catatan", "catatanexpert"])
    sumber_col = pick_column(
        df, ["sumber_label_pakar_1", "sumberlabelpakar1", "sumber label 1"]
    )

    out = pd.DataFrame({
        "review_id": df[id_col].astype(str).str.strip(),
        "text": df[text_col].map(normalize_text) if text_col else "",
        "label_pakar_1": (
            df[label1_col].map(normalize_label) if label1_col else None
        ),
        "label_pakar_2": (
            df[label2_col].map(normalize_label) if label2_col else None
        ),
        "catatan": df[notes_col].fillna("") if notes_col else "",
        "sumber_label_pakar_1": (
            df[sumber_col].fillna("") if sumber_col else ""
        ),
    })
    return out


def as_label(value: object) -> str:
    """Kembalikan label sebagai string; NaN/None menjadi string kosong."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


# ==========================================================================
# PEMETAAN FORM LAMA -> reviewId
# ==========================================================================

def map_expert_lama(
    expert: pd.DataFrame,
    canonical: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Petakan baris form experts lama ke reviewId dataset canonical.

    Metode (berturutan):
      1. content exact normalized
      2. content prefix (canonical diawali teks form)
      3. fuzzy token (Jaccard + selisih terhadap kandidat kedua)
    """
    canon_norm = canonical["_content_norm"].tolist()
    exact_index: Dict[str, List[int]] = {}
    for pos, text in enumerate(canon_norm):
        if text:
            exact_index.setdefault(text, []).append(pos)

    canon_tokens = [tokenize(t) for t in canon_norm]

    counters = {
        "exact_content": 0,
        "prefix_content": 0,
        "fuzzy_content": 0,
        "ambiguous": 0,
        "unmatched": 0,
    }

    rows = []

    for _, row in expert.iterrows():
        text = row["text"]
        method = ""
        pos = None
        detail = ""

        # 1. Exact
        candidates = exact_index.get(text, [])
        if len(candidates) == 1:
            pos = candidates[0]
            method = "content_exact"
            counters["exact_content"] += 1
        elif len(candidates) > 1:
            method = "ambiguous_content_duplicate"
            counters["ambiguous"] += 1

        # 2. Prefix
        if pos is None and not method:
            prefix = [
                p for p, t in enumerate(canon_norm)
                if t and t.startswith(text) and len(text) >= 20
            ]
            if len(prefix) == 1:
                pos = prefix[0]
                method = "content_prefix"
                counters["prefix_content"] += 1
            elif len(prefix) > 1:
                method = "ambiguous_content_prefix"
                counters["ambiguous"] += 1

        # 3. Fuzzy token
        if pos is None and not method:
            q = tokenize(text)
            if q:
                scored = []
                for p, tt in enumerate(canon_tokens):
                    if not tt:
                        continue
                    union = q | tt
                    j = len(q & tt) / len(union) if union else 0.0
                    scored.append((j, p))
                scored.sort(reverse=True)
                if scored:
                    j1, p1 = scored[0]
                    j2 = scored[1][0] if len(scored) > 1 else 0.0
                    if (
                        j1 >= FUZZY_MIN_JACCARD
                        and (j1 - j2) >= FUZZY_MIN_GAP
                    ):
                        pos = p1
                        method = "content_fuzzy"
                        detail = f"jaccard={j1:.3f} gap={j1 - j2:.3f}"
                        counters["fuzzy_content"] += 1
                    else:
                        method = "unmatched"
                        detail = f"best_jaccard={j1:.3f}"
                        counters["unmatched"] += 1

        review_id = ""
        if pos is not None:
            review_id = canonical.iloc[pos]["reviewId"]

        rows.append({
            "expert_no": row["expert_no"],
            "text": text,
            "expert_label": row["label"],
            "review_id": review_id,
            "match_method": method,
            "match_detail": detail,
        })

    return pd.DataFrame(rows), counters


# ==========================================================================
# COHEN'S KAPPA
# ==========================================================================

def cohen_kappa(a: List[str], b: List[str]) -> Dict[str, object]:
    """Cohen's kappa dua annotator + matriks kesesuaian."""
    if not a or not b:
        return {
            "n": 0, "kappa": None, "observed_agreement": None,
            "expected_agreement": None,
            "confusion": {},
            "catatan": "Tidak ada pasangan label yang bisa dibandingkan.",
        }

    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n

    count_a = {c: a.count(c) for c in CLASSES}
    count_b = {c: b.count(c) for c in CLASSES}
    expected = sum(
        (count_a[c] / n) * (count_b[c] / n) for c in CLASSES
    )
    kappa = (
        (observed - expected) / (1 - expected)
        if expected < 1 else None
    )

    confusion: Dict[str, Dict[str, int]] = {}
    for c in CLASSES:
        confusion[c] = {}
        for d in CLASSES:
            confusion[c][d] = sum(
                1 for x, y in zip(a, b) if x == c and y == d
            )

    return {
        "n": n,
        "kappa": None if kappa is None else round(float(kappa), 4),
        "observed_agreement": round(float(observed), 4),
        "expected_agreement": round(float(expected), 4),
        "distribusi_annotator_1": count_a,
        "distribusi_annotator_2": count_b,
        "confusion": confusion,
    }


# ==========================================================================
# INTEGRASI
# ==========================================================================

def integrate(
    stage3_path: Path,
    expert_lama_path: Optional[Path],
    expert_baru_path: Optional[Path],
    output_path: Path,
    allow_unfilled_annotation: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    canonical = read_dataset(stage3_path)
    canonical["rating"] = pd.to_numeric(
        canonical["rating"], errors="coerce"
    )

    for col in ("reviewId", "content", "processed_text"):
        if col not in canonical.columns:
            raise ValueError(
                f"Kolom '{col}' tidak ada pada dataset canonical."
            )

    canonical["_content_norm"] = canonical["content"].map(normalize_text)
    if canonical.reviewId.isna().any() or canonical.reviewId.astype(str).str.strip().eq("").any():
        raise ValueError("Dataset contains missing reviewId")
    canonical["reviewId"] = canonical.reviewId.astype(str).str.strip()
    if canonical.reviewId.duplicated().any():
        raise ValueError("Dataset contains duplicate reviewId")

    # ------------------------------------------------------------------
    # Label lexicon v2 untuk seluruh baris (label cadangan).
    # ------------------------------------------------------------------
    lexicon = get_active_lexicon()
    canonical["label_lexicon_v2"] = [
        label_text(t if isinstance(t, str) else "", lexicon=lexicon)["label"]
        for t in canonical["processed_text"].fillna("")
    ]
    canonical["lexicon_version"] = lexicon.version

    # ------------------------------------------------------------------
    # Label experts lama -> reviewId.
    # ------------------------------------------------------------------
    mapping_rows = []
    counters_lama = {}
    if expert_lama_path is not None and expert_lama_path.exists():
        lama = read_expert_lama(expert_lama_path)
        mapped, counters_lama = map_expert_lama(lama, canonical)
        exact = mapped[mapped.match_method.eq("content_exact")]
        if exact.groupby("review_id").expert_label.nunique().gt(1).any():
            raise ValueError("Conflicting expert labels require adjudication")
        mapping_rows.append(mapped)
    else:
        counters_lama = {"file": "tidak ada"}

    # ------------------------------------------------------------------
    # Label experts baru (worksheet anotasi).
    # ------------------------------------------------------------------
    baru = None
    counters_baru: Dict[str, object] = {"file": "tidak ada"}
    if expert_baru_path is not None and expert_baru_path.exists():
        baru = read_expert_baru(expert_baru_path)
        annotated = baru[baru.label_pakar_1.notna() | baru.label_pakar_2.notna()]
        if annotated.review_id.duplicated().any():
            raise ValueError("Duplicate annotated reviewId requires adjudication")

        # Hitung label yang BENAR-BENAR baru diisikan annotator.
        # label_pakar_1 yang sudah terisi sejak awal (diisi ulang dari
        # 288 label experts lama) tidak dihitung sebagai anotasi baru.
        sumber = baru["sumber_label_pakar_1"].fillna("").astype(str).str.lower()
        dari_lama = sumber.str.contains("lama")
        baru_1 = baru["label_pakar_1"].notna() & ~dari_lama
        baru_2 = baru["label_pakar_2"].notna()
        filled = int(baru_1.sum() + baru_2.sum())

        counters_baru["label_baru_terisi"] = filled
        counters_baru["label_pakar_1_dari_label_lama"] = int(
            (baru["label_pakar_1"].notna() & dari_lama).sum()
        )

        if filled == 0 and not allow_unfilled_annotation:
            raise SystemExit(
                "\n" + SEP + "\n"
                "INTEGRASI DITOLAK: worksheet anotasi baru belum terisi.\n"
                f"File : {expert_baru_path}\n"
                "Isi kolom label_pakar_1 dan/atau label_pakar_2 terlebih "
                "dahulu, lalu jalankan ulang script ini.\n"
                "Tahap 2 berhenti di sini sampai anotasi selesai.\n"
                + SEP + "\n"
            )

        id_index = {
            rid: pos for pos, rid in enumerate(canonical["reviewId"])
        }
        matched = 0
        unmatched = 0
        for _, row in baru.iterrows():
            rid = row["review_id"]
            if rid and rid in id_index:
                matched += 1
            else:
                unmatched += 1

        counters_baru = {
            "file": str(expert_baru_path),
            "baris": int(len(baru)),
            "label_pakar_1_terisi": int(baru["label_pakar_1"].notna().sum()),
            "label_pakar_2_terisi": int(baru["label_pakar_2"].notna().sum()),
            "review_id_cocok": matched,
            "review_id_tidak_cocok": unmatched,
        }

    # ------------------------------------------------------------------
    # Susun label akhir.
    # ------------------------------------------------------------------
    result = canonical.copy()
    result["sentiment_label_lexicon_v2"] = result["label_lexicon_v2"]
    result["sentiment_label_expert"] = ""
    result["sentiment_label_expert_2"] = ""
    result["label_source"] = f"Lexicon {lexicon.version}"
    result["label_source_detail"] = (
        f"lexicon {lexicon.version} (threshold default)"
    )
    result["expert_validation_status"] = "Tidak tervalidasi"

    audit_rows = []

    # (a) experts lama
    if mapping_rows:
        mapped_lama = pd.concat(mapping_rows, ignore_index=True)
        for _, row in mapped_lama.iterrows():
            rid = row["review_id"]
            if not rid or row["match_method"] != "content_exact":
                audit_rows.append({
                    "sumber": "expert_lama",
                    "expert_no": row["expert_no"],
                    "review_id": "",
                    "match_method": row["match_method"],
                    "match_detail": row["match_detail"],
                    "expert_label": row["expert_label"] or "",
                    "label_akhir": "",
                })
                continue

            mask = result["reviewId"] == rid
            result.loc[mask, "sentiment_label_expert"] = row["expert_label"]
            result.loc[mask, "label_source"] = "Expert"
            result.loc[mask, "label_source_detail"] = (
                f"expert lama, metode: {row['match_method']}"
            )
            result.loc[mask, "expert_validation_status"] = (
                "Tervalidasi pakar lama"
            )

            audit_rows.append({
                "sumber": "expert_lama",
                "expert_no": row["expert_no"],
                "review_id": rid,
                "match_method": row["match_method"],
                "match_detail": row["match_detail"],
                "expert_label": row["expert_label"] or "",
                "label_akhir": row["expert_label"] or "",
            })

    # (b) experts baru (worksheet anotasi)
    kappa_1_2 = None
    kappa_lama_vs_baru = None

    if baru is not None and counters_baru.get("label_pakar_1_terisi", 0) > 0:
        pairs_12 = []
        pairs_lama_baru = []

        for _, row in baru.iterrows():
            rid = row["review_id"]
            l1 = row["label_pakar_1"]
            l2 = row["label_pakar_2"]
            if not rid:
                continue

            mask = result["reviewId"] == rid

            # A matching ID is insufficient when a worksheet contains stale text.
            if not mask.any() or not row["text"] or not result.loc[mask, "_content_norm"].eq(row["text"]).all():
                audit_rows.append({"sumber": "expert_baru", "review_id": rid,
                                   "match_method": "content_not_verified", "label_akhir": ""})
                continue

            # Capture the old value BEFORE any update; the previous code compared
            # the newly assigned label with itself and could report false kappa=1.
            lama_val = result.loc[mask, "sentiment_label_expert"].iloc[0]
            if as_label(l1) and as_label(l2):
                pairs_12.append((l1, l2))
            if as_label(lama_val) and as_label(l1):
                pairs_lama_baru.append((lama_val, l1))
            if ((as_label(l1) and as_label(l2) and l1 != l2)
                    or (as_label(lama_val) and as_label(l1) and lama_val != l1)):
                audit_rows.append({"sumber": "expert_baru", "review_id": rid,
                                   "match_method": "conflicting_labels", "label_akhir": ""})
                continue

            if as_label(l1):
                result.loc[mask, "sentiment_label_expert"] = l1
                result.loc[mask, "label_source"] = "Expert"
                result.loc[mask, "label_source_detail"] = (
                    "annotasi baru (label_pakar_1)"
                )
                result.loc[mask, "expert_validation_status"] = (
                    "Tervalidasi anotasi baru"
                )

            if as_label(l2):
                result.loc[mask, "sentiment_label_expert_2"] = l2
                result.loc[mask, "label_source_detail"] = (
                    "annotasi baru (label_pakar_1 + label_pakar_2)"
                )

            audit_rows.append({
                "sumber": "expert_baru",
                "expert_no": "",
                "review_id": rid,
                "match_method": "review_id_exact",
                "match_detail": "",
                "expert_label": as_label(l1) + (
                    f" | annotator2: {as_label(l2)}"
                    if as_label(l2) else ""
                ),
                "label_akhir": as_label(l1),
            })

        kappa_1_2 = cohen_kappa(
            [a for a, _ in pairs_12], [b for _, b in pairs_12]
        )
        kappa_lama_vs_baru = cohen_kappa(
            [a for a, _ in pairs_lama_baru], [b for _, b in pairs_lama_baru]
        )

    # Label akhir: experts menang, kalau tidak ada dipakai lexicon v2.
    has_expert = result["sentiment_label_expert"].astype(str).str.strip() != ""
    result["sentiment_label_final"] = result["label_lexicon_v2"]
    result.loc[has_expert, "sentiment_label_final"] = result.loc[
        has_expert, "sentiment_label_expert"
    ]

    audit_df = pd.DataFrame(audit_rows)

    # ------------------------------------------------------------------
    # Tulis keluaran.
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    drop_cols = [c for c in result.columns if c.startswith("_")]
    out_df = result.drop(columns=drop_cols)

    out_df.to_csv(
        output_path, index=False, encoding="utf-8-sig",
        sep=",", quoting=csv.QUOTE_MINIMAL,
    )
    if not audit_df.empty:
        audit_df.to_csv(
            AUDIT_FILE, index=False, encoding="utf-8-sig",
            sep=",", quoting=csv.QUOTE_MINIMAL,
        )
    if mapping_rows:
        pd.concat(mapping_rows, ignore_index=True).to_csv(
            MAPPING_FILE, index=False, encoding="utf-8-sig", sep=",",
        )

    summary = {
        "stage3_file": str(stage3_path),
        "expert_lama": counters_lama,
        "expert_baru": counters_baru,
        "output_file": str(output_path),
        "baris_dataset": int(len(result)),
        "distribusi_label_akhir": result[
            "sentiment_label_final"
        ].value_counts().to_dict(),
        "distribusi_label_source": result[
            "label_source"
        ].value_counts().to_dict(),
        "jumlah_berlabel_expert": int(has_expert.sum()),
        "cohen_kappa_annotator1_vs_annotator2": kappa_1_2,
        "cohen_kappa_expert_lama_vs_annotasi_baru": kappa_lama_vs_baru,
        "annotator_independence_verified": False,
        "catatan_metodologi": (
            "Label experts (lama + baru) dipetakan ke reviewId. "
            "Label akhir memakai experts bila tersedia, lexicon v2 bila tidak. "
            "Prefix/fuzzy memerlukan tinjauan manual. Independensi anotator "
            "dan paparan holdout historis belum diverifikasi."
        ),
    }
    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return result, audit_df, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrasi label experts (lama + baru) via reviewId."
    )
    parser.add_argument("--stage3", type=Path, default=DEFAULT_STAGE3)
    parser.add_argument("--expert-lama", type=Path, default=DEFAULT_EXPERT_LAMA)
    parser.add_argument("--expert-baru", type=Path, default=DEFAULT_EXPERT_BARU)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-unfilled-annotation",
        action="store_true",
        help=(
            "Izinkan integrasi dijalankan walau worksheet anotasi baru "
            "belum terisi (hanya untuk pengujian teknis)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 78)
    print("INTEGRASI LABEL PAKAR - VERSI BARU (Tahap 2)")
    print("=" * 78)
    print(f"[STAGE3] {args.stage3}")
    print(f"[LAMA  ] {args.expert_lama}")
    print(f"[BARU  ] {args.expert_baru}")
    print(f"[OUTPUT] {args.output}")
    print()

    _, _, summary = integrate(
        stage3_path=args.stage3,
        expert_lama_path=args.expert_lama,
        expert_baru_path=args.expert_baru,
        output_path=args.output,
        allow_unfilled_annotation=args.allow_unfilled_annotation,
    )

    print("[LABEL AKHIR]")
    print(f"  {summary['distribusi_label_akhir']}")
    print(f"[SUMBER LABEL]")
    print(f"  {summary['distribusi_label_source']}")
    print(f"  Baris berlabel experts : {summary['jumlah_berlabel_expert']}")
    print()
    print("[COHEN'S KAPPA]")
    print(f"  annotator1 vs annotator2 : "
          f"{summary['cohen_kappa_annotator1_vs_annotator2']['kappa']} "
          f"(n={summary['cohen_kappa_annotator1_vs_annotator2']['n']})")
    print(f"  expert lama vs baru      : "
          f"{summary['cohen_kappa_expert_lama_vs_annotasi_baru']['kappa']} "
          f"(n={summary['cohen_kappa_expert_lama_vs_annotasi_baru']['n']})")
    print()
    print(f"[OUTPUT] {args.output}")
    print("=" * 78)


if __name__ == "__main__":
    main()
