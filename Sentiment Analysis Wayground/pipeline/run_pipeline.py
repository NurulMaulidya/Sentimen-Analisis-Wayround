"""
WAYGROUND DATA PIPELINE - STAGE 1 TO STAGE 8

Alur:
1. Scraping / validasi data awal
2. Preprocessing 7 tahap
3. Initial labeling InSet
4. Aspect detection detail dari dataset canonical
5. Train/holdout split
6. Model experiment summary / configuration check
7. Train final model + evaluasi holdout
8. Final aspect analysis

Catatan:
- Stage 3 tetap menghasilkan stage3_labeled.csv.
- Dataset canonical stage3_final_labeled.csv tetap menjadi sumber utama
  untuk split, model final, dan analisis aspek.
- Holdout dibuat sekali secara stratified dan disimpan.
- Stage 7 memanggil scripts/train_final_model.py sehingga konfigurasi final
  tetap mengikuti output lock_final_config.py.
- Script ini tidak mengubah best_config.json secara otomatis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================================
# PROJECT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.constants import (  # noqa: E402
    AUDIT_DIR,
    EXPERIMENTS_DIR,
    FINAL_MODEL_DIR,
    PIPELINE_OUTPUT_DIR,
)
from shared.experiment_artifacts import EXPERIMENT_JSON_FILES

# Folder keluaran pipeline dapat di-override (Tahap 5) supaya artefak lama
# di data/pipeline_output/ tidak tertimpa sebelum hasil baru lolos
# pemeriksaan.
OUT = PIPELINE_OUTPUT_DIR

STAGE1 = OUT / "stage1_scraped.csv"
STAGE2 = OUT / "stage2_preprocessed_new.csv"
STAGE3 = OUT / "stage3_labeled.csv"
STAGE3_FINAL = OUT / "stage3_final_labeled.csv"
STAGE4 = OUT / "stage4_aspects.csv"

# Split v2 (Tahap 3): berkelompok berdasarkan processed_text, tanpa
# kebocoran. Folder lama (stage5_split) dibiarkan utuh sebagai pembanding
# sampai hasil baru lolos pemeriksaan.
STAGE5_DIR = OUT / "stage5_split_v2"
STAGE5_DEV = STAGE5_DIR / "development.csv"
STAGE5_HOLDOUT = STAGE5_DIR / "holdout.csv"
STAGE5_META = STAGE5_DIR / "split_metadata.json"

STAGE6_DIR = OUT / "stage6_experiments"
STAGE6_SUMMARY = STAGE6_DIR / "experiment_summary.json"

STAGE7_DIR = OUT / "stage7_final"
STAGE8_DIR = OUT / "stage8_aspect_analysis"

DEFAULT_INPUT = PROJECT_ROOT / "data" / "02_data_akhir.csv"
PREPROCESSING_EXCEL = PROJECT_ROOT / "data" / "03_preprocessing_stages.xlsx"

RANDOM_STATE = 42
TEST_SIZE = 0.20
LABEL_COLUMN = "sentiment_label_final"
# Kolom yang dipakai sebagai grup pada split berkelompok (Tahap 3).
GROUP_COLUMN = "processed_text"

SENTIMENT_CLASSES = {"Positif", "Netral", "Negatif"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("wayground_pipeline")


# ============================================================================
# GENERIC HELPERS
# ============================================================================

def read_auto(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    return pd.read_csv(
        path,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
    )


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        index=False,
        sep=",",
        encoding="utf-8-sig",
    )


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def run_python_script(script: Path, description: str) -> None:
    """Jalankan script project sebagai subprocess."""
    if not script.exists():
        raise FileNotFoundError(
            f"Script untuk {description} tidak ditemukan: {script}"
        )

    log.info("=" * 80)
    log.info("%s", description)
    log.info("SCRIPT: %s", script)
    log.info("=" * 80)

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{description} gagal dengan exit code {result.returncode}: {script}"
        )


def validate_final_dataset(df: pd.DataFrame) -> None:
    required = {
        "reviewId",
        "processed_text",
        LABEL_COLUMN,
    }

    missing = required - set(df.columns)

    if missing:
        raise KeyError(
            "Dataset canonical tidak lengkap. "
            f"Kolom hilang: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Dataset canonical kosong.")

    empty_text = (
        df["processed_text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    empty_label = (
        df[LABEL_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    if empty_text.any():
        raise ValueError(
            f"Dataset canonical memiliki {int(empty_text.sum())} "
            "processed_text kosong."
        )

    if empty_label.any():
        raise ValueError(
            f"Dataset canonical memiliki {int(empty_label.sum())} "
            "label final kosong."
        )

    labels = set(
        df[LABEL_COLUMN]
        .dropna()
        .astype(str)
        .str.strip()
    )

    invalid = labels - SENTIMENT_CLASSES

    if invalid:
        raise ValueError(
            f"Label final tidak valid: {sorted(invalid)}"
        )


# ============================================================================
# STAGE 1
# ============================================================================

def stage1(input_path: Path, force: bool) -> pd.DataFrame:
    from shared.data_contract import filter_research_period

    if STAGE1.exists() and not force:
        log.info("Stage 1 memakai cache: %s", STAGE1)
        cached, excluded = filter_research_period(read_auto(STAGE1))
        if not excluded.empty:
            raise ValueError("Stage 1 cache contains dates outside research period; rebuild it.")
        return cached

    df = read_auto(input_path)

    if "content" not in df.columns:
        raise KeyError(
            "Kolom 'content' tidak ditemukan pada input scraping."
        )

    before = len(df)

    df = df[
        df["content"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    ].copy()

    df, excluded = filter_research_period(df)
    if not excluded.empty:
        save_csv(excluded, OUT / "stage1_excluded_period.csv")
    df = df.reset_index(drop=True)
    save_csv(df, STAGE1)

    log.info("STAGE 1: %d -> %d baris", before, len(df))
    return df


# ============================================================================
# STAGE 2
# ============================================================================

def stage2(df: pd.DataFrame, force: bool) -> pd.DataFrame:
    if STAGE2.exists() and not force:
        log.info("Stage 2 memakai cache: %s", STAGE2)
        return read_auto(STAGE2)

    # Gunakan preprocessing project yang sudah ada.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from shared.preprocessing import preprocess_text

    result = df.copy()

    result["processed_text"] = (
        result["content"]
        .fillna("")
        .astype(str)
        .apply(preprocess_text)
    )

    result = result[
        result["processed_text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    ].copy()

    result = result.reset_index(drop=True)
    save_csv(result, STAGE2)

    log.info("STAGE 2: %d baris", len(result))
    return result


def export_preprocessing_excel(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """Ekspor hasil preprocessing ke Excel tanpa mengubah dataset utama."""

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from shared.preprocessing import preprocess_text

    rows = []

    for _, row in df.iterrows():
        original = str(row.get("content", "") or "")

        final_text, steps = preprocess_text(
            original,
            return_steps=True,
        )

        record = {
            "reviewId": row.get("reviewId"),
            "Tanggal": row.get("at"),
            "Versi Aplikasi": row.get("appVersion"),
            "Rating": row.get("score", row.get("rating")),
            "Ulasan Awal": original,
        }

        mapping = [
            ("1_cleaning", "1. Cleaning"),
            ("2_case_folding", "2. Case Folding"),
            ("3_tokenizing", "3. Tokenizing"),
            ("4_normalization", "4. Normalisasi Slang"),
            ("5_negation", "5. Negation Handling"),
            ("6_stopwords", "6. Stopword Removal"),
            ("7_stemming", "7. Stemming"),
        ]

        for key, label in mapping:
            value = steps.get(key, "")
            if isinstance(value, list):
                value = " ".join(map(str, value))
            record[label] = value

        record["Hasil Akhir"] = final_text
        rows.append(record)

    sheet = pd.DataFrame(rows)

    summary = []
    stage_columns = [
        ("1_cleaning", "1. Cleaning"),
        ("2_case_folding", "2. Case Folding"),
        ("3_tokenizing", "3. Tokenizing"),
        ("4_normalization", "4. Normalisasi Slang"),
        ("5_negation", "5. Negation Handling"),
        ("6_stopwords", "6. Stopword Removal"),
        ("7_stemming", "7. Stemming"),
    ]

    def count_tokens(value) -> int:
        return len(str(value).split()) if value else 0

    for i, (_, label) in enumerate(stage_columns):
        before_label = (
            stage_columns[i - 1][1]
            if i > 0
            else "Ulasan Awal"
        )

        before_tokens = sheet[before_label].map(count_tokens)
        after_tokens = sheet[label].map(count_tokens)

        summary.append({
            "Tahap": label,
            "Jumlah Baris": len(sheet),
            "Total Token": int(after_tokens.sum()),
            "Rata-rata Token": round(float(after_tokens.mean()), 2),
            "Token Hilang": int((before_tokens - after_tokens).sum()),
        })

    path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        sheet.to_excel(
            writer,
            sheet_name="Tahapan Preprocessing",
            index=False,
        )
        pd.DataFrame(summary).to_excel(
            writer,
            sheet_name="Ringkasan",
            index=False,
        )

    log.info("Laporan preprocessing: %s", path)


# ============================================================================
# STAGE 3
# ============================================================================

def stage3(df: pd.DataFrame, force: bool) -> pd.DataFrame:
    if STAGE3.exists() and not force:
        log.info("Stage 3 memakai cache: %s", STAGE3)
        return read_auto(STAGE3)

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # Jalur tunggal pelabelan: shared.labeling.
    # Fungsi yang sama dipakai scripts/audit_oov.py.
    from shared.labeling import (
        DEFAULT_NEUTRAL_THRESHOLD,
        get_active_lexicon_version,
        label_text,
    )

    lexicon_version = get_active_lexicon_version()
    log.info(
        "Stage 3 memakai lexicon versi %s (threshold %.2f)",
        lexicon_version,
        DEFAULT_NEUTRAL_THRESHOLD,
    )

    results = (
        df["processed_text"]
        .fillna("")
        .astype(str)
        .apply(
            lambda text: label_text(
                text,
                threshold=DEFAULT_NEUTRAL_THRESHOLD,
            )
        )
    )

    out = df.copy()

    out["sentiment_label_inset"] = results.map(
        lambda r: r["label"]
    )

    out["sentiment_score"] = results.map(
        lambda r: r.get(
            "raw_score",
            r.get("score", 0.0),
        )
    )

    out["sentiment_score_normalized"] = results.map(
        lambda r: r.get("normalized_score", 0.0)
    )

    out["sentiment_label"] = out["sentiment_label_inset"]
    # Penanda sumber label harus mencerminkan lexicon yang benar-benar
    # dipakai (Tahap 1: jalur pelabelan tunggal). Nilai lama "InSet"
    # menyesatkan karena hasil Stage 3 sudah memakai lexicon v2.
    out["labeling_method"] = f"Lexicon {lexicon_version}"
    out["label_source"] = f"Lexicon {lexicon_version}"
    out["expert_validation_status"] = "Tidak tervalidasi"
    out["sentiment_label_expert"] = pd.NA
    out["sentiment_label_final"] = out["sentiment_label_inset"]

    invalid = (
        set(out["sentiment_label_final"].dropna())
        - SENTIMENT_CLASSES
    )

    if invalid:
        raise ValueError(
            f"Label tidak valid: {sorted(invalid)}"
        )

    save_csv(out, STAGE3)

    log.info("STAGE 3: %d baris dilabeli InSet", len(out))
    log.info(
        "Distribusi:\n%s",
        out["sentiment_label_inset"].value_counts(),
    )

    return out


# ============================================================================
# STAGE 3.5 - INTEGRASI LABEL PAKAR
# ============================================================================

def integrate_expert_labels(force: bool) -> None:
    """
    Membangun dataset canonical: label experts (lama) > lexicon v2.

    Memanggil pipeline/integrate_expert_labels.py sebagai subprocess
    supaya hanya ada SATU implementasi integrasi (Tahap 1: jalur tunggal).
    """

    script = PROJECT_ROOT / "pipeline" / "integrate_expert_labels.py"

    if not script.exists():
        raise FileNotFoundError(
            f"Script integrasi tidak ditemukan: {script}"
        )

    cmd = [
        sys.executable,
        str(script),
        "--stage3", str(STAGE3),
        "--output", str(STAGE3_FINAL),
        # Worksheet anotasi baru belum terisi (Tahap 2 tertahan).
        "--allow-unfilled-annotation",
    ]

    log.info("STAGE 3.5: integrasi label pakar -> %s", STAGE3_FINAL)

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.stdout:
        log.info(result.stdout.rstrip())

    if result.returncode != 0:
        raise RuntimeError(
            "Integrasi label pakar gagal:\n"
            f"{result.stdout}\n{result.stderr}"
        )

    if not STAGE3_FINAL.exists():
        raise RuntimeError(
            f"Integrasi label pakar selesai tetapi {STAGE3_FINAL} tidak ada."
        )


# ============================================================================
# STAGE 4
# ============================================================================

def stage4(force: bool) -> pd.DataFrame:
    """
    Aspect detection selalu menggunakan dataset canonical:
    stage3_final_labeled.csv
    """

    if STAGE4.exists() and not force:
        log.info("Stage 4 memakai cache: %s", STAGE4)
        return read_auto(STAGE4)

    if not STAGE3_FINAL.exists():
        raise FileNotFoundError(
            f"Dataset canonical tidak ditemukan: {STAGE3_FINAL}\n\n"
            "Jalankan integrasi label pakar terlebih dahulu:\n"
            "python pipeline/integrate_expert_labels.py"
        )

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from pipeline.aspect_detection import detect_all_aspects

    canonical = read_auto(STAGE3_FINAL)
    validate_final_dataset(canonical)

    rows = []

    for _, row in canonical.iterrows():
        content = str(row.get("content", "") or "")
        detected = detect_all_aspects(content)

        for aspect, sentences in detected.items():
            if aspect == "Umum" or not sentences:
                continue

            rows.append({
                "reviewId": row.get("reviewId"),
                "aspect": aspect,
                "aspect_text": " ".join(sentences).strip(),
                "content": content,
                "processed_text": row.get("processed_text"),
                "sentiment_label_inset": row.get(
                    "sentiment_label_inset"
                ),
                "sentiment_label_expert": row.get(
                    "sentiment_label_expert"
                ),
                "sentiment_label_final": row.get(
                    "sentiment_label_final"
                ),
            })

    out = pd.DataFrame(rows)

    if not out.empty:
        out = (
            out.drop_duplicates(
                subset=[
                    "reviewId",
                    "aspect",
                    "aspect_text",
                ],
                keep="first",
            )
            .reset_index(drop=True)
        )

    save_csv(out, STAGE4)

    log.info(
        "STAGE 4: %d baris aspek dari canonical",
        len(out),
    )

    return out


# ============================================================================
# STAGE 5 - SPLIT
# ============================================================================

def stage5_split(force: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Membuat development/holdout split **berkelompok** dari canonical dataset.

    Perubahan Tahap 3:
    - Dulu dipakai `train_test_split(stratify=...)` per baris, sehingga
      52 `processed_text` identik muncul di development DAN holdout
      (kebocoran data).
    - Sekarang dipakai `shared.splitting.grouped_stratified_split`:
      grup = `processed_text`, jadi setiap teks utuh hanya ada di satu
      split. Stratifikasi dijaga semampunya.

    Holdout tidak digunakan untuk pemilihan parameter.
    """

    if (
        STAGE5_DEV.exists()
        and STAGE5_HOLDOUT.exists()
        and STAGE5_META.exists()
        and not force
    ):
        log.info("Stage 5 memakai cache.")
        return (
            read_auto(STAGE5_DEV),
            read_auto(STAGE5_HOLDOUT),
        )

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from shared.splitting import (
        assert_no_overlap,
        grouped_stratified_split,
    )

    canonical = read_auto(STAGE3_FINAL)
    validate_final_dataset(canonical)

    development, holdout, info = grouped_stratified_split(
        canonical,
        group_col=GROUP_COLUMN,
        label_col=LABEL_COLUMN,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Penjaga: tidak boleh ada processed_text yang sama di kedua split.
    assert_no_overlap(development, holdout, group_col=GROUP_COLUMN)

    save_csv(development, STAGE5_DEV)
    save_csv(holdout, STAGE5_HOLDOUT)

    metadata = {
        "stage": 5,
        "version": "v2 (berkelompok, tanpa kebocoran processed_text)",
        "holdout_used_for_selection": False,
        **info,
    }

    save_json(metadata, STAGE5_META)

    log.info(
        "STAGE 5: development=%d | holdout=%d | grup tumpang tindih=%d",
        len(development),
        len(holdout),
        info["grup_tumpang_tindih"],
    )

    return development, holdout


# ============================================================================
# STAGE 6 - EXPERIMENT SUMMARY
# ============================================================================

def stage6_experiment_summary(force: bool) -> dict:
    """
    Stage 6 tidak menjalankan ulang seluruh eksperimen secara otomatis.
    Tahap ini mengumpulkan artifact eksperimen yang sudah tersedia agar
    pipeline_output mempunyai checkpoint Stage 6 yang jelas.
    """

    STAGE6_DIR.mkdir(parents=True, exist_ok=True)

    # Always inspect current sources: a cached summary can outlive a run.
    experiment_files = [*EXPERIMENT_JSON_FILES.values(), "best_config.json"]

    artifacts = {}
    provenance = {}
    missing = []
    invalid = []

    for filename in experiment_files:
        source = EXPERIMENTS_DIR / filename

        if source.exists():
            try:
                payload = json.loads(
                    source.read_text(encoding="utf-8-sig")
                )
                if not isinstance(payload, dict) or not payload:
                    raise ValueError("Expected a non-empty JSON object")
                artifacts[filename] = payload
                provenance[filename] = {
                    "source": str(source),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
                shutil.copy2(source, STAGE6_DIR / filename)
            except (ValueError, UnicodeError) as exc:
                invalid.append(filename)
                artifacts[filename] = {
                    "status": "invalid",
                    "error": str(exc),
                }
        else:
            missing.append(filename)
            artifacts[filename] = {
                "status": "not_found",
                "expected": str(source),
            }

    summary = {
        "stage": 6,
        "description": "Experiment artifacts and final configuration checkpoint",
        "status": "incomplete" if missing or invalid else "collected",
        "validation_scope": "JSON presence and syntax only; not model selection validation",
        "missing_artifacts": missing,
        "invalid_artifacts": invalid,
        "provenance": provenance,
        "artifacts": artifacts,
    }

    save_json(summary, STAGE6_SUMMARY)

    log.info("STAGE 6: artifact eksperimen dikumpulkan.")
    return summary


# ============================================================================
# STAGE 7 - FINAL MODEL
# ============================================================================

def stage7_final(force: bool) -> None:
    """
    Melatih final model menggunakan scripts/train_final_model.py.

    best_config.json harus sudah tersedia.
    """

    required_config = EXPERIMENTS_DIR / "best_config.json"

    if not required_config.exists():
        raise FileNotFoundError(
            "best_config.json belum tersedia.\n"
            "Jalankan terlebih dahulu:\n"
            "python scripts/experiments/lock_final_config.py"
        )

    # Jangan retrain kalau artifact final sudah ada dan force=False.
    final_model = FINAL_MODEL_DIR / "final_pipeline.joblib"

    if final_model.exists() and not force:
        log.info(
            "STAGE 7 memakai final model yang sudah ada: %s",
            final_model,
        )
    else:
        run_python_script(
            PROJECT_ROOT
            / "scripts"
            / "train_final_model.py",
            "STAGE 7 - TRAIN FINAL MODEL",
        )

    STAGE7_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = [
        FINAL_MODEL_DIR / "final_pipeline.joblib",
        FINAL_MODEL_DIR / "metadata.json",
        EXPERIMENTS_DIR / "final_model_evaluation.json",
        EXPERIMENTS_DIR / "final_classification_report.json",
        EXPERIMENTS_DIR / "final_confusion_matrix.csv",
        EXPERIMENTS_DIR / "final_holdout_predictions.csv",
    ]

    copied = []

    for source in artifacts:
        if source.exists():
            destination = STAGE7_DIR / source.name
            shutil.copy2(source, destination)
            copied.append(str(destination))

    audit_script = (
        PROJECT_ROOT
        / "scripts"
        / "final_pipeline_audit.py"
    )

    if audit_script.exists():
        run_python_script(
            audit_script,
            "STAGE 7 - FINAL PIPELINE AUDIT",
        )

    save_json(
        {
            "stage": 7,
            "final_model": str(final_model),
            "artifacts": copied,
            "audit_executed": audit_script.exists(),
        },
        STAGE7_DIR / "stage7_final_summary.json",
    )

    log.info(
        "STAGE 7 selesai. Artifact final: %d file.",
        len(copied),
    )


# ============================================================================
# STAGE 8 - ASPECT ANALYSIS
# ============================================================================

def stage8_aspect_analysis(force: bool) -> None:
    """
    Menjalankan analisis aspek berdasarkan stage4_aspects.csv.
    Script yang tersedia di project akan dipakai jika ditemukan.
    """

    STAGE8_DIR.mkdir(parents=True, exist_ok=True)

    candidates = [
        PROJECT_ROOT / "pipeline" / "per_aspect_analysis.py",
        PROJECT_ROOT / "scripts" / "evaluasi_per_aspek.py",
    ]

    executed = []

    for script in candidates:
        if script.exists():
            run_python_script(
                script,
                f"STAGE 8 - {script.name}",
            )
            executed.append(str(script))

    # Kumpulkan artifact aspek yang sudah dihasilkan.
    source_candidates = [
        OUT / "stage4_aspects.csv",
        OUT / "stage4_aspects_summary.xlsx",
        OUT / "aspect_sentiment_summary.json",
        OUT / "per_aspect_results.json",
        OUT / "per_aspect_summary.csv",
        OUT / "per_aspect_evaluation_summary.csv",
        OUT / "per_aspect_evaluation.json",
        OUT / "expert_comparison.csv",
        OUT / "labeling_validation_report.json",
        OUT / "pipeline_report.txt",
    ]

    copied = []

    for source in source_candidates:
        if source.exists():
            destination = STAGE8_DIR / source.name
            shutil.copy2(source, destination)
            copied.append(str(destination))

    save_json(
        {
            "stage": 8,
            "description": "Final aspect analysis",
            "scripts_executed": executed,
            "artifacts": copied,
        },
        STAGE8_DIR / "stage8_aspect_analysis_summary.json",
    )

    log.info(
        "STAGE 8 selesai. Artifact aspek: %d file.",
        len(copied),
    )


# ============================================================================
# MAIN
# ============================================================================

def run(
    input_path: str | None = None,
    force: bool = False,
    stop_after: int = 8,
) -> None:

    source = (
        Path(input_path)
        if input_path
        else DEFAULT_INPUT
    )

    log.info("=" * 90)
    log.info("WAYGROUND DATA PIPELINE - STAGE 1 TO STAGE 8")
    log.info("Input: %s", source)
    log.info("Force: %s", force)
    log.info("=" * 90)

    # Stage 1
    d1 = stage1(source, force)

    if stop_after <= 1:
        return

    # Stage 2
    d2 = stage2(d1, force)

    if stop_after <= 2:
        return

    export_preprocessing_excel(
        d2,
        PREPROCESSING_EXCEL,
    )

    # Stage 3
    d3 = stage3(d2, force)

    if stop_after <= 3:
        return

    # ------------------------------------------------------------------
    # Canonical dataset (Stage 3.5 - integrasi label pakar)
    # ------------------------------------------------------------------
    #
    # Tahap 5: canonical dataset WAJIB dibangun ulang dari Stage 3 +
    # label experts (288), karena Stage 3 memakai lexicon v2 dan label
    # experts punya prioritas lebih tinggi.
    #
    # Worksheet anotasi baru (Tahap 2) belum terisi, sehingga integrasi
    # dijalankan dengan --allow-unfilled-annotation. Guard di
    # integrate_expert_labels.py tetap menolak integrasi bila pemanggil
    # tidak secara eksplisit mengizinkan hal itu.
    #
    if force or not STAGE3_FINAL.exists():
        integrate_expert_labels(force)

    if not STAGE3_FINAL.exists():
        raise FileNotFoundError(
            "\nDataset canonical belum tersedia:\n"
            f"{STAGE3_FINAL}\n\n"
            "Stage 1-3 berhasil.\n"
            "Jalankan integrasi label pakar:\n\n"
            "    python pipeline/integrate_expert_labels.py\n\n"
            "Kemudian lanjutkan pipeline dengan:\n\n"
            "    python pipeline/run_pipeline.py --from-stage 4\n"
        )

    # Stage 4
    d4 = stage4(force)

    if stop_after <= 4:
        return

    # Stage 5
    stage5_split(force)

    if stop_after <= 5:
        return

    # Stage 6
    stage6_experiment_summary(force)

    if stop_after <= 6:
        return

    # Stage 7
    stage7_final(force)

    if stop_after <= 7:
        return

    # Stage 8
    stage8_aspect_analysis(force)

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------

    print("\n" + "=" * 90)
    print("WAYGROUND DATA PIPELINE - STAGE 1 TO STAGE 8 SELESAI")
    print("=" * 90)

    print(f"Stage 1 : {len(d1):,} baris")
    print(f"Stage 2 : {len(d2):,} baris")
    print(f"Stage 3 : {len(d3):,} baris")
    print(f"Stage 4 : {len(d4):,} aspect rows")

    if STAGE5_DEV.exists():
        dev = read_auto(STAGE5_DEV)
        holdout = read_auto(STAGE5_HOLDOUT)
        print(f"Stage 5 : development={len(dev):,} | holdout={len(holdout):,}")

    print("\nOutput:")
    print(f"  Stage 1 : {STAGE1}")
    print(f"  Stage 2 : {STAGE2}")
    print(f"  Stage 3 : {STAGE3}")
    print(f"  Canonical: {STAGE3_FINAL}")
    print(f"  Stage 4 : {STAGE4}")
    print(f"  Stage 5 : {STAGE5_DIR}")
    print(f"  Stage 6 : {STAGE6_DIR}")
    print(f"  Stage 7 : {STAGE7_DIR}")
    print(f"  Stage 8 : {STAGE8_DIR}")
    print(f"  Excel   : {PREPROCESSING_EXCEL}")

    print("\n" + "=" * 90)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wayground Data Pipeline Stage 1-8"
    )

    parser.add_argument(
        "--input",
        default=None,
        help=(
            "CSV input. Default: "
            "data/02_data_akhir.csv"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate stage yang memiliki cache.",
    )

    parser.add_argument(
        "--from-stage",
        type=int,
        default=1,
        choices=range(1, 9),
        help=(
            "Mulai dari stage tertentu. "
            "Contoh: --from-stage 4"
        ),
    )

    parser.add_argument(
        "--stop-after",
        type=int,
        default=8,
        choices=range(1, 9),
        help=(
            "Berhenti setelah stage tertentu. "
            "Contoh: --stop-after 5"
        ),
    )

    args = parser.parse_args()

    # Untuk menjalankan ulang mulai Stage 4, jangan mengulang Stage 1-3.
    if args.from_stage > 1:
        if not STAGE3_FINAL.exists():
            raise FileNotFoundError(
                f"Canonical dataset belum ditemukan: {STAGE3_FINAL}"
            )

        if args.from_stage <= 4:
            stage4(args.force)

        if args.from_stage <= 5:
            stage5_split(args.force)

        if args.from_stage <= 6:
            stage6_experiment_summary(args.force)

        if args.from_stage <= 7:
            stage7_final(args.force)

        if args.from_stage <= 8:
            stage8_aspect_analysis(args.force)

        print("\nPIPELINE SELESAI.")
    else:
        run(
            input_path=args.input,
            force=args.force,
            stop_after=args.stop_after,
        )
