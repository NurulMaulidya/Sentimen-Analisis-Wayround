"""
Validasi konsistensi pipeline end-to-end (brief D4 §15).

13 pemeriksaan wajib:
  1. Canonical dataset          8. Backend model loader
  2. Dataset row count          9. Prediction consistency
  3. Label distribution        10. No data leakage
  4. Preprocessing consistency  11. Frontend API integration
  5. TF-IDF configuration      12. Aspect analysis
  6. SVM configuration         13. Reproducibility
  7. Model artifact

Exit code 0 = semua lolos. Non-zero = daftar kegagalan dicetak.

Jalankan:
    python scripts/validate_pipeline_consistency.py
    python scripts/validate_pipeline_consistency.py --refit  # environment riset terkunci
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

# Terminal Windows default-nya cp1252 dan gagal menulis karakter non-ASCII.
# Set stream ke UTF-8 dengan fallback supaya output tidak pernah crash.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - stream tanpa reconfigure
        pass

import pandas as pd  # noqa: E402

from shared.constants import PIPELINE_OUTPUT_DIR  # noqa: E402
from shared.final_config import (  # noqa: E402
    CANONICAL_DATASET_PATH,
    EXPECTED_CLASS_DISTRIBUTION,
    EXPECTED_DATASET_ROWS,
    EXPECTED_DEVELOPMENT_ROWS,
    EXPECTED_EXPERT_VALIDATED_ROWS,
    EXPECTED_HOLDOUT_ROWS,
    FINAL_MODEL_CONFIG,
    LABEL_COLUMN,
    NEUTRAL_THRESHOLD,
    PROJECT_ROOT as FINAL_CONFIG_ROOT,
    TEXT_COLUMN,
)

SPLIT_DIR = PIPELINE_OUTPUT_DIR / "stage5_split_v2"
DEVELOPMENT_CSV = SPLIT_DIR / "development.csv"
HOLDOUT_CSV = SPLIT_DIR / "holdout.csv"
MANIFEST_JSON = PIPELINE_OUTPUT_DIR / "dataset_manifest.json"
SELECTED_JSON = FINAL_CONFIG_ROOT / "output_v4" / "experiments" / "selected_configuration.json"

VALID_LABELS = ("negatif", "netral", "positif")

SAMPLE_TEXTS = [
    "Aplikasi ini sangat membantu, materinya lengkap dan mudah diakses",
    "Sering error dan lambat, loadingnya kelamaan",
    "Saya gunakan untuk latihan harian, tidak ada catatan khusus",
]


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passed = 0

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        if ok:
            self.passed += 1
            print(f"[OK]   {name}")
        else:
            self.failures.append(f"{name}: {detail}")
            print(f"[FAIL] {name} — {detail}")
        return ok

    def section(self, title: str) -> None:
        print()
        print("=" * 68)
        print(title)
        print("=" * 68)


def _load_dataset() -> pd.DataFrame:
    return pd.read_csv(CANONICAL_DATASET_PATH, low_memory=False)


# ---------------------------------------------------------------------------
# 1-3. Dataset
# ---------------------------------------------------------------------------

def check_dataset(report: Report) -> pd.DataFrame | None:
    report.section("1-3. Canonical dataset, jumlah baris, distribusi label")

    if not report.check(
        "canonical dataset ada",
        CANONICAL_DATASET_PATH.exists(),
        f"tidak ditemukan: {CANONICAL_DATASET_PATH}",
    ):
        return None

    frame = _load_dataset()

    report.check(
        f"jumlah baris == {EXPECTED_DATASET_ROWS}",
        len(frame) == EXPECTED_DATASET_ROWS,
        f"ditemukan {len(frame)}",
    )
    report.check(
        f"kolom teks '{TEXT_COLUMN}' ada",
        TEXT_COLUMN in frame.columns,
        "kolom tidak ditemukan",
    )
    report.check(
        f"kolom label '{LABEL_COLUMN}' ada",
        LABEL_COLUMN in frame.columns,
        "kolom tidak ditemukan",
    )

    labels = frame[LABEL_COLUMN].astype("string").str.strip().str.lower()
    report.check(
        "label hanya berisi negatif/netral/positif",
        bool(labels.isin(VALID_LABELS).all()),
        "ada label kosong atau tak dikenal",
    )
    actual_distribution = {label: int((labels == label).sum()) for label in VALID_LABELS}
    report.check(
        "distribusi label sesuai kontrak final",
        actual_distribution == EXPECTED_CLASS_DISTRIBUTION,
        f"expected {EXPECTED_CLASS_DISTRIBUTION}, actual {actual_distribution}",
    )

    empty = frame[TEXT_COLUMN].fillna("").astype(str).str.strip().eq("")
    report.check("tidak ada processed_text kosong", not bool(empty.any()),
                 f"{int(empty.sum())} baris kosong")

    if "label_source" in frame.columns:
        expert_rows = int((frame["label_source"].astype(str).str.lower() == "expert").sum())
        report.check(
            f"baris validasi expert == {EXPECTED_EXPERT_VALIDATED_ROWS}",
            expert_rows == EXPECTED_EXPERT_VALIDATED_ROWS,
            f"ditemukan {expert_rows}",
        )
    if "sentiment_label_inset" in frame.columns:
        inset = frame["sentiment_label_inset"].astype("string").str.strip().str.lower()
        changed = int((inset != labels).sum())
        report.check(
            "validasi expert benar-benar mengubah label",
            changed > 0,
            "label final identik dengan label lexicon → expert tidak berpengaruh",
        )

    if MANIFEST_JSON.exists():
        manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8-sig"))
        report.check(
            "manifest mencatat holdout tidak independen",
            manifest.get("holdout_independence_verified") is False,
            "manifest tidak menandai keterbatasan holdout",
        )
    return frame


# ---------------------------------------------------------------------------
# 4. Preprocessing consistency
# ---------------------------------------------------------------------------

def check_preprocessing(report: Report) -> None:
    report.section("4. Konsistensi preprocessing (training vs prediksi)")

    from shared.preprocessing import preprocess_text

    import backend.services.preprocessing_service as service

    report.check(
        "backend preprocessing service memakai shared.preprocessing",
        getattr(service, "_canonical_preprocess", None) is preprocess_text,
        "backend tidak memakai implementasi canonical",
    )

    same = all(
        service.preprocess_full(text)["final"] == preprocess_text(text)
        for text in SAMPLE_TEXTS
    )
    report.check("hasil preprocessing backend == shared", same, "hasil berbeda")

    stages = service.STAGE_ORDER
    report.check(
        "urutan 7 tahap utuh",
        stages[:7] == ["original", "cleaning", "case_folding", "tokenizing",
                       "normalization", "negation", "stopwords"]
        and stages[7:] == ["stemming", "final"],
        f"ditemukan {stages}",
    )

    steps = preprocess_text(SAMPLE_TEXTS[1], return_steps=True)[1]
    report.check(
        "preprocess_text mengembalikan 7 tahap",
        len([k for k in steps if k[0].isdigit()]) == 7,
        f"ditemukan {len(steps)} tahap",
    )

    from shared.labeling import get_active_lexicon, get_active_lexicon_version

    report.check(
        f"lexicon aktif == v2 (threshold {NEUTRAL_THRESHOLD})",
        get_active_lexicon_version() == "v2",
        f"ditemukan {get_active_lexicon_version()!r}",
    )
    report.check(
        "objek lexicon berhasil dimuat",
        get_active_lexicon() is not None,
        "lexicon gagal dimuat",
    )


# ---------------------------------------------------------------------------
# 5-6. Konfigurasi TF-IDF & SVM
# ---------------------------------------------------------------------------

def check_configuration(report: Report) -> None:
    report.section("5-6. Konfigurasi TF-IDF dan SVM")

    tfidf = FINAL_MODEL_CONFIG["tfidf"]
    svm = FINAL_MODEL_CONFIG["svm"]

    report.check("TF-IDF max_features == 5000", tfidf["max_features"] == 5000,
                 f"ditemukan {tfidf['max_features']}")
    report.check("TF-IDF ngram_range == (1, 3)", tuple(tfidf["ngram_range"]) == (1, 3),
                 f"ditemukan {tuple(tfidf['ngram_range'])}")
    report.check("TF-IDF sublinear_tf == False", tfidf["sublinear_tf"] is False,
                 f"ditemukan {tfidf['sublinear_tf']}")
    report.check("TF-IDF min_df == 1", tfidf["min_df"] == 1, f"ditemukan {tfidf['min_df']}")
    report.check("TF-IDF max_df == 0.90", abs(float(tfidf["max_df"]) - 0.90) < 1e-9,
                 f"ditemukan {tfidf['max_df']}")

    report.check("SVM kernel == linear", svm["kernel"] == "linear", f"ditemukan {svm['kernel']}")
    report.check("SVM C == 10.0", float(svm["C"]) == 10.0, f"ditemukan {svm['C']}")
    report.check("SVM class_weight is None", svm["class_weight"] is None,
                 f"ditemukan {svm['class_weight']}")
    report.check("multiclass == ovo", FINAL_MODEL_CONFIG["multiclass"]["strategy"] == "ovo",
                 f"ditemukan {FINAL_MODEL_CONFIG['multiclass']['strategy']}")
    report.check("imbalance == none", FINAL_MODEL_CONFIG["imbalance"]["method"] == "none",
                 f"ditemukan {FINAL_MODEL_CONFIG['imbalance']['method']}")

    if SELECTED_JSON.exists():
        selected = json.loads(SELECTED_JSON.read_text(encoding="utf-8-sig"))
        selected_model = selected.get("model", {})
        report.check(
            "selected_configuration.json cocok dengan FINAL_MODEL_CONFIG",
            selected_model.get("tfidf", {}).get("max_features") == tfidf["max_features"]
            and selected_model.get("svm", {}).get("C") == svm["C"]
            and selected_model.get("multiclass", {}).get("strategy") == "ovo",
            "konfigurasi terpilih berbeda dari final_config",
        )
        report.check(
            "eksperimen 7 tercatat sebagai diagnostik (bukan selection)",
            selected.get("config_source") == "sequential_experiment_winners_3_to_6",
            f"config_source = {selected.get('config_source')!r}",
        )
        report.check(
            "hasil negatif eksperimen 7 tercatat",
            selected.get("regularization_diagnostic", {}).get("best", {}).get("gap_pass") is False,
            "gap_pass tidak ditandai False",
        )


# ---------------------------------------------------------------------------
# 7-9. Artifact, backend loader, prediction consistency
# ---------------------------------------------------------------------------

def check_artifact_and_prediction(report: Report) -> None:
    report.section("7-9. Artifact, backend loader, konsistensi prediksi")

    from shared.deployment_integrity import load_verified_pipeline

    try:
        pipeline, metadata = load_verified_pipeline()
        report.check("load_verified_pipeline() lolos", True)
        report.check(
            "pipeline punya langkah tfidf + classifier",
            list(pipeline.named_steps) == ["tfidf", "classifier"],
            f"ditemukan {list(pipeline.named_steps)}",
        )
    except Exception as exc:  # noqa: BLE001
        report.check("load_verified_pipeline() lolos", False, str(exc))
        return

    from pipeline.predict import SentimentPredictor

    predictor = SentimentPredictor()
    report.check(
        "pipeline/predict.py memakai artifact final yang sama",
        Path(pipeline_path_of(predictor)).resolve() == Path(str(FINAL_CONFIG_ROOT / "models" / "final_v4" / "final_pipeline.joblib")).resolve()
        if (pipeline_path_of(predictor) and FINAL_CONFIG_ROOT.joinpath("models", "final_v4", "final_pipeline.joblib").exists()) else True,
        "predictor memakai artifact lain",
    )

    from backend.services.prediction_service import predict_single

    results_direct = []
    results_backend = []
    for text in SAMPLE_TEXTS:
        results_direct.append(str(pipeline.predict([preprocess_canonical(text)])[0]))
        results_backend.append(str(predict_single(text)["predicted_label"]))

    report.check(
        "prediksi pipeline == prediksi backend",
        results_direct == results_backend,
        f"{results_direct} != {results_backend}",
    )
    report.check(
        "label prediksi valid",
        all(label in VALID_LABELS for label in results_backend),
        f"ditemukan {results_backend}",
    )


def pipeline_path_of(predictor) -> str:
    try:
        return str(predictor.pipeline_path)
    except Exception:  # noqa: BLE001
        return ""


def preprocess_canonical(text: str) -> str:
    from shared.preprocessing import preprocess_text

    return preprocess_text(text)


# ---------------------------------------------------------------------------
# 10. Data leakage
# ---------------------------------------------------------------------------

def check_leakage(report: Report) -> None:
    report.section("10. Data leakage")

    if not (DEVELOPMENT_CSV.exists() and HOLDOUT_CSV.exists()):
        report.check("split development/holdout tersedia", False, "file split tidak ditemukan")
        return

    development = pd.read_csv(DEVELOPMENT_CSV, low_memory=False)
    holdout = pd.read_csv(HOLDOUT_CSV, low_memory=False)

    report.check(f"development == {EXPECTED_DEVELOPMENT_ROWS} baris",
                 len(development) == EXPECTED_DEVELOPMENT_ROWS, f"ditemukan {len(development)}")
    report.check(f"holdout == {EXPECTED_HOLDOUT_ROWS} baris",
                 len(holdout) == EXPECTED_HOLDOUT_ROWS, f"ditemukan {len(holdout)}")

    for column in ("reviewId", "content", TEXT_COLUMN):
        if column not in development.columns or column not in holdout.columns:
            continue
        left = set(development[column].astype(str))
        right = set(holdout[column].astype(str))
        overlap = len(left & right)
        report.check(
            f"tidak ada overlap development↔holdout pada '{column}'",
            overlap == 0,
            f"{overlap} nilai tumpang tindih",
        )

    if SELECTED_JSON.exists():
        selected = json.loads(SELECTED_JSON.read_text(encoding="utf-8-sig"))
        split = selected.get("split", {})
        report.check("holdout tidak dipakai untuk seleksi", split.get("holdout_evaluated") is False,
                     "holdout_evaluated bukan False")
        report.check(
            "paparan holdout historis tercatat (bukan disembunyikan)",
            split.get("holdout_rows_exposed_in_invalidated_attempts", 0) > 0,
            "paparan historis tidak tercatat",
        )


# ---------------------------------------------------------------------------
# 11. Frontend API integration
# ---------------------------------------------------------------------------

def check_frontend(report: Report) -> None:
    report.section("11. Integrasi frontend ↔ backend")

    api_file = PROJECT_ROOT / "frontend" / "src" / "services" / "api.js"
    if not report.check("frontend api.js ada", api_file.exists(), "tidak ditemukan"):
        return

    source = api_file.read_text(encoding="utf-8")
    # PENTING: pola harus MENCAKUP path berparameter (template literal
    # seperti `/analysis/${id}/runs`). Versi lama berhenti di karakter `$`
    # sehingga endpoint berparameter tidak pernah diperiksa dan pemeriksaan
    # selalu lolos walau route-nya tidak ada.
    calls = set(re.findall(
        r"apiClient\.(?:get|post|delete|put)\(\s*[`'\"]([^`'\"]*)", source
    ))

    # PENTING: pakai instance app yang sudah jadi. Memanggil create_app()
    # lagi hanya menghasilkan blueprint, tanpa route yang didaftarkan di
    # level modul backend/app.py, sehingga endpoint seperti /api/v1/health
    # akan terlihat "hilang" padahal ada.
    from backend.app import app as flask_app

    rules = set()
    for rule in flask_app.url_map.iter_rules():
        if rule.endpoint == "static":
            continue
        rules.add(rule.rule)

    def normalize(path: str) -> str:
        path = path.split("?")[0].rstrip("/")
        return path if path.startswith("/") else "/" + path

    backend_rules = {normalize(rule) for rule in rules}

    def matches_backend(call: str) -> bool:
        """Cocokkan path frontend (berplaceholder) dengan rule backend."""
        full = normalize("/api/v1" + call)
        # Bagian literal di-escape, placeholder ${...} menjadi satu segmen.
        parts = re.split(r"(\$\{[^}]+\})", full)
        pattern = "".join(
            "[^/]+" if part.startswith("${") else re.escape(part) for part in parts
        )
        return any(re.fullmatch(pattern, rule) for rule in backend_rules)

    api_prefixed = {call for call in calls if call}
    missing = sorted(call for call in api_prefixed if not matches_backend(call))
    report.check(
        "semua endpoint yang dipanggil frontend ada di backend",
        not missing,
        f"tidak ditemukan: {missing}",
    )

    report.check(
        "frontend tidak memakai data dummy",
        not re.search(r"Math\.random|mockData|dummyData|fakeData", source),
        "frontend memuat data dummy",
    )
    report.check(
        "frontend tidak hardcode metrik",
        not re.search(r"(accuracy|f1_?macro)\s*[:=]\s*[0-9]{2,}", source, re.IGNORECASE),
        "frontend memuat metrik hardcoded",
    )


# ---------------------------------------------------------------------------
# 12. Aspect analysis
# ---------------------------------------------------------------------------

def check_aspect(report: Report) -> None:
    report.section("12. Analisis aspek (modul tambahan, terpisah)")

    from pipeline.aspect_detection import detect_all_aspects

    text = "Materinya lengkap tapi aplikasinya sering error dan lambat."
    detected = detect_all_aspects(text)
    found = [aspect for aspect, sentences in detected.items() if aspect != "Umum" and sentences]
    report.check("deteksi aspek menemukan minimal satu aspek", bool(found), f"ditemukan {found}")
    report.check("aspek 'Performa Teknis' terdeteksi",
                 "Performa Teknis" in found, f"ditemukan {found}")

    from shared.constants import ASPECT_KEYWORDS, ASPECTS

    report.check("kata kunci aspek berasal dari shared.constants",
                 ASPECT_KEYWORDS is not None and len(ASPECTS) == 4, "konfigurasi aspek tidak lengkap")

    backend_aspects = PROJECT_ROOT / "backend" / "services" / "aspect_service.py"
    if backend_aspects.exists():
        backend_source = backend_aspects.read_text(encoding="utf-8")
        report.check(
            "backend memakai deteksi aspek yang sama",
            "pipeline.aspect_detection" in backend_source
            or "from pipeline.aspect_detection" in backend_source,
            "backend masih punya implementasi deteksi aspek sendiri",
        )

    # Artefak analisis aspek harus berasal dari dataset canonical yang sama
    # dan memuat relasi Aspect -> jumlah review -> Positif/Netral/Negatif.
    aspect_artifact = FINAL_CONFIG_ROOT / "output_v4" / "experiments" / "aspect_analysis.json"
    if not report.check("artefak analisis aspek ada", aspect_artifact.exists(),
                        f"tidak ditemukan: {aspect_artifact}"):
        return

    payload = json.loads(aspect_artifact.read_text(encoding="utf-8"))
    dataset_path = str(payload.get("dataset", {}).get("path", ""))
    report.check(
        "analisis aspek memakai dataset canonical",
        dataset_path.replace("\\", "/").endswith("stage3_final_labeled.csv")
        and "pipeline_output_v4" in dataset_path.replace("\\", "/"),
        f"dataset pada artefak aspek: {dataset_path}",
    )
    split = payload.get("split", {})
    cv_uses_holdout = split.get("holdout_used_in_aspect_cv", split.get("holdout_used_in_this_analysis"))
    report.check("CV per aspek memakai development", cv_uses_holdout is False,
                 f"holdout dipakai untuk CV aspek: {cv_uses_holdout}")

    aspects = payload.get("aspects", {})
    incomplete = [
        name for name, entry in aspects.items()
        if not entry.get("reviews_with_aspect")
        or not entry.get("model_sentiment_distribution", {}).get("Positif")
        or not entry.get("model_sentiment_distribution", {}).get("Netral")
        or not entry.get("model_sentiment_distribution", {}).get("Negatif")
    ]
    report.check(
        "setiap aspek punya jumlah review + Positif/Netral/Negatif",
        bool(aspects) and not incomplete,
        f"aspek tidak lengkap: {incomplete}",
    )
    report.check(
        "semua aspek research dipakai di laporan",
        set(aspects) == set(ASPECTS),
        f"aspek pada artefak: {sorted(aspects)}",
    )


# ---------------------------------------------------------------------------
# 13. Reproducibility
# ---------------------------------------------------------------------------

def check_reproducibility(report: Report) -> None:
    report.section("13. Reproduabilitas prediksi")

    from shared.deployment_integrity import load_verified_pipeline

    try:
        pipeline, _ = load_verified_pipeline()
    except Exception as exc:  # noqa: BLE001
        report.check("pipeline dapat dimuat", False, str(exc))
        return

    first = [str(pipeline.predict([preprocess_canonical(t)])[0]) for t in SAMPLE_TEXTS]
    second = [str(pipeline.predict([preprocess_canonical(t)])[0]) for t in SAMPLE_TEXTS]
    report.check("prediksi deterministik (2× run identik)", first == second, f"{first} != {second}")

    from shared.preprocessing import preprocess_text

    once = [preprocess_text(t) for t in SAMPLE_TEXTS]
    twice = [preprocess_text(t) for t in SAMPLE_TEXTS]
    report.check("preprocessing deterministik", once == twice, "hasil preprocessing tidak stabil")


def check_artifact_reproducibility(report: Report) -> None:
    """
    Apakah artefak model final bisa DIREPRODUKSI dari dataset canonical +
    konfigurasi terkunci?

    Cara cek: refit manual memakai resep yang sama (TF-IDF + SVM OVO pada
    development canonical), lalu bandingkan:
      * vocabulary_ dan idf_ vectorizer
      * prediksi pada holdout canonical
      * metrik holdout

    Model harus identik. Selisih berarti artefak terpasang tidak berasal dari
    dataset + konfigurasi yang dikunci, sehingga angka evaluasinya tidak
    dapat direproduksi.
    """
    report.section("14. Reproduktivitas artefak model final")

    from shared.deployment_integrity import load_verified_pipeline

    if not DEVELOPMENT_CSV.exists():
        report.check("development canonical tersedia", False, f"tidak ditemukan: {DEVELOPMENT_CSV}")
        return

    from scripts.train_final_model import build_final_pipeline

    try:
        pipeline, _ = load_verified_pipeline()
    except Exception as exc:  # noqa: BLE001
        report.check("pipeline dapat dimuat", False, str(exc))
        return

    development = pd.read_csv(DEVELOPMENT_CSV, encoding="utf-8-sig", low_memory=False)
    holdout = pd.read_csv(HOLDOUT_CSV, encoding="utf-8-sig", low_memory=False)

    refit = build_final_pipeline()
    refit.fit(development[TEXT_COLUMN].astype(str),
              development[LABEL_COLUMN].astype(str).str.strip().str.lower())

    deployed_vectorizer = pipeline.named_steps["tfidf"]
    refit_vectorizer = refit.named_steps["tfidf"]
    same_vocabulary = deployed_vectorizer.vocabulary_ == refit_vectorizer.vocabulary_
    report.check(
        "vocabulary_ vectorizer artefak == refit resep terkunci",
        same_vocabulary,
        "vocabulary berbeda; artefak tidak berasal dari development canonical",
    )

    import numpy as np

    report.check(
        "idf_ vectorizer artefak == refit resep terkunci",
        bool(np.allclose(deployed_vectorizer.idf_, refit_vectorizer.idf_)),
        "idf berbeda; artefak tidak berasal dari development canonical",
    )

    holdout_texts = holdout[TEXT_COLUMN].astype(str)
    holdout_truth = holdout[LABEL_COLUMN].astype(str).str.strip().str.lower()
    deployed_predictions = pipeline.predict(holdout_texts)
    refit_predictions = refit.predict(holdout_texts)
    agreement = float((deployed_predictions == refit_predictions).mean())
    report.check(
        "prediksi artefak == prediksi refit resep terkunci (holdout)",
        agreement == 1.0,
        f"kesesuaian {agreement:.4%} pada {len(holdout)} baris holdout",
    )

    from sklearn.metrics import accuracy_score, f1_score

    deployed_f1 = f1_score(holdout_truth, deployed_predictions, average="macro", zero_division=0)
    refit_f1 = f1_score(holdout_truth, refit_predictions, average="macro", zero_division=0)
    report.check(
        "metrik holdout artefak == metrik refit resep terkunci",
        abs(float(deployed_f1) - float(refit_f1)) < 1e-9,
        f"artefak macro F1 {deployed_f1:.6f} vs refit {refit_f1:.6f}",
    )


def main(refit: bool = False) -> int:
    if refit:
        from shared.environment import EnvironmentDriftError, require_locked_environment

        try:
            require_locked_environment()
        except EnvironmentDriftError as exc:
            print(exc, file=sys.stderr)
            return 1
    print("=" * 68)
    print("WAYGROUND — VALIDASI KONSISTENSI PIPELINE END-TO-END")
    print("=" * 68)
    print(f"Project root    : {PROJECT_ROOT}")
    print(f"Canonical data  : {CANONICAL_DATASET_PATH}")
    print(f"Split dir       : {SPLIT_DIR}")
    print(f"Config TF-IDF   : {FINAL_MODEL_CONFIG['tfidf']}")
    print(f"Config SVM      : {FINAL_MODEL_CONFIG['svm']}")

    report = Report()
    check_dataset(report)
    check_preprocessing(report)
    check_configuration(report)
    check_artifact_and_prediction(report)
    check_leakage(report)
    check_frontend(report)
    check_aspect(report)
    check_reproducibility(report)
    if refit:
        check_artifact_reproducibility(report)

    print()
    print("=" * 68)
    if report.failures:
        print(f"HASIL: GAGAL — {len(report.failures)} pemeriksaan bermasalah, {report.passed} lolos")
        print("=" * 68)
        for item in report.failures:
            print(f"  - {item}")
        return 1
    print(f"HASIL: LOLOS — {report.passed} pemeriksaan")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refit", action="store_true", help="Bandingkan hasil training ulang di environment riset.")
    raise SystemExit(main(refit=parser.parse_args().refit))
