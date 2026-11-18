"""
Kontrak environment, ekuivalensi preprocessing, dan asal sentimen aspek.

Tiga hal yang diuji di sini:

1. ENVIRONMENT
   Environment aktif dibandingkan dengan environment penelitian terkunci
   (requirements-lock.txt). Test ini TIDAK menuntut environment aktif
   sudah terkunci - backend dan website boleh jalan di environment lain.
   Yang diuji adalah bahwa deteksi drift bekerja dan gerbang training
   menolak environment yang tidak terkunci.

2. EKUIVALENSI PREPROCESSING
   `processed_text` pada dataset canonical adalah input TF-IDF saat
   training. Kalau preprocessing canonical berubah, kolom itu tidak lagi
   merepresentasikan teks yang masuk ke model, dan artefak final tidak
   lagi bisa dipercaya. Test ini memastikan preprocessing saat ini masih
   mereproduksi kolom itu untuk seluruh baris.

3. ASAL SENTIMEN ASPEK
   Sentimen sebuah aspek harus dihitung dari kalimat-kalimat aspek itu
   sendiri memakai artefak SVM canonical, BUKAN disalin dari label
   sentimen tingkat ulasan. Test ini memakai kasus yang sengaja
   bertentangan: satu ulasan dengan dua aspek yang sentimennya berbeda.
"""
from __future__ import annotations

import csv
import io

import pytest

from shared.environment import (
    LOCKED_VERSIONS,
    EnvironmentDriftError,
    active_versions,
    environment_drift,
    require_locked_environment,
)


# ---------------------------------------------------------------------------
# 1. KONTRAK ENVIRONMENT
# ---------------------------------------------------------------------------

def test_locked_versions_match_the_canonical_metadata():
    """Versi terkunci harus sama dengan yang tercatat saat training."""
    import json
    from shared.final_config import FINAL_METADATA_PATH

    metadata = json.loads(FINAL_METADATA_PATH.read_text(encoding="utf-8-sig"))
    versions = metadata["versions"]

    assert versions["scikit_learn"] == LOCKED_VERSIONS["scikit-learn"]
    assert versions["numpy"] == LOCKED_VERSIONS["numpy"]
    assert versions["pandas"] == LOCKED_VERSIONS["pandas"]
    assert versions["python"] == LOCKED_VERSIONS["python"]


def test_drift_report_lists_every_locked_library():
    drift = environment_drift()
    listed = {item["library"] for item in drift}

    assert listed == set(LOCKED_VERSIONS)
    for item in drift:
        assert item["locked"] == LOCKED_VERSIONS[item["library"]]
        assert item["severity"] in {"ok", "blocking", "informational"}


def test_drift_is_detected_and_gate_blocks_unlocked_environment(monkeypatch):
    """Environment yang meleset harus terdeteksi dan memblokir training."""
    monkeypatch.setenv("WAYGROUND_ALLOW_ENV_DRIFT", "")

    # Simulasikan environment yang meleset: versi numpy dipalsukan.
    import shared.environment as environment

    real_active = environment.active_versions

    def fake_active():
        versions = dict(real_active())
        versions["numpy"] = "0.0.1-not-a-real-version"
        return versions

    monkeypatch.setattr(environment, "active_versions", fake_active)

    blocking = [i for i in environment.environment_drift() if i["severity"] == "blocking"]
    assert any(item["library"] == "numpy" for item in blocking)

    with pytest.raises(EnvironmentDriftError) as excinfo:
        environment.require_locked_environment()

    message = str(excinfo.value)
    assert "numpy" in message
    assert "requirements-lock.txt" in message


def test_override_allows_training_but_still_reports_drift(monkeypatch):
    """Override boleh meloloskan training, tetapi drift tetap tercatat."""
    import shared.environment as environment

    real_active = environment.active_versions

    def fake_active():
        versions = dict(real_active())
        versions["pandas"] = "0.0.1-not-a-real-version"
        return versions

    monkeypatch.setattr(environment, "active_versions", fake_active)
    monkeypatch.setenv("WAYGROUND_ALLOW_ENV_DRIFT", "1")

    returned = environment.require_locked_environment()
    assert returned["pandas"] == "0.0.1-not-a-real-version"

    blocking = [i for i in environment.environment_drift() if i["severity"] == "blocking"]
    assert any(item["library"] == "pandas" for item in blocking)


def test_active_versions_report_python_and_numeric_libraries():
    versions = active_versions()

    for library in ("python", "numpy", "pandas", "scikit-learn"):
        assert library in versions
        assert versions[library] and versions[library] != "tidak diketahui"


def test_training_script_has_the_environment_gate():
    """Gerbang environment harus benar-benar ada di script training."""
    from pathlib import Path

    from shared.final_config import PROJECT_ROOT

    source = (PROJECT_ROOT / "scripts" / "train_final_model.py").read_text(encoding="utf-8")
    assert "require_locked_environment" in source


# ---------------------------------------------------------------------------
# 2. EKUIVALENSI PREPROCESSING CANONICAL
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_current_preprocessing_reproduces_canonical_processed_text():
    """
    Preprocessing canonical saat ini harus sama dengan kolom `processed_text`
    pada dataset canonical.

    Kolom inilah yang menjadi input TF-IDF saat training. Selisih sekecil
    apa pun berarti artefak final tidak lagi merepresentasikan teks yang
    masuk ke model, dan seluruh hasil evaluasi kehilangan makna.
    """
    import pandas as pd

    from shared.constants import CANONICAL_DATASET_PATH
    from shared.final_config import TEXT_COLUMN
    from shared.preprocessing import preprocess_text

    df = pd.read_csv(CANONICAL_DATASET_PATH, encoding="utf-8-sig")
    source_column = "content" if "content" in df.columns else None
    assert source_column, "Dataset canonical harus menyimpan kolom teks sumber."

    mismatches = []
    for index, row in df.iterrows():
        expected = str(row[TEXT_COLUMN])
        actual = preprocess_text(str(row[source_column]))
        if actual != expected:
            mismatches.append((index, expected, actual))

    assert not mismatches, (
        f"{len(mismatches)} baris tidak cocok dengan processed_text canonical. "
        f"Contoh: {mismatches[:3]}"
    )


def test_dataset_manifest_hash_matches_current_dataset():
    """Manifest snapshot harus masih cocok dengan dataset canonical."""
    import hashlib
    import json

    from shared.constants import CANONICAL_DATASET_PATH
    from shared.deployment_integrity import sha256

    manifest_path = CANONICAL_DATASET_PATH.parent / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))

    for name, expected in manifest["output_hashes"].items():
        target = CANONICAL_DATASET_PATH.parent / name
        assert target.exists(), f"Berkas snapshot hilang: {name}"
        assert sha256(target) == expected, f"Isi snapshot berubah: {name}"


# ---------------------------------------------------------------------------
# 3. ASAL SENTIMEN ASPEK
# ---------------------------------------------------------------------------

@pytest.fixture
def api():
    from backend.app import app
    from backend.database_models import db
    from tests.conftest import PRODUCTION_DB
    from pathlib import Path

    app.config["TESTING"] = True
    with app.app_context():
        assert Path(db.engine.url.database).resolve() != PRODUCTION_DB.resolve()
        db.drop_all()
        db.create_all()
    return app.test_client()


def file_payload(rows, headers=("content", "label")):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(headers)
    writer.writerows(rows)
    return io.BytesIO(stream.getvalue().encode("utf-8"))


def test_aspect_sentiment_is_not_copied_from_the_review_label(api):
    """
    Sentimen aspek harus berasal dari kalimat aspeknya sendiri.

    Ulasan contoh menyebut dua aspek dengan nada berbeda dalam satu
    kalimat, sehingga label sentimen tingkat ulasan tidak boleh dipakai
    untuk kedua aspek.
    """
    review = (
        "Materi pembelajarannya sangat lengkap dan membantu. "
        "Aplikasi ini sering error dan lambat sekali."
    )
    response = api.post(
        "/api/v1/analysis/upload",
        data={"file": (file_payload([(review, "")]), "aspect.csv")},
    )
    assert response.status_code == 200
    sid = response.json["analysis"]["id"]

    for stage in ("deduplicate", "preprocess"):
        assert api.post(f"/api/v1/analysis/{sid}/{stage}").status_code == 200
    assert api.post(f"/api/v1/analysis/{sid}/tfidf").status_code == 200
    assert api.post(f"/api/v1/analysis/{sid}/predict").status_code == 200

    response = api.post(f"/api/v1/analysis/{sid}/aspects")
    assert response.status_code == 200, response.json

    aspects = api.get(f"/api/v1/analysis/{sid}/aspects").json
    details = aspects["data"]
    assert details, "Ulasan contoh harus menghasilkan minimal satu aspek."

    sentiments = {row["aspect_name"]: row["sentiment"] for row in details}
    assert all(value in {"positif", "netral", "negatif"} for value in sentiments.values())

    # Setiap sentimen aspek harus sama dengan prediksi SVM atas kalimat
    # aspek itu sendiri, bukan label tingkat ulasan.
    from shared.deployment_integrity import load_verified_pipeline
    from shared.preprocessing import preprocess_text

    pipeline, _ = load_verified_pipeline()
    for row in details:
        joined = " ".join(row["sentences"])
        assert joined.strip(), "Baris aspek harus menyimpan kalimatnya."
        expected = str(pipeline.predict([preprocess_text(joined)])[0])
        assert row["sentiment"] == expected

    # Kalau ada dua aspek dengan sentimen berbeda, keduanya harus benar
    # diambil dari kalimatnya sendiri, bukan disalin dari satu label.
    if len(set(sentiments.values())) > 1:
        assert len(sentiments) >= 2


def test_single_prediction_reports_the_same_aspect_scope(api):
    """Prediksi satu ulasan harus memakai skope sentimen aspek yang sama."""
    response = api.post(
        "/api/v1/predict-single",
        json={"text": "Materi pembelajarannya sangat membantu. Aplikasinya sering error."},
    )
    assert response.status_code == 200, response.json
    payload = response.json

    assert payload["aspect_sentiment_scope"] == "per_aspect_sentiment_from_aspect_sentences"
    assert payload["detected_aspects"]
    for detail in payload["aspect_details"].values():
        assert detail["sentiment"] in {"positif", "netral", "negatif"}


def test_review_sentiment_and_aspect_sentiment_are_reported_separately(api):
    """
    Sentimen ulasan dan sentimen aspek harus dua hal yang berbeda.

    Kalau keduanya sama untuk semua kasus, berarti sentimen aspek selalu
    disalin dari label ulasan - yang bertentangan dengan definisi ABSA.
    """
    rows = [
        ("Materi pembelajarannya sangat lengkap. Aplikasinya sering error.", ""),
        ("Aplikasi lambat sekali tapi materinya bagus.", ""),
        ("Aplikasi berjalan normal seperti biasa.", ""),
    ]
    response = api.post(
        "/api/v1/analysis/upload",
        data={"file": (file_payload(rows), "scope.csv")},
    )
    assert response.status_code == 200
    sid = response.json["analysis"]["id"]

    for stage in ("deduplicate", "preprocess"):
        assert api.post(f"/api/v1/analysis/{sid}/{stage}").status_code == 200
    assert api.post(f"/api/v1/analysis/{sid}/tfidf").status_code == 200
    assert api.post(f"/api/v1/analysis/{sid}/predict").status_code == 200
    assert api.post(f"/api/v1/analysis/{sid}/aspects").status_code == 200

    predictions = api.get(f"/api/v1/analysis/{sid}/predictions").json["data"]
    aspects = api.get(f"/api/v1/analysis/{sid}/aspects").json["data"]

    review_labels = {row["id"]: row["predicted_label"] for row in predictions}
    assert review_labels

    aspect_pairs = [
        (review_labels[row["review_id"]], row["sentiment"])
        for row in aspects
        if row["sentiment"]
    ]
    assert aspect_pairs, "Harus ada pasangan sentimen ulasan dan sentimen aspek."

    # Tidak semua pasangan boleh identik: kalau semuanya identik, sentimen
    # aspek jelas disalin dari label ulasan.
    assert any(review != aspect for review, aspect in aspect_pairs), (
        "Semua sentimen aspek identik dengan sentimen ulasan; "
        "perlakuan sentimen aspek kemungkinan menyalin label ulasan."
    )
