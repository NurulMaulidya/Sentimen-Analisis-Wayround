"""Deployment identity, failure handling and API/model consistency."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from shared import deployment_integrity as integrity
from shared import final_config


@pytest.fixture(scope="module")
def bundle():
    return integrity.load_verified_pipeline()


@pytest.fixture(scope="module")
def api():
    # Set this before importing the app: never create/modify the user's DB.
    from backend import config
    config.DATABASE_URI = "sqlite:///:memory:"
    from backend import app as module
    module.app.config["TESTING"] = True
    return module


def test_bundle_parameters_and_saved_predictions_reproduce(bundle):
    pipeline, metadata = bundle
    frame = pd.read_csv(final_config.FINAL_PREDICTIONS_PATH, encoding="utf-8-sig")
    assert len(frame) == 604
    assert np.array_equal(pipeline.predict(frame.processed_text), frame.predicted_sentiment)
    assert metadata["evaluation_provenance"]["historical_exposed_holdout_rows"] == 527
    assert not metadata["evaluation_provenance"]["holdout_pristine"]
    assert metadata["model"]["multiclass"]["strategy"] == "ovo"
    assert metadata["labeling"]["neutral_threshold"] == 1.0


def test_changed_parameter_is_rejected(bundle):
    pipeline = copy.deepcopy(bundle[0])
    pipeline.named_steps["classifier"].estimators_[0].C = 1.5
    with pytest.raises(ValueError, match="Fitted SVM differs"):
        integrity.verify_fitted_parameters(pipeline, final_config.FINAL_MODEL_CONFIG)


def test_changed_binary_checksum_is_rejected(tmp_path, bundle):
    metadata = copy.deepcopy(bundle[1])
    key = str(final_config.FINAL_PIPELINE_PATH.relative_to(final_config.PROJECT_ROOT))
    metadata["artifact_hashes"][key] = "incorrect"
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))
    with pytest.raises(integrity.DeploymentIntegrityError, match="artifact changed"):
        integrity.load_verified_pipeline(metadata_path=path)


def test_metadata_cannot_claim_another_model_version(tmp_path, bundle):
    metadata = copy.deepcopy(bundle[1])
    metadata["model_version"] = "wrong-version"
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))
    with pytest.raises(integrity.DeploymentIntegrityError, match="version mismatch"):
        integrity.load_verified_pipeline(metadata_path=path)


def test_changed_preprocessing_fingerprint_is_rejected(monkeypatch):
    monkeypatch.setattr(integrity, "runtime_fingerprint", lambda: {"changed": "changed"})
    with pytest.raises(integrity.DeploymentIntegrityError, match="Preprocessing/resources"):
        integrity.load_verified_pipeline()


def test_training_refuses_to_overwrite_deployment():
    from scripts.train_final_model import main

    before = integrity.sha256(final_config.FINAL_PIPELINE_PATH)
    with pytest.raises(FileExistsError, match="overwrite"):
        main()
    assert integrity.sha256(final_config.FINAL_PIPELINE_PATH) == before


def test_api_reports_one_verified_version(api, bundle):
    client = api.app.test_client()
    health = client.get("/api/v1/health")
    config = client.get("/api/v1/config")
    stats = client.get("/api/v1/model/stats")
    assert health.status_code == config.status_code == stats.status_code == 200
    version = bundle[1]["model_version"]
    assert health.json["identity"]["model_version"] == config.json["model_version"] == stats.json["stats"]["model_version"] == version
    assert stats.json["stats"]["confusion_matrix_labels"] == ["negatif", "netral", "positif"]
    assert sum(map(sum, stats.json["stats"]["confusion_matrix"])) == 604
    assert config.json["tfidf"]["max_features"] == 5000


def test_single_analyze_and_predict_use_same_model(api, bundle):
    from shared.preprocessing import preprocess_text

    client = api.app.test_client()
    text = "Aplikasi error tidak bisa login"
    expected = bundle[0].predict([preprocess_text(text)])[0]
    single = client.post("/api/v1/predict-single", json={"text": text})
    predict = client.post("/api/v1/predict", json={"text": text})
    analyze = client.post("/api/v1/analyze", json={"text": text})
    assert single.status_code == predict.status_code == analyze.status_code == 200
    assert single.json["predicted_label"] == predict.json["results"][0]["predicted_label"] == analyze.json["results"][0]["predicted_label"] == expected
    assert single.json["model_version"] == bundle[1]["model_version"]


@pytest.mark.parametrize("path", ["/api/v1/health", "/api/v1/config", "/api/v1/model/stats", "/api/v1/predict-single", "/api/v1/evaluate"])
def test_api_fails_closed_on_invalid_bundle(api, monkeypatch, path):
    def reject(*args, **kwargs):
        raise ValueError("simulated integrity failure")
    monkeypatch.setattr(integrity, "_load_verified_pipeline", reject)
    client = api.app.test_client()
    if path.endswith("predict-single"):
        response = client.post(path, json={"text": "bagus"})
    elif path.endswith("evaluate"):
        response = client.post(path, json={"texts": ["bagus"], "labels": ["positif"]})
    else:
        response = client.get(path)
    assert response.status_code == 503


@pytest.mark.parametrize("text", [None, 123, "", "   ", "1234 !!!"])
def test_invalid_or_empty_processed_text_is_not_classified(api, text):
    response = api.app.test_client().post("/api/v1/predict-single", json={"text": text})
    assert response.status_code == 400


def test_database_history_cannot_replace_research_distribution(api):
    from backend.database_models import db, AnalysisSession, ReviewData

    with api.app.app_context():
        session = AnalysisSession(name="isolated audit", filename="memory.csv", filepath="memory",
                                  original_count=1, after_dedup_count=1, status="predicted")
        db.session.add(session)
        db.session.flush()
        db.session.add(ReviewData(analysis_id=session.id, original_content="bagus", predicted_label="positif"))
        db.session.commit()
    stats = api.app.test_client().get("/api/v1/model/stats")
    assert stats.status_code == 200
    assert stats.json["stats"]["total_reviews"] == 3078
    assert stats.json["stats"]["positive"] == 1947


def test_api_macro_metric_has_explicit_three_class_scope(api, bundle):
    from shared.preprocessing import preprocess_text

    text = "Aplikasi error tidak bisa login"
    actual = bundle[0].predict([preprocess_text(text)])[0]
    result = api.app.test_client().post("/api/v1/evaluate", json={"texts": [text], "labels": [actual]})
    assert result.status_code == 200
    assert result.json["metrics"]["accuracy"] == 1.0
    assert result.json["metrics"]["f1_macro"] == pytest.approx(1 / 3)


def test_analysis_batch_uses_same_deployment_and_saves_evaluation(api, bundle):
    from backend.database_models import db, AnalysisSession, ReviewData

    with api.app.app_context():
        session = AnalysisSession(name="batch memory audit", filename="memory.csv", filepath="memory",
                                  original_count=2, has_actual_labels=True, status="uploaded")
        db.session.add(session)
        db.session.flush()
        sid = session.id
        for text, label in [("Aplikasi error tidak bisa login", "negatif"),
                            ("Sangat membantu belajar dan mengerjakan soal", "positif")]:
            db.session.add(ReviewData(analysis_id=sid, original_content=text, actual_label=label))
        db.session.commit()
    client = api.app.test_client()
    for step in ["deduplicate", "preprocess", "predict", "evaluate"]:
        response = client.post(f"/api/v1/analysis/{sid}/{step}")
        assert response.status_code == 200
        if step == "predict":
            assert response.json["predicted"] == 2
            assert response.json["model_version"] == bundle[1]["model_version"]
    evaluation = client.get(f"/api/v1/analysis/{sid}/evaluation")
    assert evaluation.status_code == 200
    assert sum(map(sum, evaluation.json["evaluation"]["confusion_matrix"])) == 2
