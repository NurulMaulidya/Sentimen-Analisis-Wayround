import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from shared.experiment_contract import snapshot_sizes


def test_reused_holdout_keeps_historical_exposure(tmp_path):
    from shared.constants import BASE_DIR
    from shared.deployment_integrity import evaluation_provenance

    path = BASE_DIR / "output_v4/research_run/run_manifest.json"
    history = json.loads(path.read_text(encoding="utf-8-sig"))
    exposed = history["invalidation"]["exposed_review_ids"][0]
    holdout = tmp_path / "holdout.csv"
    holdout.write_text(f"reviewId\n{exposed}\nnew-review\n", encoding="utf-8")

    provenance = evaluation_provenance({}, holdout)

    assert provenance["historical_exposed_holdout_rows"] == 1
    assert provenance["exposed_review_ids"] == [exposed]


def test_snapshot_contract_rejects_modified_split(tmp_path):
    names = ["stage3_final_labeled.csv", "stage5_split_v2/development.csv", "stage5_split_v2/holdout.csv"]
    hashes = {}
    for name in names:
        p = tmp_path / name
        p.parent.mkdir(exist_ok=True)
        p.write_text("original")
        hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path / "dataset_manifest.json").write_text(json.dumps({"output_hashes": hashes,
        "rows": {"final": 30}, "split": {"development_rows": 24, "holdout_rows": 6}}))
    assert snapshot_sizes(tmp_path / names[0]) == (30, 24, 6)
    (tmp_path / names[1]).write_text("different split")
    with pytest.raises(ValueError, match="changed"):
        snapshot_sizes(tmp_path / names[0])


def test_ratio_diagnostic_never_trains_on_canonical_holdout(monkeypatch):
    from scripts.experiments import experiment_1_split as exp

    frame = pd.DataFrame({"reviewId": [str(i) for i in range(60)],
                          "processed_text": [f"teks {i}" for i in range(60)],
                          "sentiment_label_final": ["Positif", "Netral", "Negatif"] * 20})
    dev, hold = frame.iloc[:45].copy(), frame.iloc[45:].copy()
    monkeypatch.setattr(exp, "split_parts_for_dataset", lambda *a, **kw: (dev, hold, {}))
    seen = []

    def scores(pipe, x, y, **kwargs):
        assert not set(x) & set(hold.processed_text)
        assert len(kwargs["groups"]) == len(x)
        assert type(kwargs["cv"]).__name__ == "StratifiedGroupKFold"
        seen.append(len(x))
        return np.array([0.7, 0.8, 0.9, 0.8, 0.7])

    monkeypatch.setattr(exp, "cross_val_score", scores)
    monkeypatch.setattr(exp, "save_json", lambda *a: None)
    base = json.loads(exp.DEFAULT_PREVIOUS_CONFIG.read_text())
    result = exp.run_split_experiment(frame, "processed_text", "sentiment_label_final",
                                     base["tfidf"], base["svm"], exp.DEFAULT_PREVIOUS_CONFIG)
    assert len(seen) == 3
    assert result["configuration"]["canonical_holdout_rows_excluded"] == 15
    assert result["configuration"]["final_split"].startswith("80%:20%")
    assert result["results"][0]["cv_f1_macro_std"] == pytest.approx(np.std([0.7, 0.8, 0.9, 0.8, 0.7], ddof=1))


def test_tfidf_loader_keeps_review_ids(monkeypatch, tmp_path):
    from scripts.experiments import experiment_3_tfidf as exp

    path = tmp_path / "data.csv"
    pd.DataFrame({"reviewId": ["a", "b", "c"], "processed_text": ["bagus", "jelek", "apa"],
                  "sentiment_label_final": ["Positif", "Negatif", "Netral"]}).to_csv(path, index=False)
    monkeypatch.setattr(exp, "DATASET_PATH", path)
    monkeypatch.setattr(exp, "EXPECTED_DATASET_ROWS", 3)
    frame, _, _ = exp.load_canonical_dataset()
    assert frame.reviewId.tolist() == ["a", "b", "c"]


def test_smote_neighbors_do_not_change_between_experiments():
    from scripts.experiments import experiment_5_class_imbalance as e5
    from scripts.experiments import experiment_6_ovo_ovr as e6
    from scripts.experiments import experiment_7_regularization as e7

    tfidf = {"max_features": 2000, "ngram_range": [1, 1], "sublinear_tf": True,
             "min_df": 1, "max_df": 0.9}
    svm = {"kernel": "linear", "C": 1.0, "gamma": "scale", "class_weight": None}
    p5 = e5.build_estimator(tfidf, svm, "smote")
    p6 = e6.build_pipeline(tfidf, svm, "smote", "ovr")
    p7 = e7.build_pipeline(tfidf, svm, "smote", "ovr", 1.0)
    for pipe in [p5, p6, p7]:
        assert pipe.named_steps["sampler"].k_neighbors == 3
        assert pipe.named_steps["sampler"].random_state == 42


def test_incomplete_run_cannot_be_declared_final(tmp_path, monkeypatch):
    from scripts.select_research_configuration import select

    monkeypatch.setenv("WAYGROUND_OUTPUT_DIR", "test-before-call")
    directory = tmp_path / "research_run"
    directory.mkdir()
    (directory / "run_manifest.json").write_text(json.dumps({"status": "running", "experiments": []}))
    with pytest.raises(ValueError, match="seven experiments"):
        select(tmp_path)
    assert not (tmp_path / "experiments/selected_configuration.json").exists()


def test_svm_grid_includes_baseline_c_and_has_128_combinations():
    from scripts.experiments import experiment_4_svm as exp

    assert 10.0 in exp.C_VALUES
    assert exp.calculate_total_experiments() == 128


def test_resume_rejects_changed_completed_artifact(tmp_path, monkeypatch):
    from scripts import run_research_experiments as runner

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    preparation, selected, output = tmp_path / "data/prep", tmp_path / "data/selected", tmp_path / "out"
    for folder in [preparation, selected]:
        folder.mkdir(parents=True)
        (folder / "dataset_manifest.json").write_text("{}")
    logs = output / "research_run"
    logs.mkdir(parents=True)
    (tmp_path / "changed.json").write_text("new content")
    manifest = {"status": "failed", "preparation_manifest_sha256": runner.digest(preparation / "dataset_manifest.json"),
                "selected_manifest_sha256": runner.digest(selected / "dataset_manifest.json"),
                "experiments": [{"artifact": "changed.json", "artifact_sha256": "stale-hash"}]}
    path = logs / "run_manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Completed experiment changed"):
        runner.run(preparation, selected, output, resume=True)
    assert json.loads(path.read_text())["status"] == "failed"


def test_common_loader_preserves_bom_review_id(tmp_path):
    from scripts.experiments.common import load_dataset

    path = tmp_path / "reviews.csv"
    pd.DataFrame({"reviewId": ["a"], "processed_text": ["bagus"],
                  "sentiment_label_final": ["Positif"]}).to_csv(path, sep=";", encoding="utf-8-sig", index=False)
    frame, _ = load_dataset(path)
    assert "reviewId" in frame
    assert "\ufeffreviewId" not in frame
