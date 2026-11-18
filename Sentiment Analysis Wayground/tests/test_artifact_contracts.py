"""Regression checks for dataset versions and experiment checkpoints."""

import json

import pandas as pd
import pytest

from shared import final_config, splitting
from shared.experiment_artifacts import EXPERIMENT_JSON_FILES


@pytest.fixture
def rows():
    return pd.DataFrame({
        "reviewId": [str(i) for i in range(30)],
        "processed_text": [f"teks {i}" for i in range(30)],
        "sentiment_label_final": ["Positif", "Netral", "Negatif"] * 10,
    })


@pytest.mark.parametrize("mutation", ["extra", "text", "label", "duplicate_id"])
def test_saved_split_cannot_hide_changed_dataset(monkeypatch, rows, mutation):
    dev, hold = rows.iloc[:24].copy(), rows.iloc[24:].copy()
    monkeypatch.setattr(splitting, "load_canonical_split", lambda **kw: (dev, hold, {}))
    changed = rows.copy()
    if mutation == "extra":
        changed.loc[30] = ["30", "new review", "Positif"]
    elif mutation == "text":
        changed.loc[0, "processed_text"] = changed.loc[24, "processed_text"]
    elif mutation == "label":
        changed.loc[0, "sentiment_label_final"] = "Negatif"
    else:
        changed.loc[0, "reviewId"] = changed.loc[1, "reviewId"]
    actual_dev, actual_hold, info = splitting.split_parts_for_dataset(changed)
    assert info["source"] == "recomputed"
    assert len(actual_dev) + len(actual_hold) == len(changed)
    splitting.assert_no_overlap(actual_dev, actual_hold)


def test_saved_split_accepts_reordered_rows_and_label_case(monkeypatch, rows):
    dev, hold = rows.iloc[:24].copy(), rows.iloc[24:].copy()
    monkeypatch.setattr(splitting, "load_canonical_split", lambda **kw: (dev, hold, {}))
    changed = rows.sample(frac=1, random_state=42)
    changed["sentiment_label_final"] = changed.sentiment_label_final.str.lower()
    actual_dev, actual_hold, info = splitting.split_parts_for_dataset(changed)
    assert info["source"] == "stage5_files"
    assert set(actual_dev.reviewId) == set(dev.reviewId)
    assert set(actual_hold.reviewId) == set(hold.reviewId)


def test_config_validation_handles_bom_delimiter_and_label_case(tmp_path, monkeypatch, rows):
    path = tmp_path / "canonical.csv"
    rows.to_csv(path, sep=";", encoding="utf-8-sig", index=False)
    monkeypatch.setattr(final_config, "DATASET_RELATIVE_PATH", path)
    monkeypatch.setattr(final_config, "EXPECTED_DATASET_ROWS", len(rows))
    monkeypatch.setattr(final_config, "EXPECTED_CLASS_DISTRIBUTION", {
        "positif": 10, "netral": 10, "negatif": 10,
    })
    final_config.validate_config()
    rows.loc[0, "processed_text"] = " "
    rows.to_csv(path, sep=";", encoding="utf-8-sig", index=False)
    with pytest.raises(ValueError, match="preprocessing kosong"):
        final_config.validate_config()


def test_checkpoint_checks_current_files_and_missing_experiments(tmp_path, monkeypatch):
    from pipeline import run_pipeline as runner

    source, checkpoint = tmp_path / "experiments", tmp_path / "checkpoint"
    source.mkdir()
    monkeypatch.setattr(runner, "EXPERIMENTS_DIR", source)
    monkeypatch.setattr(runner, "STAGE6_DIR", checkpoint)
    monkeypatch.setattr(runner, "STAGE6_SUMMARY", checkpoint / "summary.json")
    name = EXPERIMENT_JSON_FILES[3]
    (source / name).write_text('{"version": 1}', encoding="utf-8")
    first = runner.stage6_experiment_summary(False)
    assert first["status"] == "incomplete"
    assert EXPERIMENT_JSON_FILES[1] in first["missing_artifacts"]
    assert (checkpoint / name).exists()
    (source / name).write_text('{"version": 2}', encoding="utf-8")
    second = runner.stage6_experiment_summary(False)
    assert second["artifacts"][name]["version"] == 2
    assert first["provenance"][name]["sha256"] != second["provenance"][name]["sha256"]
    (source / name).write_text('[]', encoding="utf-8")
    third = runner.stage6_experiment_summary(False)
    assert name in third["invalid_artifacts"]
    assert json.loads((checkpoint / "summary.json").read_text())["status"] == "incomplete"
