import json

import pandas as pd
import pytest

from scripts.improve_sentiment import annotations, fresh_reviews, paired_groups
from shared.candidate_sentiment import aspect_clauses, candidate_weak_label, preprocess_candidate
from shared.preprocessing import preprocess_text


def test_candidate_preserves_negation_and_ignores_topic_polarity():
    assert "tidak_bisa" in preprocess_candidate("tdk bs login").split()
    assert candidate_weak_label(preprocess_candidate("tidak ada masalah")) != "negatif"
    assert candidate_weak_label(preprocess_candidate("Guru memberi soal")) == "netral"
    assert "buruk" not in preprocess_candidate("menyenangkan tidak").split()
    text = "Aplikasi sangat membantu belajar"
    assert preprocess_candidate(text, preprocess_text(text)) == preprocess_candidate(text)


def test_contrast_clause_keeps_topic_and_technical_text_separate():
    aspects = aspect_clauses("Soalnya bagus, tetapi sering error.")
    assert aspects["Materi/Konten"] == ["Soalnya bagus"]
    assert aspects["Performa Teknis"] == ["sering error."]


def test_groups_include_transitive_duplicates():
    groups = paired_groups(["a", "a", "c", "d"], ["z", "y", "y", "x"])
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_fresh_reviews_exclude_previous_period_and_duplicate_text(tmp_path):
    canonical = pd.DataFrame({"reviewId": ["1"], "content": ["Aplikasi ini bagus"]})
    raw = pd.DataFrame({
        "reviewId": ["2", "3", "4", "5"],
        "at": ["2026-07-01", "2026-06-30", "2026-07-02", "2026-08-01"],
        "content": ["APLIKASI INI BAGUS!", "Kuis menarik", "Aplikasi ini bagus sekali", "Tidak dapat masuk"],
        "language_keep": [True, True, True, True],
    })
    path = tmp_path / "raw.csv"
    raw.to_csv(path, index=False)
    assert fresh_reviews(path, canonical)["reviewId"].tolist() == ["5"]


def test_final_evaluation_requires_complete_labels_from_the_frozen_set(tmp_path):
    path = tmp_path / "annotation.csv"
    pd.DataFrame({"reviewId": ["a", "b", "c"], "label": ["negatif", "netral", ""]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Lengkapi"):
        annotations(path, {"a", "b", "c"}, require_all=True)
    with pytest.raises(ValueError, match="di luar"):
        annotations(path, {"a", "b"})


def test_final_evaluation_uses_frozen_text_and_compares_same_reviews(tmp_path, monkeypatch):
    from scripts import improve_sentiment as experiment

    labels = ["negatif", "netral", "positif"]
    source = pd.DataFrame({"reviewId": ["a", "b", "c"], "content": labels})
    source.to_csv(tmp_path / "sumber_uji_baru.csv", index=False)
    sheet = tmp_path / "anotasi.csv"
    source.assign(content="ignore this edit", label=labels,
                  **dict.fromkeys(experiment.ASPECT_COLUMNS.values(), "")).to_csv(sheet, index=False)

    class EchoModel:
        def predict(self, texts):
            return list(texts)

    monkeypatch.setattr(experiment, "load_verified_pipeline", lambda: (EchoModel(), {"model_version": "test"}))
    monkeypatch.setattr(experiment.joblib, "load", lambda path: EchoModel())
    monkeypatch.setattr(experiment, "preprocess_text", lambda text: text)

    experiment.evaluate(tmp_path, sheet, "karakter_saja", "benchmark_awal")

    result = json.loads((tmp_path / "evaluasi_uji_baru.json").read_text(encoding="utf-8"))
    assert result["rows"] == 3
    assert result["v4"]["macro_f1"] == result["candidate_metrics"]["macro_f1"] == 1.0
    with pytest.raises(FileExistsError):
        experiment.evaluate(tmp_path, sheet, "karakter_saja", "benchmark_awal")
