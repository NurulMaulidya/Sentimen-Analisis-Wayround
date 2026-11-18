import pandas as pd
import pytest

from shared.data_contract import filter_research_period, normalize_review_columns
from shared.labeling import label_text, DEFAULT_NEUTRAL_THRESHOLD
from shared.preprocessing import map_emojis, normalize_repeated_letters
from scripts.prepare_dataset_snapshot import attach_experts


def reviews():
    return pd.DataFrame({
        "reviewId": ["a", "b", "c", "d"],
        "tanggal": ["2023-12-31", "2024-01-01", "2026-06-30", "2026-07-01"],
        "content": ["bagus", "jelek", "biasa", "baik"],
        "rating": [5, 1, 3, 4], "appVersion": ["1"] * 4,
    })


def test_period_boundaries_and_date_alias():
    kept, excluded = filter_research_period(reviews())
    assert kept.reviewId.tolist() == ["b", "c"]
    assert excluded.reviewId.tolist() == ["a", "d"]
    assert "at" in kept and "tanggal" not in kept


@pytest.mark.parametrize("mutation", ["id", "rating", "date", "alias"])
def test_invalid_raw_contract_is_rejected(mutation):
    frame = reviews()
    if mutation == "id":
        frame.loc[0, "reviewId"] = "b"
    elif mutation == "rating":
        frame["rating"] = frame.rating.astype(float)
        frame.loc[0, "rating"] = 4.5
    elif mutation == "date":
        frame.loc[0, "tanggal"] = "not-a-date"
    else:
        frame["at"] = "2024-01-01"
    with pytest.raises(ValueError):
        normalize_review_columns(frame)


def test_thumbs_down_skin_variants_remain_negative():
    assert map_emojis("\U0001f44e\U0001f3fb").strip() == "emoji_negatif"
    assert map_emojis("\U0001f44e\U0001f3fd").strip() == "emoji_negatif"
    assert map_emojis("\U0001f44d\U0001f3fb").strip() == "emoji_positif"


def test_repetition_correction_protects_valid_words():
    assert normalize_repeated_letters("bangett error allah app saat") == "banget error allah app saat"


def test_pipeline_labeler_uses_active_lexicon_and_baseline():
    from pipeline.labeling import label_review
    from shared.final_config import NEUTRAL_THRESHOLD

    expected = label_text("bagus")
    assert expected["lexicon_version"] == "v2"
    assert expected["threshold"] == DEFAULT_NEUTRAL_THRESHOLD == NEUTRAL_THRESHOLD
    actual = label_review("bagus", processed_text="bagus")
    assert actual[0] == expected["label"]
    assert actual[1] == expected["raw_score"]


def test_nonexact_annotations_are_queued_not_applied():
    frame = pd.DataFrame({
        "reviewId": ["a", "b"], "content": ["aplikasi ini bagus sekali untuk belajar", "jelek"],
        "sentiment_label_inset": ["Positif", "Negatif"],
    })
    old = pd.DataFrame({"expert_no": ["1", "2"], "text": ["aplikasi ini bagus sekali", "jelek"], "label": ["Negatif", "Negatif"]})
    new = pd.DataFrame({"review_id": ["b"], "text": ["content berbeda"], "label_pakar_1": ["Positif"], "label_pakar_2": [None]})
    result, _, pending, _ = attach_experts(frame, old, new)
    assert result.loc[0, "sentiment_label_expert"] == ""
    assert result.loc[0, "sentiment_label_final"] == "Positif"
    assert result.loc[1, "sentiment_label_expert"] == "Negatif"
    assert set(pending.match_method) == {"content_prefix", "content_not_verified"}
    assert not result.expert_independence_verified.any()


def test_conflicting_old_labels_are_not_arbitrarily_applied():
    frame = pd.DataFrame({"reviewId": ["a"], "content": ["bagus"], "sentiment_label_inset": ["Positif"]})
    old = pd.DataFrame({"expert_no": ["1", "2"], "text": ["bagus", "bagus"], "label": ["Positif", "Negatif"]})
    new = pd.DataFrame(columns=["review_id", "text", "label_pakar_1", "label_pakar_2"])
    result, _, pending, _ = attach_experts(frame, old, new)
    assert result.loc[0, "sentiment_label_expert"] == ""
    assert "conflicting_labels" in set(pending.match_method)


def test_threshold_selection_only_receives_development(monkeypatch):
    from scripts.experiments import experiment_2_threshold as experiment

    frame = pd.DataFrame({"reviewId": list("abcdef"),
                          "processed_text": ["bagus", "buruk", "apa", "baik", "jelek", "saja"]})
    dev, hold = frame.iloc[:3].copy(), frame.iloc[3:].copy()
    monkeypatch.setattr(experiment, "split_parts_for_dataset", lambda df: (dev, hold, {"source": "test"}))
    received = []

    def validation_set(canonical_df, text_column, expert_df):
        received.extend(canonical_df.reviewId)
        return canonical_df.assign(_expert_label=["Positif", "Negatif", "Netral"])

    monkeypatch.setattr(experiment, "build_expert_validation_set", validation_set)
    monkeypatch.setattr(experiment, "save_json", lambda *args: None)
    metadata = experiment.run_threshold_experiment(frame, "processed_text", pd.DataFrame({"dummy": [1]}))
    assert received == list("abc")
    assert not set(received) & set(hold.reviewId)
    assert metadata["expert_validation"]["selection_partition"] == "development_only"
    assert metadata["expert_validation"]["lexicon"]["version"] == "v2"


@pytest.mark.parametrize("directory", ["data", "runs"])
def test_snapshot_refuses_overwrite_before_reading_inputs(tmp_path, monkeypatch, directory):
    from scripts import prepare_dataset_snapshot as snapshot

    output = tmp_path / directory / "existing"
    output.mkdir(parents=True)
    monkeypatch.setattr(snapshot, "ROOT", tmp_path)
    with pytest.raises(FileExistsError):
        snapshot.build_snapshot("missing.csv", "missing.tsv", "missing2.tsv", output)


def test_language_filter_can_rebuild_an_audited_raw_export(monkeypatch):
    from pipeline import scraper

    raw = pd.DataFrame({
        "content": ["Sangat membantu"],
        "language_detected": ["unknown"],
        "language_confidence": [0.0],
        "language_keep": [False],
    })
    monkeypatch.setattr(scraper, "detect_indonesian", lambda text: (True, "id", 1.0))

    accepted, rejected = scraper.filter_indonesian_reviews(raw)

    assert accepted["language_detected"].tolist() == ["id"]
    assert accepted["language_confidence"].tolist() == [1.0]
    assert rejected.empty
    assert raw["language_keep"].tolist() == [False]


def test_integrator_compares_old_labels_before_updates(tmp_path, monkeypatch):
    from pipeline import integrate_expert_labels as integration

    source = tmp_path / "stage3.csv"
    pd.DataFrame({"reviewId": ["a", "b"], "content": ["bagus", "jelek"],
                  "processed_text": ["bagus", "jelek"], "rating": [5, 1]}).to_csv(source, sep=";", index=False)
    old_path, new_path = tmp_path / "old.tsv", tmp_path / "new.tsv"
    old_path.touch()
    new_path.touch()
    old = pd.DataFrame({"expert_no": ["1", "2"], "text": ["bagus", "jelek"], "label": ["Positif", "Negatif"]})
    new = pd.DataFrame({"review_id": ["a", "b"], "text": ["bagus", "jelek"],
                        "label_pakar_1": ["Positif", "Positif"], "label_pakar_2": [None, None],
                        "sumber_label_pakar_1": ["manual", "manual"]})
    monkeypatch.setattr(integration, "read_expert_lama", lambda path: old)
    monkeypatch.setattr(integration, "read_expert_baru", lambda path: new)
    monkeypatch.setattr(integration, "AUDIT_DIR", tmp_path)
    for key in ["AUDIT_FILE", "MAPPING_FILE", "SUMMARY_FILE"]:
        monkeypatch.setattr(integration, key, tmp_path / (key + ".json"))
    result, audit, summary = integration.integrate(source, old_path, new_path, tmp_path / "final.csv")
    assert summary["cohen_kappa_expert_lama_vs_annotasi_baru"]["observed_agreement"] == 0.5
    assert result.loc[result.reviewId.eq("b"), "sentiment_label_expert"].iloc[0] == "Negatif"
    assert "conflicting_labels" in set(audit.match_method)
