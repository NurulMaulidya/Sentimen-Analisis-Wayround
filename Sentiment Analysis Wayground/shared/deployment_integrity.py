"""Verify selection, serialized model, preprocessing and evaluation as one bundle."""

import csv
import hashlib
import json
from pathlib import Path


class DeploymentIntegrityError(ValueError):
    """The deployment bundle cannot be trusted for inference."""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.loads(json.dumps(value))


def verify_selection_contract():
    from shared import final_config as config
    from shared.constants import EXPERIMENTS_DIR, CANONICAL_DATASET_PATH

    path = EXPERIMENTS_DIR / "selected_configuration.json"
    selected = json.loads(path.read_text(encoding="utf-8-sig"))
    if canonical(selected["model"]) != canonical(config.FINAL_MODEL_CONFIG):
        raise ValueError("Deployment configuration differs from the selected experiment configuration")
    if selected["labeling"]["neutral_threshold"] != config.NEUTRAL_THRESHOLD:
        raise ValueError("Deployment labeling threshold differs from selection")
    if selected["dataset"]["sha256"] != sha256(CANONICAL_DATASET_PATH):
        raise ValueError("Deployment dataset differs from selected snapshot")
    run_path = EXPERIMENTS_DIR.parent / "research_run/run_manifest.json"
    if selected["run_manifest_sha256"] != sha256(run_path):
        raise ValueError("Selected run manifest changed")
    run = json.loads(run_path.read_text())
    if run["status"] != "complete":
        raise ValueError("Experiment run is incomplete")
    for entry in run["experiments"]:
        if sha256(config.PROJECT_ROOT / entry["artifact"]) != entry["artifact_sha256"]:
            raise ValueError("Selection experiment artifact changed")
    snapshot = json.loads((CANONICAL_DATASET_PATH.parent / "dataset_manifest.json").read_text())
    for name, expected in snapshot["output_hashes"].items():
        if sha256(CANONICAL_DATASET_PATH.parent / name) != expected:
            raise ValueError(f"Selected snapshot changed: {name}")
    return selected, run


def evaluation_provenance(run, holdout_path=None):
    from shared.constants import BASE_DIR, PIPELINE_OUTPUT_DIR

    if holdout_path is None:
        holdout_path = PIPELINE_OUTPUT_DIR / "stage5_split_v2/holdout.csv"
    with Path(holdout_path).open(encoding="utf-8-sig", newline="") as handle:
        holdout_ids = {row["reviewId"] for row in csv.DictReader(handle)}

    # A new run with the same holdout IDs retains the documented v4 exposure.
    exposed = set(run.get("invalidation", {}).get("exposed_review_ids", []))
    history = BASE_DIR / "output_v4/research_run/run_manifest.json"
    if history.is_file():
        previous = json.loads(history.read_text(encoding="utf-8-sig"))
        exposed.update(previous.get("invalidation", {}).get("exposed_review_ids", []))
    exposed = sorted(exposed & holdout_ids)
    return {
        "scope": "diagnostic_holdout_with_prior_exposure",
        "holdout_pristine": False,
        "independent_gold_evaluation": False,
        "historical_exposed_holdout_rows": len(exposed),
        "exposed_review_ids": exposed,
        "corrected_selection_uses_holdout": False,
        "invalidated_scores_used_for_final_selection": False,
        "note": f"{len(exposed)} canonical holdout rows entered training/CV in invalidated split attempts. Gold labels and independence from historical lexicon development are not established. These metrics are descriptive, not an unbiased independent-test estimate.",
    }


def runtime_fingerprint():
    """
    Fingerprint sumber daya yang MENENTUKAN perilaku inferensi.

    Cakupannya sengaja dibatasi pada apa yang benar-benar masuk ke model:

      * source `shared/preprocessing.py` (7 tahap canonical)
      * source `shared/research_config.py` (versi preprocessing, ambang netral)
      * kamus `data/slang_dict.json` + `data/normalization_dict.json`
      * NILAI konstanta preprocessing di `shared/constants.py`
        (NEGATION_WORDS, ADDITIONAL_STOPWORDS, SENTIMENT_PROTECTED_WORDS,
        PREPROCESSING_STAGES)

    PERBAIKAN (Tahap 6): versi lama meng-hash SELURUH isi file
    `shared/constants.py` dan `shared/lexicon.py`. Itu membuat gate menolak
    model hanya karena ada edit komentar, perubahan path default, atau
    refactor pada jalur pelabelan - perubahan yang tidak menyentuh perilaku
    inferensi sama sekali. Akibatnya backend mati total (HTTP 503) setelah
    refactor yang secara empiris terbukti tidak mengubah output.

    fingerprint dihitung dari sumber daya yang benar-benar memengaruhi
    inferensi, sehingga:
      * perubahan komentar/path tidak mematikan deployment, dan
      * perubahan perilaku preprocessing TETAP terdeteksi (gate menolak).

    Sumber daya jalur riset (pelabelan lexicon) dipisahkan ke
    `research_fingerprint()` dan tidak lagi menjadi syarat inferensi.
    """
    from shared.constants import (
        BASE_DIR,
        SLANG_DICT_PATH,
        NORMALIZATION_DICT_PATH,
        NEGATION_WORDS,
        ADDITIONAL_STOPWORDS,
        SENTIMENT_PROTECTED_WORDS,
        PREPROCESSING_STAGES,
    )

    files = [
        BASE_DIR / "shared" / "preprocessing.py",
        BASE_DIR / "shared" / "research_config.py",
        Path(SLANG_DICT_PATH),
        Path(NORMALIZATION_DICT_PATH),
    ]

    constants = {
        "NEGATION_WORDS": sorted(NEGATION_WORDS),
        "ADDITIONAL_STOPWORDS": sorted(ADDITIONAL_STOPWORDS),
        "SENTIMENT_PROTECTED_WORDS": sorted(SENTIMENT_PROTECTED_WORDS),
        "PREPROCESSING_STAGES": list(PREPROCESSING_STAGES),
    }

    return {
        "scheme": "inference-semantic-v1",
        "files": {
            path.relative_to(BASE_DIR).as_posix(): sha256(path)
            for path in files
        },
        "constants": {
            name: hashlib.sha256(
                json.dumps(value, sort_keys=True).encode("utf-8")
            ).hexdigest()
            for name, value in constants.items()
        },
    }


def research_fingerprint():
    """
    Fingerprint sumber daya jalur RISET (pelabelan lexicon).

    Dipakai untuk audit dan re-verifikasi, bukan syarat inferensi:
    model yang sudah dilatih tidak memakai lexicon saat prediksi.
    """
    from shared.constants import BASE_DIR

    paths = [
        BASE_DIR / "shared" / "labeling.py",
        BASE_DIR / "shared" / "lexicon.py",
    ] + [
        BASE_DIR / "data" / "lexicon" / name
        for name in [
            "lexicon_positive.tsv",
            "lexicon_negative.tsv",
            "v2/lexicon_positive.tsv",
            "v2/lexicon_negative.tsv",
            "v2/phrases_positive.tsv",
            "v2/phrases_negative.tsv",
        ]
    ]

    return {
        "scheme": "research-semantic-v1",
        "files": {
            path.relative_to(BASE_DIR).as_posix(): sha256(path)
            for path in paths
        },
    }


def legacy_runtime_fingerprint():
    """
    Fingerprint lama (skema 'whole-file'): sha256 SELURUH isi 13 sumber daya.

    Dipakai hanya untuk membandingkan bundle yang belum punya
    `inference_fingerprint` (metadata lama). Bundle canonical v4 yang sudah
    di-rebind memakai `runtime_fingerprint()` (skema semantik).
    """
    from shared.constants import BASE_DIR, SLANG_DICT_PATH, NORMALIZATION_DICT_PATH

    paths = [BASE_DIR / name for name in ["shared/preprocessing.py", "shared/constants.py",
             "shared/research_config.py", "shared/labeling.py", "shared/lexicon.py"]]
    paths += [Path(SLANG_DICT_PATH), Path(NORMALIZATION_DICT_PATH)]
    paths += [BASE_DIR / "data/lexicon" / name for name in [
        "lexicon_positive.tsv", "lexicon_negative.tsv", "v2/lexicon_positive.tsv",
        "v2/lexicon_negative.tsv", "v2/phrases_positive.tsv", "v2/phrases_negative.tsv"]]
    return {
        str(path.relative_to(BASE_DIR)): sha256(path) for path in paths
    }


def effective_fingerprint(metadata):
    """
    Fingerprint yang berlaku untuk sebuah metadata bundle.

    Bundle yang sudah di-rebind memakai `inference_fingerprint` (skema
    semantik). Bundle lama memakai `runtime_fingerprint` (skema whole-file).
    """
    if metadata.get("inference_fingerprint") is not None:
        return metadata["inference_fingerprint"]
    return metadata["runtime_fingerprint"]


def verify_fitted_parameters(pipeline, expected):
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
    from sklearn.svm import SVC

    if not isinstance(pipeline, Pipeline) or list(pipeline.named_steps) != ["tfidf", "classifier"]:
        raise ValueError("Deployment requires a fitted TF-IDF/classifier pipeline")
    vectorizer = pipeline.named_steps["tfidf"]
    if not isinstance(vectorizer, TfidfVectorizer) or not hasattr(vectorizer, "vocabulary_"):
        raise ValueError("TF-IDF is not fitted")
    for key, value in expected["tfidf"].items():
        if canonical(getattr(vectorizer, key)) != canonical(value):
            raise ValueError(f"Fitted TF-IDF differs: {key}")
    classifier = pipeline.named_steps["classifier"]
    cls = OneVsOneClassifier if expected["multiclass"]["strategy"] == "ovo" else OneVsRestClassifier
    if not isinstance(classifier, cls) or not hasattr(classifier, "estimators_"):
        raise ValueError("Fitted multiclass strategy differs")
    for estimator in [classifier.estimator, *classifier.estimators_]:
        if not isinstance(estimator, SVC):
            raise ValueError("Classifier is not SVC")
        for key, value in expected["svm"].items():
            if getattr(estimator, key) != value:
                raise ValueError(f"Fitted SVM differs: {key}")
    if list(classifier.classes_) != ["negatif", "netral", "positif"]:
        raise ValueError("Model class labels/order differ")


def _load_verified_pipeline(pipeline_path=None, metadata_path=None):
    import joblib
    from shared import final_config as config

    path = Path(pipeline_path) if pipeline_path else config.FINAL_PIPELINE_PATH
    metadata_path = Path(metadata_path) if metadata_path else config.FINAL_METADATA_PATH
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if metadata.get("bundle_schema") != 1 or metadata.get("status") != "FINAL":
        raise ValueError("Unverified/legacy deployment metadata")
    if canonical(metadata["model"]) != canonical(config.FINAL_MODEL_CONFIG):
        raise ValueError("Metadata/configuration mismatch")
    if metadata["labeling"]["neutral_threshold"] != config.NEUTRAL_THRESHOLD:
        raise ValueError("Metadata threshold mismatch")
    key = str(path.resolve().relative_to(config.PROJECT_ROOT.resolve()))
    if key not in metadata["artifact_hashes"]:
        raise ValueError("Pipeline is not bound to metadata")
    for name, expected in metadata["artifact_hashes"].items():
        target = (config.PROJECT_ROOT / name).resolve()
        if not target.is_relative_to(config.PROJECT_ROOT.resolve()) or sha256(target) != expected:
            raise ValueError(f"Deployment artifact changed: {name}")
    evaluation = json.loads((config.PROJECT_ROOT / metadata["artifacts"]["evaluation"]).read_text(encoding="utf-8-sig"))
    if evaluation["model_version"] != metadata["model_version"]:
        raise ValueError("Evaluation/model version mismatch")
    if canonical(evaluation["configuration"]) != canonical(metadata["model"]):
        raise ValueError("Evaluation/model configuration mismatch")
    if evaluation["dataset"] != metadata["dataset"]:
        raise ValueError("Evaluation/model dataset mismatch")
    if metadata.get("inference_fingerprint") is not None:
        # Bundle yang sudah di-rebind (Tahap 6): skema semantik, hanya
        # sumber daya yang menentukan perilaku inferensi.
        if canonical(metadata["inference_fingerprint"]) != canonical(runtime_fingerprint()):
            raise ValueError("Preprocessing/resources differ from training")
    elif metadata["runtime_fingerprint"] != legacy_runtime_fingerprint():
        raise ValueError("Preprocessing/resources differ from training")
    pipeline = joblib.load(path)
    verify_fitted_parameters(pipeline, config.FINAL_MODEL_CONFIG)
    pipeline._deployment_model_version = metadata["model_version"]
    return pipeline, metadata


def load_verified_pipeline(pipeline_path=None, metadata_path=None):
    try:
        return _load_verified_pipeline(pipeline_path, metadata_path)
    except Exception as exc:
        raise DeploymentIntegrityError(str(exc)) from exc
