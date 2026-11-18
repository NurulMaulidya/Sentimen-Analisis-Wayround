"""Load the verified web baseline or the explicitly selected candidate."""

import json

import joblib
from werkzeug.exceptions import BadRequest

from shared.deployment_integrity import (
    DeploymentIntegrityError, load_verified_pipeline, runtime_fingerprint, sha256,
)
from shared.final_config import FINAL_PIPELINE_PATH, PROJECT_ROOT

CANDIDATE_KEY = "karakter"
CANDIDATE_PREFIX = "kandidat-karakter-"
CANDIDATE_DIR = PROJECT_ROOT / "models/candidate_char"


def model_key_for_version(version):
    return CANDIDATE_KEY if (version or "").startswith(CANDIDATE_PREFIX) else "v4"


def load_inference_model(key="v4"):
    if key == "v4":
        pipeline, metadata = load_verified_pipeline()
        digest = metadata["artifact_hashes"][str(FINAL_PIPELINE_PATH.relative_to(PROJECT_ROOT))]
        return pipeline, metadata, digest
    if key != CANDIDATE_KEY:
        raise BadRequest("Pilihan model tidak dikenal.")
    path = CANDIDATE_DIR / "pipeline.joblib"
    try:
        metadata = json.loads((CANDIDATE_DIR / "metadata.json").read_text(encoding="utf-8"))
        digest = sha256(path)
        if digest != metadata["pipeline_sha256"] or metadata["model_version"] != CANDIDATE_PREFIX + digest[:8]:
            raise ValueError("Identitas artefak kandidat berubah.")
        if metadata["inference_fingerprint"] != runtime_fingerprint():
            raise ValueError("Preprocessing web berbeda dari preprocessing kandidat.")
        return joblib.load(path), metadata, digest
    except Exception as exc:
        raise DeploymentIntegrityError(f"Kandidat karakter tidak tersedia: {exc}") from exc
