"""
Validasi artifact model final (brief D4 §15 item 7, 8, 9, 13).

Memastikan satu-satunya artifact inference yang dipakai backend:
- lengkap & hash-consistent dengan metadata,
- konfigurasi fitted identik dengan `shared.final_config.FINAL_MODEL_CONFIG`,
- terikat pada dataset & evaluasi yang benar,
- fingerprint preprocessing/runtime masih cocok.

Exit code 0 = semua pemeriksaan lolos. Non-zero = daftar kegagalan dicetak.

Jalankan:
    python scripts/validate_final_artifact.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.deployment_integrity import (  # noqa: E402
    canonical,
    runtime_fingerprint,
    sha256,
    verify_fitted_parameters,
    verify_selection_contract,
)
from shared.final_config import (  # noqa: E402
    CANONICAL_DATASET_PATH,
    FINAL_EVALUATION_PATH,
    FINAL_METADATA_PATH,
    FINAL_MODEL_CONFIG,
    FINAL_PIPELINE_PATH,
    LABEL_COLUMN,
    NEUTRAL_THRESHOLD,
    PROJECT_ROOT as FINAL_CONFIG_ROOT,
    TEXT_COLUMN,
)


class Report:
    """Kolektor hasil pemeriksaan."""

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


def validate_artifact(report: Report) -> None:
    report.section("1. Keberadaan artifact (portable path)")

    pipeline_path = Path(FINAL_PIPELINE_PATH)
    metadata_path = Path(FINAL_METADATA_PATH)

    report.check(
        "final_pipeline.joblib ada",
        pipeline_path.exists(),
        f"tidak ditemukan: {pipeline_path}",
    )
    report.check(
        "metadata.json ada",
        metadata_path.exists(),
        f"tidak ditemukan: {metadata_path}",
    )
    if not (pipeline_path.exists() and metadata_path.exists()):
        return

    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))

    report.section("2. Bentuk metadata bundle")

    report.check(
        "bundle_schema == 1",
        metadata.get("bundle_schema") == 1,
        f"ditemukan {metadata.get('bundle_schema')!r}",
    )
    report.check(
        "status == FINAL",
        metadata.get("status") == "FINAL",
        f"ditemukan {metadata.get('status')!r}",
    )
    report.check(
        "model_version ada",
        bool(metadata.get("model_version")),
        "tidak ada model_version",
    )
    report.check(
        "holdout_used_for_selection == False",
        metadata.get("holdout_used_for_selection") is False,
        f"ditemukan {metadata.get('holdout_used_for_selection')!r}",
    )

    report.section("3. Konfigurasi terkunci")

    report.check(
        "metadata.model == FINAL_MODEL_CONFIG",
        canonical(metadata.get("model")) == canonical(FINAL_MODEL_CONFIG),
        "metadata model berbeda dari shared/final_config.py",
    )
    report.check(
        f"metadata.labeling.neutral_threshold == {NEUTRAL_THRESHOLD}",
        metadata.get("labeling", {}).get("neutral_threshold") == NEUTRAL_THRESHOLD,
        f"ditemukan {metadata.get('labeling', {}).get('neutral_threshold')!r}",
    )

    report.section("4. Integritas hash artifact (artifact_hashes)")

    hashes = metadata.get("artifact_hashes") or {}
    report.check("artifact_hashes tidak kosong", bool(hashes), "metadata tidak punya artifact_hashes")
    for relative, expected in hashes.items():
        target = FINAL_CONFIG_ROOT / relative
        if not target.exists():
            report.check(f"hash {relative}", False, "file tidak ditemukan")
            continue
        report.check(
            f"hash {relative}",
            sha256(target) == expected,
            "sha256 tidak cocok",
        )

    report.section("5. Parameter fitted pada pipeline")

    import joblib  # noqa: E402

    pipeline = joblib.load(pipeline_path)
    try:
        verify_fitted_parameters(pipeline, FINAL_MODEL_CONFIG)
        report.check("parameter fitted == FINAL_MODEL_CONFIG", True)
    except Exception as exc:  # noqa: BLE001
        report.check("parameter fitted == FINAL_MODEL_CONFIG", False, str(exc))

    classes = list(getattr(pipeline.named_steps["classifier"], "classes_", []))
    report.check(
        "urutan kelas == [negatif, netral, positif]",
        classes == ["negatif", "netral", "positif"],
        f"ditemukan {classes}",
    )
    steps = list(pipeline.named_steps)
    report.check(
        "langkah pipeline == [tfidf, classifier]",
        steps == ["tfidf", "classifier"],
        f"ditemukan {steps}",
    )

    report.section("6. Ikatan dengan dataset & evaluasi")

    dataset_path = FINAL_CONFIG_ROOT / metadata.get("dataset", {}).get("path", "")
    report.check(
        "path dataset di metadata ada",
        dataset_path.exists(),
        f"tidak ditemukan: {dataset_path}",
    )
    if dataset_path.exists():
        expected_hash = hashes.get(str(dataset_path.relative_to(FINAL_CONFIG_ROOT)), None)
        if expected_hash:
            report.check(
                "sha256 dataset cocok dengan artifact_hashes",
                sha256(dataset_path) == expected_hash,
                "sha256 dataset tidak cocok",
            )

    evaluation_path = Path(FINAL_EVALUATION_PATH)
    if report.check(
        "final_model_evaluation.json ada",
        evaluation_path.exists(),
        f"tidak ditemukan: {evaluation_path}",
    ):
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8-sig"))
        report.check(
            "evaluation.model_version == metadata.model_version",
            evaluation.get("model_version") == metadata.get("model_version"),
            "versi model tidak cocok",
        )
        report.check(
            "evaluation.configuration == metadata.model",
            canonical(evaluation.get("configuration")) == canonical(metadata.get("model")),
            "konfigurasi evaluasi tidak cocok",
        )
        report.check(
            "evaluation.dataset == metadata.dataset",
            canonical(evaluation.get("dataset")) == canonical(metadata.get("dataset")),
            "dataset evaluasi tidak cocok",
        )
        report.check(
            "evaluation.provenance menandai holdout tidak pristine",
            evaluation.get("evaluation_provenance", {}).get("holdout_pristine") is False,
            "provenance hilang atau salah tandai",
        )

    report.section("7. Fingerprint preprocessing & resource")

    report.check(
        "inference_fingerprint cocok",
        canonical(metadata.get("inference_fingerprint")) == canonical(runtime_fingerprint()),
        "fingerprint sumber daya inferensi berubah sejak re-binding terakhir",
    )

    report.section("8. Kontrak pemilihan eksperimen")

    try:
        selected, run = verify_selection_contract()
        report.check(
            "verify_selection_contract() lolos",
            True,
        )
        report.check(
            "run manifest berstatus complete",
            run.get("status") == "complete",
            f"status {run.get('status')!r}",
        )
        report.check(
            "holdout tidak dipakai untuk seleksi",
            selected.get("split", {}).get("holdout_evaluated") is False,
            "holdout_evaluated bukan False",
        )
    except Exception as exc:  # noqa: BLE001
        report.check("verify_selection_contract() lolos", False, str(exc))


def main() -> int:
    print("=" * 68)
    print("WAYGROUND — VALIDASI ARTIFACT MODEL FINAL")
    print("=" * 68)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Artifact     : {FINAL_PIPELINE_PATH}")
    print(f"Metadata     : {FINAL_METADATA_PATH}")
    print(f"Dataset      : {CANONICAL_DATASET_PATH}")
    print(f"Teks / label : {TEXT_COLUMN} / {LABEL_COLUMN}")

    report = Report()
    validate_artifact(report)

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
    raise SystemExit(main())
