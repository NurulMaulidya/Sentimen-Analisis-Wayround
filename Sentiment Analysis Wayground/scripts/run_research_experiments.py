"""Run all seven experiments with development-only selection and provenance.

Order: threshold (2), derived snapshot, sensitivity (1), then model chain (3-7).
Does not train/promote the final deployment model or score the holdout.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared.experiment_artifacts import EXPERIMENT_JSON_FILES
from scripts.derive_threshold_snapshot import derive


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(preparation, selected, output, resume=False):
    preparation, selected, output = [Path(p).resolve() for p in (preparation, selected, output)]
    if not preparation.is_relative_to(ROOT) or not output.is_relative_to(ROOT):
        raise ValueError("Preparation and output must remain inside the project")
    if not any(selected.is_relative_to(ROOT / name) for name in ("data", "runs")):
        raise ValueError("Selected snapshot must be inside project data/ or runs/")
    experiments = output / "experiments"
    logdir = output / "research_run"
    experiments.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)
    manifest_path = logdir / "run_manifest.json"
    if manifest_path.exists() and not resume:
        raise FileExistsError("Run namespace exists; choose a new --output instead of overwriting history")
    manifest = {"status": "running", "order": [2, 1, 3, 4, 5, 6, 7],
                "preparation_snapshot": str(preparation.relative_to(ROOT)),
                "preparation_manifest_sha256": digest(preparation / "dataset_manifest.json"),
                "selection_metric": "mean_grouped_cv_macro_f1", "standard_deviation_ddof": 1,
                "holdout_evaluated": False, "deployment_model_trained": False, "experiments": []}

    if resume:
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] != "failed":
            raise ValueError("Only a failed run can be resumed")
        if manifest["preparation_manifest_sha256"] != digest(preparation / "dataset_manifest.json"):
            raise ValueError("Preparation snapshot changed")
        if manifest.get("selected_manifest_sha256") != digest(selected / "dataset_manifest.json"):
            raise ValueError("Selected snapshot changed")
        for item in manifest["experiments"]:
            if (digest(ROOT / item["artifact"]) != item["artifact_sha256"]
                    or digest(item["command"][1]) != item["script_sha256"]):
                raise ValueError("Completed experiment changed; cannot resume")
            dataset = preparation if item["number"] == 2 else selected
            for key, name in [("dataset_sha256", "stage3_final_labeled.csv"),
                              ("development_sha256", "stage5_split_v2/development.csv"),
                              ("holdout_sha256", "stage5_split_v2/holdout.csv")]:
                if digest(dataset / name) != item[key]:
                    raise ValueError("Dataset/split changed; cannot resume")
        attempt = len(list(logdir.glob("run_manifest_attempt_*.json"))) + 1
        (logdir / f"run_manifest_attempt_{attempt}.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        manifest.update(status="running", resumed_after_failure=True)
        manifest.pop("error", None)

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    save()
    scripts = {1: "experiment_1_split.py", 2: "experiment_2_threshold.py", 3: "experiment_3_tfidf.py",
               4: "experiment_4_svm.py", 5: "experiment_5_class_imbalance.py",
               6: "experiment_6_ovo_ovr.py", 7: "experiment_7_regularization.py"}
    try:
        for number in manifest["order"]:
            if number in {entry["number"] for entry in manifest["experiments"]}:
                continue
            dataset = preparation if number == 2 else selected
            if number == 1:
                threshold_artifact = experiments / EXPERIMENT_JSON_FILES[2]
                if not selected.exists():
                    derive(preparation, selected, threshold_artifact)
                else:
                    current = json.loads((selected / "dataset_manifest.json").read_text())
                    if (current.get("threshold_experiment_sha256") != digest(threshold_artifact)
                            or current.get("parent_manifest_sha256") != manifest["preparation_manifest_sha256"]):
                        raise ValueError("Existing derived snapshot does not match this threshold selection")
                manifest["selected_snapshot"] = str(selected.relative_to(ROOT))
                manifest["selected_manifest_sha256"] = digest(selected / "dataset_manifest.json")
            env = dict(os.environ, WAYGROUND_PIPELINE_OUTPUT_DIR=str(dataset), WAYGROUND_OUTPUT_DIR=str(output),
                       PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            script = ROOT / "scripts/experiments" / scripts[number]
            command = [sys.executable, str(script)]
            log = logdir / f"experiment_{number}.txt"
            attempt = 2
            while log.exists():
                log = logdir / f"experiment_{number}_attempt_{attempt}.txt"
                attempt += 1
            print(f"Experiment {number}: {script.name}; log={log.relative_to(ROOT)}", flush=True)
            dependencies = {EXPERIMENT_JSON_FILES[i]: digest(experiments / EXPERIMENT_JSON_FILES[i])
                            for i in range(3, number) if i <= 6}
            with log.open("w", encoding="utf-8") as handle:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f"Experiment {number} failed; see {log}")
            artifact = experiments / EXPERIMENT_JSON_FILES[number]
            json.loads(artifact.read_text(encoding="utf-8-sig"))
            manifest["experiments"].append({"number": number, "script_sha256": digest(script),
                "command": command, "log": str(log.relative_to(ROOT)), "dataset_sha256": digest(dataset / "stage3_final_labeled.csv"),
                "development_sha256": digest(dataset / "stage5_split_v2/development.csv"),
                "holdout_sha256": digest(dataset / "stage5_split_v2/holdout.csv"),
                "dependencies_sha256": dependencies,
                "artifact": str(artifact.relative_to(ROOT)), "artifact_sha256": digest(artifact)})
            save()
            print(f"Experiment {number}: completed", flush=True)
        manifest["status"] = "complete"
        save()
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        save()
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preparation", type=Path, default=ROOT / "data/pipeline_output_v3")
    p.add_argument("--selected", type=Path, default=ROOT / "data/pipeline_output_v4")
    p.add_argument("--output", type=Path, default=ROOT / "output_v4")
    p.add_argument("--resume", action="store_true", help="Resume a failed run after validating completed artifacts")
    args = p.parse_args()
    run(args.preparation, args.selected, args.output, args.resume)
