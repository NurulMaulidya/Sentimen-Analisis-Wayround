"""Validate the completed seven-experiment run and record its selected config.

No training, holdout scoring, or edits to the active deployment configuration.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def select(output):
    output = Path(output).resolve()
    os.environ["WAYGROUND_OUTPUT_DIR"] = str(output)
    # Import after binding the run namespace. This entry point runs in a fresh process.
    from shared.deployment_integrity import evaluation_provenance
    from shared.experiment_config_loader import load_experiment_chain
    import numpy as np
    import pandas as pd
    import sklearn
    from shared.splitting import assert_no_overlap

    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    run = json.loads((output / "research_run/run_manifest.json").read_text())
    if run["status"] != "complete" or {item["number"] for item in run["experiments"]} != set(range(1, 8)):
        raise ValueError("All seven experiments must be complete")
    for item in run["experiments"]:
        if sha(item["command"][1]) != item["script_sha256"]:
            raise ValueError(f"Experiment script changed since run: {item['number']}")
        if sha(ROOT / item["artifact"]) != item["artifact_sha256"]:
            raise ValueError(f"Changed experiment artifact: {item['number']}")
        for name, expected in item["dependencies_sha256"].items():
            if sha(output / "experiments" / name) != expected:
                raise ValueError(f"Changed dependency: {name}")
    dataset = ROOT / run["selected_snapshot"]
    if sha(dataset / "dataset_manifest.json") != run["selected_manifest_sha256"]:
        raise ValueError("Selected snapshot manifest changed")
    snapshot = json.loads((dataset / "dataset_manifest.json").read_text())
    for name, expected in snapshot["output_hashes"].items():
        if sha(dataset / name) != expected:
            raise ValueError(f"Changed snapshot artifact: {name}")
    data = pd.read_csv(dataset / "stage3_final_labeled.csv", encoding="utf-8-sig")
    dev = pd.read_csv(dataset / "stage5_split_v2/development.csv", encoding="utf-8-sig")
    holdout_path = dataset / "stage5_split_v2/holdout.csv"
    hold = pd.read_csv(holdout_path, encoding="utf-8-sig")
    provenance = evaluation_provenance(run, holdout_path)
    assert_no_overlap(dev, hold)
    if len(dev) + len(hold) != len(data) or set(dev.reviewId) | set(hold.reviewId) != set(data.reviewId):
        raise ValueError("Split does not cover the selected dataset")
    for item in run["experiments"]:
        if item["number"] == 2:
            continue  # Threshold selected on the preparation snapshot's development.
        for field, name in [("dataset_sha256", "stage3_final_labeled.csv"),
                            ("development_sha256", "stage5_split_v2/development.csv"),
                            ("holdout_sha256", "stage5_split_v2/holdout.csv")]:
            if item[field] != sha(dataset / name):
                raise ValueError("Experiments used different datasets/splits")
    chain = load_experiment_chain(include_exp7=True)
    final = chain["final"]
    strategies = chain["exp6"]["data"]["strategies"]
    chosen = next(s for s in strategies if s["strategy"] == final["multiclass"])
    if chain["exp5"]["data"]["dataset"]["rows_development_train"] != len(dev):
        raise ValueError("Experiment 5 trained on a different development partition")
    if chain["exp6"]["data"]["dataset"]["rows_development_train"] != len(dev):
        raise ValueError("Experiment 6 trained on a different development partition")
    winner = sorted(strategies, key=lambda s: (-s["mean_f1_macro"], s["std_f1_macro"]))[0]
    if chosen["strategy"] != winner["strategy"]:
        raise ValueError("Selected strategy is not the Macro F1 winner")
    # No averaged/pooled metric substitution: report both with distinct labels.
    for strategy in strategies:
        scores = [fold["f1_macro"] for fold in strategy["fold_results"]]
        if not np.isclose(np.std(scores, ddof=1), strategy["std_f1_macro"]):
            raise ValueError("Inconsistent CV standard deviation")
    e7 = chain["exp7"]["data"]
    reference = next(row for row in e7["results"] if float(row["C"]) == float(final["svm"]["C"]))
    if not np.isclose(reference["cv_f1_mean"], chosen["mean_f1_macro"]):
        raise ValueError("Experiment 7 reference configuration does not reproduce Experiment 6")
    e2 = json.loads((output / "experiments/experiment_2_threshold_results.json").read_text())
    result = {
        "status": "selected_pending_deployment", "config_source": "sequential_experiment_winners_3_to_6",
        "selection_rule": "Highest development grouped-CV mean Macro F1, then lower sample SD; Experiment 7 diagnostic only",
        "dataset": {"path": str((dataset / "stage3_final_labeled.csv").relative_to(ROOT)),
                    "sha256": sha(dataset / "stage3_final_labeled.csv"), "rows": len(data),
                    "class_distribution": data.sentiment_label_final.str.lower().value_counts().to_dict(),
                    "expert_rows": int(data.sentiment_label_expert.notna().sum())},
        "split": {"development_rows": len(dev), "holdout_rows": len(hold),
                  "method": "StratifiedGroupKFold", "group_column": "processed_text", "random_state": 42,
                  "cv_folds": 5, "group_overlap": 0, "holdout_evaluated": False,
                  "holdout_rows_exposed_in_invalidated_attempts": run.get("invalidation", {}).get("canonical_holdout_rows_exposed_in_invalidated_cv", 0),
                  "invalidated_attempts_excluded_from_final_selection": True,
                  "holdout_independence_verified": False},
        "labeling": {"lexicon_version": "v2", "neutral_threshold": e2["configuration"]["selected_threshold"],
                     "threshold_selection_partition": "preparation_development_only",
                     "threshold_expert_rows": e2["expert_validation"]["n"],
                     "threshold_expert_macro_f1": e2["selected_result"]["expert_validation"]["f1_macro"]},
        "model": {"tfidf": final["tfidf"], "svm": final["svm"],
                  "imbalance": {"method": final["imbalance"]}, "multiclass": {"strategy": final["multiclass"]}},
        "sampling_parameters": {"random_state": 42, "smote_k_neighbors": 3},
        "development_cv": {"f1_macro_mean": chosen["mean_f1_macro"], "f1_macro_std": chosen["std_f1_macro"],
                           "std_ddof": 1, "folds": chosen["fold_results"],
                           "pooled_oof_classification_report": chosen["oof_classification_report"]},
        "regularization_diagnostic": {key: e7.get(key) for key in ["best", "selection_status", "recommended_gap_compliant_candidate"]},
        "limitations": ["Sequential search, not a joint global optimum or nested-CV unbiased performance estimate",
                        "Most labels are lexicon-derived; independent gold labels and second annotator remain unavailable",
                        provenance["note"],
                        "No final holdout performance is available for this selected configuration yet"],
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "run_manifest_sha256": sha(output / "research_run/run_manifest.json"),
        "experiment_artifact_hashes": {str(x["number"]): x["artifact_sha256"] for x in run["experiments"]},
    }
    target = output / "experiments/selected_configuration.json"
    if target.exists():
        raise FileExistsError("Selection already recorded; use a new run namespace to revise it")
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["status", "model", "labeling", "development_cv"]}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output_v4")
    select(parser.parse_args().output)
