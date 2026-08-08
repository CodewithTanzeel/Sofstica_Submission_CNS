from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.baseline import BaselineRule
from src.features import prepare_features
from src.load import load_kaggle_dataset
from src.metrics import compute_metrics
from src.model import SimpleModel


ROOT = Path(__file__).resolve().parent


def run_pipeline(dataset_dir: str | None = None) -> dict:
    dataset_dir = dataset_dir or str(ROOT / "data" / "kaggle_dataset")

    split_frames = load_kaggle_dataset(dataset_dir)

    train = split_frames["train"]
    val = split_frames["val"]
    test = split_frames["test"]

    baseline_rule = BaselineRule.from_config({"rule": "query_length + entropy", "threshold": 0.55})

    baseline_inputs = {}
    if "query_length" in val.columns:
        baseline_inputs["query_length"] = val["query_length"].to_numpy()
    elif "subdomain_length" in val.columns:
        baseline_inputs["subdomain_length"] = val["subdomain_length"].to_numpy()
    elif "len" in val.columns:
        baseline_inputs["len"] = val["len"].to_numpy()

    if "entropy" in val.columns:
        baseline_inputs["entropy"] = val["entropy"].to_numpy()
    elif "rr_name_entropy" in val.columns:
        baseline_inputs["rr_name_entropy"] = val["rr_name_entropy"].to_numpy()

    val_scores = baseline_rule.score(baseline_inputs)

    test_baseline_inputs = {}
    if "query_length" in test.columns:
        test_baseline_inputs["query_length"] = test["query_length"].to_numpy()
    elif "subdomain_length" in test.columns:
        test_baseline_inputs["subdomain_length"] = test["subdomain_length"].to_numpy()
    elif "len" in test.columns:
        test_baseline_inputs["len"] = test["len"].to_numpy()

    if "entropy" in test.columns:
        test_baseline_inputs["entropy"] = test["entropy"].to_numpy()
    elif "rr_name_entropy" in test.columns:
        test_baseline_inputs["rr_name_entropy"] = test["rr_name_entropy"].to_numpy()

    test_scores = baseline_rule.score(test_baseline_inputs)

    prepared = prepare_features(train, val, test)
    model = SimpleModel()
    model.fit(prepared["train"], prepared["train"]["label"].to_numpy())
    val_model_scores = model.predict_proba(prepared["val"])
    test_model_scores = model.predict_proba(prepared["test"])

    baseline_val_metrics = compute_metrics(val["label"].to_numpy(), val_scores, val["label_detail"].to_numpy(), "val")
    baseline_test_metrics = compute_metrics(test["label"].to_numpy(), test_scores, test["label_detail"].to_numpy(), "test")
    model_val_metrics = compute_metrics(val["label"].to_numpy(), val_model_scores, val["label_detail"].to_numpy(), "val")
    model_test_metrics = compute_metrics(test["label"].to_numpy(), test_model_scores, test["label_detail"].to_numpy(), "test")

    results = {
        "baseline_val_metrics": baseline_val_metrics,
        "baseline_test_metrics": baseline_test_metrics,
        "model_val_metrics": model_val_metrics,
        "model_test_metrics": model_test_metrics,
    }

    output_path = ROOT / "results" / "scoring_output.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the DNS exfiltration detection pipeline")
    parser.add_argument("--dataset-dir", default=str(ROOT / "data" / "kaggle_dataset"), help="Path to the Kaggle dataset folder")
    args = parser.parse_args()
    run_pipeline(args.dataset_dir)
