from __future__ import annotations

import argparse
import pickle
from sklearn.metrics import accuracy_score
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from src.model import SimpleModel, select_threshold, select_threshold_for_precision, predict_with_timing, reason_codes
from src.baseline import BaselineRule
from src.features import prepare_features
from src.leakage import validate_leakage
from src.load import load_kaggle_dataset, group_manifest
from src.metrics import compute_metrics



ROOT = Path(__file__).resolve().parent


def _baseline_inputs(frame: pd.DataFrame) -> dict:
    inputs = {}
    if "query_length" in frame.columns:
        inputs["query_length"] = frame["query_length"].to_numpy()
    elif "subdomain_length" in frame.columns:
        inputs["subdomain_length"] = frame["subdomain_length"].to_numpy()
    elif "len" in frame.columns:
        inputs["len"] = frame["len"].to_numpy()

    if "entropy" in frame.columns:
        inputs["entropy"] = frame["entropy"].to_numpy()
    elif "rr_name_entropy" in frame.columns:
        inputs["rr_name_entropy"] = frame["rr_name_entropy"].to_numpy()
    return inputs


def run_pipeline(dataset_dir: str | None = None) -> dict:
    dataset_dir = dataset_dir or str(ROOT / "data" / "kaggle_dataset")

    split_frames = load_kaggle_dataset(dataset_dir)

    train = split_frames["train"]
    val = split_frames["val"]
    test = split_frames["test"]

    # --- Mandatory leakage audit: no session/domain crosses a partition. ---
    validate_leakage({"train": train, "val": val, "test": test})
    manifest = group_manifest(split_frames)
    manifest.to_csv(ROOT / "results" / "group_split_manifest.csv", index=False)

    # --- Baseline (non-ML rule), tuned/thresholded on val, scored on test. ---
    baseline_rule = BaselineRule.from_config({"rule": "query_length + entropy", "threshold": 0.55})
    val_scores_baseline = baseline_rule.score(_baseline_inputs(val))
    test_scores_baseline = baseline_rule.score(_baseline_inputs(test))
    baseline_threshold = select_threshold(val["label"].to_numpy(), val_scores_baseline)

    # --- ML model: fit on train only, threshold selected on val only. ---
    prepared = prepare_features(train, val, test, return_transformers=True)
    model = SimpleModel()
    model.fit(prepared["train"], prepared["train"]["label"].to_numpy())

    val_model_scores = model.predict_proba(prepared["val"])
    model_threshold = select_threshold_for_precision(val["label"].to_numpy(), val_model_scores, min_precision=0.98)

    # predict_with_timing expects a plain sklearn-style estimator taking a
    # numpy array (SimpleModel.predict_proba expects a DataFrame instead),
    # so time the underlying fitted xgboost model directly.
    timed = predict_with_timing(model._primary_model, prepared["test"][model._feature_columns].to_numpy(dtype=float))
    test_model_scores = timed.scores

    baseline_val_metrics = compute_metrics(val["label"].to_numpy(), val_scores_baseline, val["label_detail"].to_numpy(), "val", threshold=baseline_threshold)
    baseline_test_metrics = compute_metrics(test["label"].to_numpy(), test_scores_baseline, test["label_detail"].to_numpy(), "test", threshold=baseline_threshold)
    model_val_metrics = compute_metrics(val["label"].to_numpy(), val_model_scores, val["label_detail"].to_numpy(), "val", threshold=model_threshold)
    model_test_metrics = compute_metrics(test["label"].to_numpy(), test_model_scores, test["label_detail"].to_numpy(), "test", threshold=model_threshold)

    # --- Per-row reason codes: top-3 features driving each individual score, ---
    # --- not just a global importance ranking. Only computed for alerts   ---
    # --- (predicted positive) to keep the results bundle small and focused. ---
    test_pred = (test_model_scores >= model_threshold).astype(int)
    X_test = prepared["test"][model._feature_columns].to_numpy(dtype=float)
    per_row_reasons = []
    for i in range(len(test)):
        if test_pred[i] == 1:
            codes = reason_codes(model._primary_model, model._feature_columns, X_test[i], top_n=3)
            per_row_reasons.append("; ".join(codes))
        else:
            per_row_reasons.append("")

    predictions = pd.DataFrame({
        "row_id": test["row_id"].to_numpy(),
        "source_file": test["source_file"].to_numpy(),
        "true_label": test["label"].to_numpy(),
        "true_label_detail": test["label_detail"].to_numpy(),
        "model_score": test_model_scores,
        "predicted_label": test_pred,
        "threshold_used": model_threshold,
        "top_reason_codes": per_row_reasons,
    })
    predictions.to_csv(ROOT / "results" / "predictions.csv", index=False)

    results = {
        "baseline_val_metrics": baseline_val_metrics,
        "baseline_test_metrics": baseline_test_metrics,
        "model_val_metrics": model_val_metrics,
        "model_test_metrics": model_test_metrics,
        "threshold_selection": {
            "baseline_threshold_selected_on_val": baseline_threshold,
            "model_threshold_selected_on_val": model_threshold,
            "method": "F1-maximizing threshold over the validation precision-recall curve (src/model.py:select_threshold)",
        },
        "accuracy": accuracy_score(test["label"].to_numpy(), test_pred),
        "latency": {
            "median_latency_ms_per_row": timed.median_latency_ms,
            "p95_latency_ms_per_row": timed.p95_latency_ms,
            "n_rows_timed_individually": 500,
            "n_rows_total_test": timed.n_rows,
            "note": "Per-row timings exclude feature extraction (CSV load + impute/scale), which is measured and reported separately in the technical report.",
            "hardware": platform.platform(),
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
        },
        "model_config": {
            "model_name": model._model_name,
            "feature_columns": model._feature_columns,
        },
        "split_methodology": (
            "File-group (proxy session) level split, stratified by label_detail; "
            "see results/group_split_manifest.csv for the full group-split assignment."
        ),
        "model_pickle_path": str(ROOT / "model.pkl"),
    }

    # Save the trained SimpleModel for later use
        # Save the trained SimpleModel for later use
    with open(ROOT / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(ROOT / "results" / "inference_bundle.pkl", "wb") as f:
        pickle.dump(
            {
                "model": model,
                "imputer": prepared["imputer"],
                "scaler": prepared["scaler"],
                "raw_feature_columns": prepared["feature_columns"],
                "threshold": model_threshold,
            },
            f,
        )
    output_path = ROOT / "results" / "scoring_output.json"
    output_path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the DNS exfiltration detection pipeline")
    parser.add_argument("--dataset-dir", default=str(ROOT / "data" / "kaggle_dataset"), help="Path to the Kaggle dataset folder")
    args = parser.parse_args()
    run_pipeline(args.dataset_dir)
