"""
Score brand-new raw stateful+stateless feature CSVs using the saved
inference bundle (model + imputer + scaler), without retraining.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.model import reason_codes


def load_bundle(path: str | Path = "results/inference_bundle.pkl") -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def score_dataframe(bundle: dict, stateful: pd.DataFrame, stateless: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Reproduce prepare_features' transform step (fit-free) on new raw rows."""
    if stateless is not None:
        n = min(len(stateful), len(stateless))
        combined = pd.concat(
            [stateful.iloc[:n].reset_index(drop=True), stateless.iloc[:n].reset_index(drop=True)], axis=1
        )
    else:
        combined = stateful.copy()

    feature_columns = bundle["raw_feature_columns"]
    missing = [c for c in feature_columns if c not in combined.columns]
    if missing:
        raise ValueError(f"Uploaded CSV is missing expected columns: {missing}")

    X = bundle["imputer"].transform(combined[feature_columns])
    X_scaled = bundle["scaler"].transform(X)
    for i, col in enumerate(feature_columns):
        combined[f"{col}_scaled"] = X_scaled[:, i]

    scores = bundle["model"].predict_proba(combined)
    threshold = bundle["threshold"]
    combined["model_score"] = scores
    combined["predicted_label"] = (scores >= threshold).astype(int)

    # reason_codes() needs the exact same columns/order the model was fit on
    # (raw + *_scaled, 62 columns) — not just the raw scaled matrix.
    raw_model = bundle["model"]._primary_model
    model_feature_columns = bundle["model"]._feature_columns
    model_input = combined[model_feature_columns].to_numpy(dtype=float)

    reasons = []
    for i in range(len(combined)):
        reasons.append(
            ", ".join(reason_codes(raw_model, model_feature_columns, model_input[i], top_n=3))
            if combined["predicted_label"].iloc[i] == 1
            else ""
        )
    combined["top_reason_codes"] = reasons
    return combined