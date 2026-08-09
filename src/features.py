from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def prepare_features(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, return_transformers: bool = False):
    """Simple feature preparation that fits on train only."""
    # sld/subdomain/timestamp are explicit domain-leakage / shortcut fields per
    # the brief (raw domain identity and capture timestamps are not valid
    # predictive evidence) — excluded deliberately, not as a side effect of
    # dtype filtering.
    excluded_columns = {
        "row_id", "capture_id", "session_id", "collection_day", "registrable_domain",
        "split", "label", "label_detail", "source_file",
        "sld", "subdomain", "timestamp",
    }
    feature_columns = [col for col in train.columns if col not in excluded_columns and pd.api.types.is_numeric_dtype(train[col])]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = imputer.fit_transform(train[feature_columns])
    X_train = scaler.fit_transform(X_train)

    X_val = imputer.transform(val[feature_columns])
    X_val = scaler.transform(X_val)

    X_test = imputer.transform(test[feature_columns])
    X_test = scaler.transform(X_test)

    train_processed = train.copy()
    val_processed = val.copy()
    test_processed = test.copy()

    for split_name, matrix in [("train", X_train), ("val", X_val), ("test", X_test)]:
        frame = train_processed if split_name == "train" else val_processed if split_name == "val" else test_processed
        for index, col in enumerate(feature_columns):
            frame[f"{col}_scaled"] = matrix[:, index]

    result = {"train": train_processed, "val": val_processed, "test": test_processed}
    if return_transformers:
        result["imputer"] = imputer
        result["scaler"] = scaler
        result["feature_columns"] = feature_columns
    return result
