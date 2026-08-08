from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier


class SimpleModel:
    def __init__(self) -> None:
        self._threshold = 0.5
        self._feature_columns: List[str] = []
        self._primary_model = None
        self._model_name = "xgboost"

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> "SimpleModel":
        excluded_columns = {
            "row_id", "capture_id", "session_id", "collection_day", "registrable_domain",
            "split", "label", "label_detail", "source_file",
            "sld", "subdomain", "timestamp",
        }
        self._feature_columns = [col for col in frame.columns if col not in excluded_columns and pd.api.types.is_numeric_dtype(frame[col])]
        X = frame[self._feature_columns].to_numpy(dtype=float)

        self._logistic = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1, solver="lbfgs")
        self._logistic.fit(X, y)

        self._primary_model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="aucpr",
            n_jobs=1,
            random_state=42,
            tree_method="hist",
            objective="binary:logistic",
        )
        self._primary_model.fit(X, y)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._feature_columns:
            excluded_columns = {
            "row_id", "capture_id", "session_id", "collection_day", "registrable_domain",
            "split", "label", "label_detail", "source_file",
            "sld", "subdomain", "timestamp",
        }
            self._feature_columns = [col for col in frame.columns if col not in excluded_columns and pd.api.types.is_numeric_dtype(frame[col])]
        X = frame[self._feature_columns].to_numpy(dtype=float)
        return self._primary_model.predict_proba(X)[:, 1]

    def explain(self, row: pd.Series) -> List[tuple[str, float]]:
        if not self._feature_columns:
            return []
        importances = self._primary_model.feature_importances_
        ranked = sorted(zip(self._feature_columns, importances), key=lambda item: abs(item[1]), reverse=True)
        return [(name, float(value)) for name, value in ranked[:10]]


def _class_weight_ratio(y: np.ndarray) -> float:
    n_pos = max(int(y.sum()), 1)
    n_neg = max(len(y) - n_pos, 1)
    return n_neg / n_pos


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1, solver="lbfgs")
    clf.fit(X_train, y_train)
    return clf


def train_primary_model(X_train, y_train, feature_names: list[str]):
    pos_weight = _class_weight_ratio(np.asarray(y_train))
    try:
        from lightgbm import LGBMClassifier

        clf = LGBMClassifier(
            n_estimators=300,
            max_depth=7,
            num_leaves=31,
            learning_rate=0.05,
            scale_pos_weight=pos_weight,
            n_jobs=-1,
            random_state=42,
            verbosity=-1,
        )
        clf.fit(X_train, y_train)
        return clf, "lightgbm"
    except ImportError:
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        return clf, "random_forest"


def select_threshold(y_val, val_scores, target_fpr: float | None = None) -> float:
    precision, recall, thresholds = precision_recall_curve(y_val, val_scores)
    precision, recall = precision[:-1], recall[:-1]

    if target_fpr is not None:
        y_val = np.asarray(y_val)
        neg_scores = val_scores[y_val == 0]
        candidate = np.quantile(neg_scores, 1 - target_fpr)
        return float(candidate)

    f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall + 1e-9), 0)
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]) if len(thresholds) else 0.5


@dataclass
class TimedPrediction:
    scores: np.ndarray
    median_latency_ms: float
    p95_latency_ms: float
    n_rows: int


def predict_with_timing(model, X) -> TimedPrediction:
    n = X.shape[0]
    sample_idx = np.random.RandomState(42).choice(n, size=min(n, 500), replace=False)
    per_row_ms = []
    for i in sample_idx:
        row = X[i : i + 1]
        t0 = time.perf_counter()
        model.predict_proba(row)
        per_row_ms.append((time.perf_counter() - t0) * 1000)

    scores = model.predict_proba(X)[:, 1]
    return TimedPrediction(
        scores=scores,
        median_latency_ms=float(np.median(per_row_ms)),
        p95_latency_ms=float(np.percentile(per_row_ms, 95)),
        n_rows=n,
    )


def reason_codes(model, feature_names: list[str], row: np.ndarray, top_n: int = 3) -> list[str]:
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        importances = np.abs(np.asarray(model.coef_)).ravel()
    else:
        raise ValueError("Model has neither feature_importances_ nor coef_")

    row = np.asarray(row).ravel()
    score = importances * np.abs(row)
    top_idx = np.argsort(score)[::-1][:top_n]
    return [feature_names[i] for i in top_idx]


def model_name(model) -> str:
    return "lightgbm" if hasattr(model, "feature_importances_") and type(model).__name__ != "LogisticRegression" else type(model).__name__
