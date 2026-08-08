from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class BaselineRule:
    rule: str
    threshold: float

    @classmethod
    def from_config(cls, config: Dict[str, object]) -> "BaselineRule":
        return cls(rule=str(config["rule"]), threshold=float(config["threshold"]))

    def score(self, feature_frame: Dict[str, np.ndarray]) -> np.ndarray:
        query_length = None
        entropy = None

        if "query_length" in feature_frame:
            query_length = np.asarray(feature_frame["query_length"], dtype=float)
        elif "subdomain_length" in feature_frame:
            query_length = np.asarray(feature_frame["subdomain_length"], dtype=float)
        elif "len" in feature_frame:
            query_length = np.asarray(feature_frame["len"], dtype=float)
        else:
            query_length = np.zeros(len(next(iter(feature_frame.values()))), dtype=float)

        if "entropy" in feature_frame:
            entropy = np.asarray(feature_frame["entropy"], dtype=float)
        elif "rr_name_entropy" in feature_frame:
            entropy = np.asarray(feature_frame["rr_name_entropy"], dtype=float)
        else:
            entropy = np.zeros(len(query_length), dtype=float)

        normalized = (query_length / 120.0) + (entropy / 6.0)
        scores = np.clip(normalized / 2.0, 0.0, 1.0)
        return scores
