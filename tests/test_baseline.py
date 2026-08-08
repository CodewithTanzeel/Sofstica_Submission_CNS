import numpy as np


def test_baseline_rule_is_documented(baseline_config):
    from src.baseline import BaselineRule

    rule = BaselineRule.from_config(baseline_config)
    assert rule.rule == baseline_config["rule"]
    assert rule.threshold == baseline_config["threshold"]


def test_baseline_beats_random():
    from src.baseline import BaselineRule

    y_true = np.array([0, 1, 0, 1, 0, 1])
    feature_frame = {
        "query_length": np.array([10, 100, 12, 95, 11, 90]),
        "entropy": np.array([2.0, 4.8, 2.1, 4.7, 2.2, 4.9]),
    }

    rule = BaselineRule.from_config({"rule": "query_length + entropy", "threshold": 0.55})
    scores = rule.score(feature_frame)
    prevalence = np.mean(y_true == 1)

    from src.metrics import compute_metrics
    metrics = compute_metrics(y_true, scores, label_detail=np.array([0, 1, 0, 2, 0, 1]), split_name="val")

    assert metrics["per_class"]["combined"]["pr_auc"] > prevalence
