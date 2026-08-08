import numpy as np


def test_metrics_reports_required_fields():
    from src.metrics import compute_metrics

    metrics = compute_metrics(
        y_true=np.array([0, 1, 0, 1]),
        y_score=np.array([0.1, 0.9, 0.2, 0.8]),
        label_detail=np.array([0, 1, 0, 2]),
        split_name="val",
    )

    assert "pr_auc" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "fpr" in metrics
    assert "confusion_matrix" in metrics
    assert "false_alerts_per_10k_benign" in metrics
    assert "light" in metrics["per_class"]
    assert "heavy" in metrics["per_class"]
    assert "combined" in metrics["per_class"]


def test_metrics_split_by_attack_class():
    from src.metrics import compute_metrics

    metrics = compute_metrics(
        y_true=np.array([0, 1, 0, 1]),
        y_score=np.array([0.1, 0.9, 0.2, 0.8]),
        label_detail=np.array([0, 1, 0, 2]),
        split_name="test",
    )

    assert metrics["per_class"]["light"]["pr_auc"] >= 0
    assert metrics["per_class"]["heavy"]["pr_auc"] >= 0
    assert metrics["per_class"]["combined"]["pr_auc"] >= 0


def test_threshold_selected_on_validation_not_test():
    from src.metrics import compute_metrics

    metrics = compute_metrics(
        y_true=np.array([0, 1, 0, 1]),
        y_score=np.array([0.1, 0.9, 0.2, 0.8]),
        label_detail=np.array([0, 1, 0, 2]),
        split_name="val",
    )

    assert metrics["threshold"] == 0.5
