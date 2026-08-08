from __future__ import annotations

from typing import Dict, Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, confusion_matrix


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_detail: np.ndarray,
    split_name: str,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    label_detail = np.asarray(label_detail)

    y_true_binary = (y_true != 0).astype(int)
    y_pred = (y_score >= threshold).astype(int)

    cm = confusion_matrix(y_true_binary, y_pred, labels=[0, 1])

    precision = precision_score(y_true_binary, y_pred, zero_division=0)
    recall = recall_score(y_true_binary, y_pred, zero_division=0)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / max(tn + fp, 1)

    combined_pr_auc = average_precision_score(y_true_binary, y_score)

    per_class: Dict[str, Dict[str, Any]] = {}
    for attack_name, target in [("light", 1), ("heavy", 2)]:
        # Compare this attack type against the SAME benign population used
        # for the combined metric (label_detail == 0), excluding rows of the
        # *other* attack type. Filtering to attack-only rows (the previous
        # implementation) leaves y_true all-positive, which makes PR-AUC
        # trivially 1.0 and precision/recall meaningless.
        class_mask = (label_detail == target) | (label_detail == 0)
        if (label_detail[class_mask] == target).sum() == 0:
            per_class[attack_name] = {"pr_auc": 0.0, "precision": 0.0, "recall": 0.0, "fpr": 0.0}
            continue
        y_true_attack = (y_true[class_mask] != 0).astype(int)
        y_score_attack = y_score[class_mask]
        attack_pred = (y_score_attack >= threshold).astype(int)
        attack_precision = precision_score(y_true_attack, attack_pred, zero_division=0)
        attack_recall = recall_score(y_true_attack, attack_pred, zero_division=0)
        attack_cm = confusion_matrix(y_true_attack, attack_pred, labels=[0, 1])
        attack_tn, attack_fp, attack_fn, attack_tp = attack_cm.ravel()
        attack_fpr = attack_fp / max(attack_tn + attack_fp, 1)
        per_class[attack_name] = {
            "pr_auc": average_precision_score(y_true_attack, y_score_attack),
            "precision": attack_precision,
            "recall": attack_recall,
            "fpr": attack_fpr,
        }

    per_class["combined"] = {
        "pr_auc": combined_pr_auc,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
    }

    return {
        "split_name": split_name,
        "threshold": threshold,
        "pr_auc": combined_pr_auc,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "confusion_matrix": cm.tolist(),
        "false_alerts_per_10k_benign": fpr * 10000,
        "per_class": per_class,
    }
