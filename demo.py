"""
Demo: shows one benign, one light-attack, and one heavy-attack case from the
held-out test set, with the model's score, the operating threshold, the
alert decision, and the reason codes behind it (if alerted).

This satisfies the brief's "Working prototype" deliverable requirement to
demonstrate at least one benign, one light-attack, and one heavy-attack case
end-to-end.

Usage:
    python demo.py                      # uses existing results/predictions.csv
    python demo.py --rerun              # re-runs the pipeline first
    python demo.py --case-index 1       # picks the 2nd example of each type
                                         # instead of the 1st (for variety)

Detection output shown here is advisory only, consistent with the brief's
safety rules — this script prints a recommendation, it does not take any
blocking action.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
LABEL_NAMES = {0: "BENIGN", 1: "LIGHT ATTACK", 2: "HEAVY ATTACK"}


def _load_predictions(rerun: bool) -> pd.DataFrame:
    predictions_path = ROOT / "results" / "predictions.csv"
    if rerun or not predictions_path.exists():
        from pipeline import run_pipeline
        print("Running pipeline to regenerate results...\n")
        run_pipeline(str(ROOT / "data"))
    return pd.read_csv(predictions_path)


def _load_scoring_summary() -> dict:
    scoring_path = ROOT / "results" / "scoring_output.json"
    return json.loads(scoring_path.read_text(encoding="utf-8"))


def _print_case(row: pd.Series, threshold: float) -> None:
    label_name = LABEL_NAMES[int(row["true_label_detail"])]
    alerted = bool(row["predicted_label"])
    correct = (row["predicted_label"] == row["true_label"])

    print(f"--- Case: {label_name} (source file: {row['source_file']}) ---")
    print(f"  row_id:          {row['row_id']}")
    print(f"  true label:      {'exfiltration' if row['true_label'] == 1 else 'benign'} ({label_name})")
    print(f"  model score:     {row['model_score']:.4f}")
    print(f"  threshold:       {threshold:.4f}")
    print(f"  decision:        {'ALERT (flagged for analyst review)' if alerted else 'no alert'}")
    print(f"  correct?:        {'yes' if correct else 'NO — misclassified'}")
    if alerted and isinstance(row["top_reason_codes"], str) and row["top_reason_codes"].strip():
        print(f"  reason codes:    {row['top_reason_codes']}")
    elif not alerted:
        print("  reason codes:    (none — no alert raised, so no reason codes shown)")
    print(
        "  human review:    advisory only — this output does not block DNS, "
        "quarantine anything, or take any autonomous action."
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo one benign, one light, one heavy case")
    parser.add_argument("--rerun", action="store_true", help="Re-run the pipeline before demoing")
    parser.add_argument("--case-index", type=int, default=0, help="Which example of each class to show (0-indexed)")
    parser.add_argument("--show-mistakes", action="store_true",
                         help="Show misclassified examples instead of correct ones (useful for the failure-analysis section of the technical report)")
    args = parser.parse_args()

    predictions = _load_predictions(args.rerun)
    scoring = _load_scoring_summary()
    threshold = scoring["threshold_selection"]["model_threshold_selected_on_val"]

    print("=" * 70)
    print("DNS EXFILTRATION DETECTION — DEMO (Track 1)")
    print("=" * 70)
    mode_note = "misclassified" if args.show_mistakes else "correctly classified"
    print(
        f"Showing one benign, one light-attack, and one heavy-attack example "
        f"(each {mode_note}) from the held-out test set. Full metrics are in "
        "results/scoring_output.json; this is a walk-through of individual "
        "decisions, not the headline score.\n"
    )

    for label_detail in (0, 1, 2):
        subset = predictions[predictions["true_label_detail"] == label_detail]
        is_correct = subset["predicted_label"] == subset["true_label"]
        filtered = subset[~is_correct] if args.show_mistakes else subset[is_correct]
        if len(filtered) == 0:
            note = "no misclassified examples of this type" if args.show_mistakes else "no correctly classified examples of this type"
            print(f"--- Case: {LABEL_NAMES[label_detail]} — {note} in test split ---\n")
            filtered = subset
        if len(filtered) == 0:
            print(f"--- Case: {LABEL_NAMES[label_detail]} — none available in test split ---\n")
            continue
        idx = min(args.case_index, len(filtered) - 1)
        row = filtered.iloc[idx]
        _print_case(row, threshold)

    combined = scoring["model_test_metrics"]["per_class"]["combined"]
    light = scoring["model_test_metrics"]["per_class"]["light"]
    heavy = scoring["model_test_metrics"]["per_class"]["heavy"]
    print("=" * 70)
    print("SUMMARY (test set, full numbers in results/scoring_output.json)")
    print("=" * 70)
    print(f"  Combined  — PR-AUC {combined['pr_auc']:.3f}  Precision {combined['precision']:.3f}  Recall {combined['recall']:.3f}  FPR {combined['fpr']:.3f}")
    print(f"  Light     — PR-AUC {light['pr_auc']:.3f}  Precision {light['precision']:.3f}  Recall {light['recall']:.3f}  FPR {light['fpr']:.3f}")
    print(f"  Heavy     — PR-AUC {heavy['pr_auc']:.3f}  Precision {heavy['precision']:.3f}  Recall {heavy['recall']:.3f}  FPR {heavy['fpr']:.3f}")
    print(
        "\nNote: light-attack detection is meaningfully weaker than heavy-attack "
        "detection (lower PR-AUC/precision above) — this is a known, documented "
        "limitation, not a demo artifact. See technical_report.md."
    )


if __name__ == "__main__":
    main()
