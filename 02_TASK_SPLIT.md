# Task Split — Person A vs Person B

Split by pipeline stage, not by seniority — you said skills are similar, so this
just needs to minimize blocking. Swap names in freely.

## Person A — Data Integrity & Evaluation
Owns everything that makes the score *trustworthy*. This is worth doing first
because "AI & Data Quality" + "Safety & Reliability" together are 40% of the
rubric, and a correct baseline alone is already a demoable end-to-end system.

1. Load organizer split manifest; build `load_split()` that returns
   train/val/test DataFrames.
2. Implement and test the leakage guards:
   - no `session_id` in more than one partition
   - no `registrable_domain` in more than one partition
   - no window/row crosses a `collection_day` boundary
3. Implement the non-ML baseline: tuned rule on query length + entropy
   (threshold tuned on train, documented).
4. Build the metrics module (see contract in 01_DATA_CONTRACT.md):
   PR-AUC, precision/recall/FPR @ threshold, confusion matrix, false alerts
   per 10k benign, split by light/heavy/combined.
5. Build the latency/throughput harness: median + p95 per row, with/without
   feature extraction time, declared batch size/hardware.
6. Leakage audit doc: list every excluded field and confirm none were used
   as predictors (feeds into the technical report).

## Person B — Model & Explanation
Owns everything that makes the score *good*.

1. Feature prep: imputation/scaling/encoding fit on train only, applied to
   val/test (no data leakage across the fit boundary).
2. Train the ML model (LightGBM/XGBoost primary, logistic regression fallback
   if environment issues eat time).
3. Threshold selection on the validation split (not test).
4. Calibrate scores if time allows (Platt/isotonic) — optional, cut first if
   behind schedule.
5. Reason codes: for each alert, surface top-N contributing features
   (feature importances, or SHAP if it installs cleanly and there's time).
   Keep it simple — a ranked list of feature names + values is enough.
6. Reproducibility package: environment lockfile, fixed random seeds, single
   command that reproduces the reported numbers.

## Joint work (do NOT split these — do them together)
- Hour 1: agree on data contract + write failing tests first (see
  03_TEST_SPECS.md).
- Integration at 3:30: plug B's model into A's real loader.
- Full pipeline run at 6:00 checkpoint.
- Technical report, results bundle, data/model statement, pitch.

## If you're ahead of schedule (stretch goals, in priority order)
1. Move decision unit from row-level to window/session-level aggregation.
2. Add confidence intervals / fold-to-fold variation on the split.
3. Simple Streamlit wrapper for the demo.
4. Calibration (if not already done).
5. Compare one-stage vs a cheap two-stage design (borrow from Track 2 idea)
   as a bonus "innovation" point — only if everything else is solid.

## If you're behind schedule (cut in this order)
1. Drop calibration.
2. Drop confidence intervals.
3. Drop SHAP, keep plain feature-importance reason codes.
4. Drop latency "with vs without feature extraction" split — report one number.
5. Never cut: leakage guards, baseline, light/heavy separate metrics, human
   review framing. These are rubric-critical and safety-critical.
