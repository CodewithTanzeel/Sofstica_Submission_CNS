# DNS Exfiltration Detection — Person A Prototype

## Objective

This prototype implements the Person A side of the hackathon pipeline: split loading, leakage controls, baseline scoring, metrics reporting, and a reproducible CLI entry point.

## What is implemented

- Manifest-driven split loading through the organizer-style row mapping.
- Leakage checks to ensure sessions and registrable domains do not overlap across train/val/test.
- A simple non-ML baseline that uses query length and entropy heuristics.
- Metrics reporting for PR-AUC, precision, recall, FPR, confusion matrix, and false alerts per 10k benign rows.
- A runnable pipeline that writes results to results/scoring_output.json.

## Caveats

This is a lightweight prototype intended for the hackathon. It uses a synthetic, schema-matching sample until the organizer data is available.
