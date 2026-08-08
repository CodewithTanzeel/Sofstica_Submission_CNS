# DNS Exfiltration Detection — Track 1 (Detection & Alert Prioritization)

## Goal
Build a working, leakage-safe detector that distinguishes benign DNS traffic from
simulated exfiltration (light + heavy) using the CIC-Bell-DNS-EXF-2021 feature
tables, beats a simple non-ML baseline, and produces a ranked alert list with
reason codes — in 12 hours, with 2 people, using TDD.

## Scope decisions (locked for v1 — do not revisit mid-hackathon)
- **Data used:** stateless + stateful feature CSVs only. No raw PCAP parsing.
- **Decision unit:** start at row/query level (simplest). Only move to
  window/session aggregation if time remains (stretch goal).
- **Model:** non-ML baseline (entropy + query-length rule) **required**, plus one
  ML model (LightGBM/XGBoost, or logistic regression as fallback if install issues).
- **Interface:** CLI + notebook. No dashboard unless everything else is done early.
- **Track:** 1 (Detection & Alert Prioritization) — primary and only track.

## Non-negotiable constraints (from the brief — treat as acceptance criteria)
- Use the **organizer split manifest**, not a random row split.
- No row/window may cross a **session, capture, or collection-day boundary**.
- **Domain leakage control:** no raw sld/full query name/hard-coded domain list
  used as a model input feature. Must be masked/excluded.
- Fit all preprocessing (imputation, scaling, encoding, thresholding) on
  **train partition only**.
- Report metrics **separately for light attacks, heavy attacks, and combined**.
- Report PR-AUC, precision/recall/FPR at a chosen threshold, confusion matrix,
  false alerts per 10,000 benign. Accuracy/ROC-AUC supplementary only.
- Report latency (median + p95) and throughput, with and without feature
  extraction time.
- Detection output is **advisory only** — no autonomous blocking logic anywhere
  in the code, even as a stub.

## Roles (see 02_TASK_SPLIT.md for detail)
- **Person A — Data Integrity & Evaluation owner**
- **Person B — Model & Explanation owner**
- Both write tests together in Hour 1 before splitting (see 03_TEST_SPECS.md).

## Timeline (12 hours)

| Time block | Together | Person A | Person B |
|---|---|---|---|
| 0:00–0:45 | Download data, read manifest/schema, agree on data contract (01), write test skeletons (03) | | |
| 0:45–3:30 | | Split loader + leakage checks + baseline rule → baseline metrics running end-to-end | Feature prep + model training against a mocked CSV matching the agreed schema |
| 3:30–4:00 | Integration: plug B's model into A's real split, fix mismatches | | |
| 4:00–6:00 | | Metrics module (PR-AUC, precision/recall/FPR, light/heavy split, FPR/10k) + latency harness | Threshold calibration on validation split + reason codes (top features per alert) |
| 6:00–7:30 | Full pipeline run on held-out-day split, sanity-check numbers | | |
| 7:30–9:00 | | Alert ranking / dedup to decision unit | Reproducibility package (lockfile, seeds, run command) |
| 9:00–10:30 | Technical report (architecture, split protocol, excluded fields, baseline vs model, failure analysis, limitations) | | |
| 10:30–11:30 | Results bundle (predictions, scoring output), data/model statement, checksums | | |
| 11:30–12:00 | Buffer, final repro run, pitch notes | | |

## Definition of "done" for the 6-hour checkpoint
By 6:00 you should be able to run one command and get:
baseline metrics + model metrics, split by light/heavy/combined, on the real
organizer split. If you're not there, cut scope (drop reason codes or latency
breakdown) rather than cutting the evaluation protocol — that's 25% of the rubric.

## Deliverables checklist (map to organizer requirements)
- [ ] Working prototype (1 benign + 1 light + 1 heavy case demoable)
- [ ] Source + repro package (lockfile, seeds, one command to reproduce score)
- [ ] Technical report
- [ ] Results bundle (predictions for organizer test IDs + scoring output)
- [ ] Data/model statement (snapshot version, SHA-256, external deps declared)
- [ ] Pitch/demo notes
