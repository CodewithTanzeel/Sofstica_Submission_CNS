# DNS Exfiltration Detection — Technical Report

Track: **[FILL IN — Track 1 Detection & Alert Prioritization, unless you're
pivoting to Track 2/3]**

## Problem and user

Network defenders and administrators of small/resource-constrained networks
need to know when DNS traffic is carrying an exfiltration tunnel instead of
normal lookups — without drowning in false alerts. [FILL IN 1–2 sentences
on why this matters to that user specifically, and what decision they make
with the output — e.g. "an analyst triaging alerts needs a ranked list with
a reason, not a raw score."]

## System architecture

- **Data loading (`src/load.py`):** parses the organizer's Kaggle-style
  folder layout (`Benign/`, `Attack_Light_Benign/`, `Attack_heavy_Benign/`),
  pairs each stateful/stateless CSV file, and infers a label from the
  file's immediate parent folder (`Attacks/` vs `Benign/`), not from the
  top-level campaign folder name (see "Split methodology" below — this
  distinction matters and is explained there).
- **Split (`src/load.py::load_kaggle_dataset`):** file-group-level,
  stratified by label class. See below.
- **Leakage guard (`src/leakage.py`):** asserts no session/domain proxy
  crosses a partition boundary; the pipeline hard-stops if it does.
- **Feature prep (`src/features.py`):** median imputation + standard
  scaling, fit on train only, applied to val/test.
- **Baseline (`src/baseline.py`):** non-ML rule on query length + entropy.
- **Model (`src/model.py`):** XGBoost classifier (`SimpleModel`), plus a
  logistic-regression fallback trained alongside it. Threshold selected by
  maximizing F1 on the validation precision-recall curve
  (`select_threshold`).
- **Metrics (`src/metrics.py`):** PR-AUC, precision/recall/FPR at the
  selected threshold, confusion matrix, false alerts per 10k benign, split
  separately for light/heavy/combined.
- **Explanation:** per-alert reason codes (`src/model.py::reason_codes`) —
  top-3 features by importance × observed magnitude, attached to every
  predicted-positive row in `results/predictions.csv`.

## Data used

CIC-Bell-DNS-EXF-2021 (organizer-provided snapshot). 14 stateless + 16
stateful DNS features, benign/light-attack/heavy-attack traffic. Snapshot
version and checksum: see `reports/data_model_statement.md` (fill in from
the organizer download page).

## Split protocol

**Deviation from the literal brief, and why:** the brief describes a split
manifest keyed by `session_id` / `registrable_domain` / `collection_day`.
The distributed snapshot does not include those columns — each CSV file
*is* the unit of capture. We therefore split at the **file-group level**
(each paired stateful+stateless file = one session-equivalent group),
stratified by label class (benign / light / heavy) so every partition has
representation from all three. This guarantees no row from a given file
ever appears in more than one of train/val/test — verified by an automated
test (`tests/test_kaggle_dataset_loader.py`) and by
`results/group_split_manifest.csv`, which lists every group's split
assignment and row count.

**Limitation:** because `collection_day` isn't present in this snapshot, we
cannot report the organizer's held-out-day robustness result — only
file-group-level held-out results. This is stated explicitly here rather
than implied away.

Group counts: 6 benign, 6 light-attack, 6 heavy-attack file groups (18
total). With this few groups, val/test each end up with a small number of
groups per class (see manifest) — result variance across classes should be
read with that in mind; it's not a large held-out set.

## Excluded (leakage) fields

Full audit in `reports/leakage_audit.md`. Summary: `row_id`, `source_file`,
`session_id`, `registrable_domain`, `collection_day`, `split`, `label`,
`label_detail` are structural/target fields, never model inputs. `sld`,
`subdomain`, and `timestamp` are excluded **by name**, deliberately — direct
inspection of the raw data confirmed `sld` sometimes contains literal raw
domain-identity strings (a device hostname, and base32-style
tunnel-encoded exfiltration subdomains in a couple of files), which is
exactly the shortcut the brief prohibits.

## Baseline

Non-ML rule: `clip((query_length/120 + entropy/6) / 2, 0, 1)`, threshold
tuned on validation by F1-maximization (not the fixed 0.55 default —
[FILL IN if you keep the default 0.55 instead, note that decision and why]).

## Metrics

*(Numbers below are from the current pipeline run on the real snapshot —
regenerate before final submission and paste fresh numbers if anything
upstream changes.)*

| Split / model | PR-AUC | Precision | Recall | FPR | Threshold |
|---|---|---|---|---|---|
| Baseline — val | 0.377 | 0.355 | 0.993 | 0.823 | 0.200 |
| Baseline — test | 0.428 | 0.405 | 0.992 | 0.809 | 0.200 |
| Model — val | 0.806 | 0.796 | 0.993 | 0.116 | 0.179 |
| Model — test | 0.810 | 0.823 | 0.997 | 0.119 | 0.179 |

Per-class (test set):

| Attack type | PR-AUC | Precision | Recall | FPR |
|---|---|---|---|---|
| Light — baseline | 0.060 | 0.054 | 0.991 | 0.809 |
| Light — model | 0.286 | 0.282 | 0.998 | 0.119 |
| Heavy — baseline | 0.407 | 0.384 | 0.992 | 0.809 |
| Heavy — model | 0.794 | 0.809 | 0.997 | 0.119 |

**Read honestly:** the model is a large, real improvement over the baseline
on FPR (81% → 12%) and on heavy-attack PR-AUC (0.41 → 0.79). Light-attack
detection is the hard case for both baseline and model — light-attack PR-AUC
tops out around 0.29 for the model, meaning confident precision on light
attacks specifically is still weak even though recall stays high at this
threshold. State this directly rather than burying it — it's exactly the
kind of "transparent limitations" and "credible light-attack detection"
signal the rubric is checking for. [FILL IN one or two sentences of your
own read on *why* light attacks are harder here — e.g. smaller payloads
per query, features closer to benign distribution — if you have time to
look at feature distributions; otherwise state it as an open question.]

## Latency / throughput

Median and p95 per-row inference latency, batch size, hardware, and
software versions are written into `results/scoring_output.json` under
`"latency"` on every pipeline run — pull the live numbers from there rather
than hardcoding them here, since they're hardware-dependent. Timings
reported exclude feature extraction (CSV parse + impute/scale); if you have
time, also report the end-to-end number (including feature prep) per the
brief's "timings both including and excluding feature extraction"
requirement — currently only the excluding-case is wired up.

## Failure analysis

[FILL IN — this is required and currently has zero content. Minimum viable
version: open `results/predictions.csv`, filter to false negatives
(`true_label=1, predicted_label=0`) and false positives
(`true_label=0, predicted_label=1`), look at 3–5 of each, and describe what
they have in common. Light-attack false negatives are the most important
ones to look at given the PR-AUC numbers above.]

## Limitations

- Controlled testbed data — does not establish real-world generalization,
  per the brief's "what the data can/cannot support" section.
- File-group split, not a true session/day/domain manifest split (see
  "Split protocol" above) — we cannot claim unseen-domain generalization.
- Light-attack detection is meaningfully weaker than heavy-attack detection
  (PR-AUC ~0.29 vs ~0.79 on test) — do not overstate light-attack
  performance in the pitch.
- Small number of file-groups (18 total) means val/test results have real
  variance; treat single-run numbers as indicative, not tightly confident.
- No outcome labels — this evaluates detection of represented testbed
  traffic, not breach prevention.

## Human review / disposition

Detection outputs are advisory only — no autonomous blocking, consistent
with the brief's safety rules. [FILL IN your actual demo's review/approval
UI or workflow description if Track 1/3 work adds one.]
