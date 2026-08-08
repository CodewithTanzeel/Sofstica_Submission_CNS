# Leakage Audit

Required by the challenge brief's "Mandatory evaluation protocol" (item 6) and
judged under "AI & Data Quality" (25%) and "Safety & Reliability" (15%).

## Split methodology (why it deviates from a literal session/day manifest)

The organizer brief describes a split manifest keyed by `session_id`,
`registrable_domain`, and `collection_day`. **The distributed CIC-Bell-DNS-EXF-2021
snapshot used for this event does not contain those columns** — the stateless
and stateful feature CSVs only carry per-row DNS features plus an `sld`
column (see below). Each CSV file in the snapshot corresponds to one capture
(one file-transfer type for attacks, one capture run for benign traffic), so
we treat **each paired stateful+stateless file as a session-equivalent
group** and split at that level. Every row belonging to a given file lands
in exactly one of train/val/test — verified programmatically (see
`tests/test_kaggle_dataset_loader.py::test_no_row_from_same_source_file_crosses_split_boundary`
and `results/group_split_manifest.csv`, which lists every group's split
assignment).

This is a documented limitation: we cannot report a true held-out-day result
because collection-day identity is not present in this snapshot. We can and
do guarantee no capture/session ever straddles a partition, which is the
leakage vector the brief is most concerned about.

## Excluded fields (never used as model input)

| Field | Reason for exclusion |
|---|---|
| `row_id` | Synthetic row index assigned at load time — not a real feature. |
| `source_file` | Directly encodes the file (and therefore, via filename, the attack type / label) — a pure shortcut. |
| `session_id` | Our file-group proxy key used only for the split, never fed to the model. |
| `registrable_domain` | Same proxy key as `session_id` (see above) — split-only, never a model input. |
| `collection_day` | Not available in this snapshot; set to a constant placeholder `"unknown"` and never used. |
| `split` | Split membership itself — obvious label leakage if used as a feature. |
| `label`, `label_detail` | The prediction targets. |
| `sld` | **Raw domain identity.** Confirmed by direct inspection: in several files this column holds literal strings (e.g. a device hostname, and base32-style tunnel-encoded exfiltration subdomains), not just the length or a hashed value. This is exactly the "registered testbed domain" shortcut the brief prohibits — masked/excluded explicitly, not by accidental dtype filtering. |
| `subdomain` | Raw subdomain/query string — same domain-identity leakage risk as `sld`. |
| `timestamp` | Capture timestamp — the brief explicitly lists timestamps as a non-valid predictive shortcut (it can correlate with collection day / attack campaign ordering). |

All exclusions are enforced in code in two places (`src/features.py` and
`src/model.py`, `excluded_columns` set) — deliberately, not as a side effect
of numeric-dtype filtering. A prior version of this codebase excluded `sld`
only because pandas happened to infer an object dtype for it in some files;
that was accidental and would have broken silently if the snapshot's mixed
int/string `sld` values had all come out numeric. It is now excluded by name
regardless of dtype.

## Leakage checks actually run

`src/leakage.py::validate_leakage()` is called from `pipeline.py` on every
run, before any modeling happens. It asserts:
- No `session_id` (file-group proxy) appears in more than one of
  train/val/test.
- No `registrable_domain` (same proxy) appears in more than one partition.

If either check fails, the pipeline raises `ValueError` and does not proceed
to train or score a model — this is a hard stop, not a warning.

## Known residual limitation (must be stated in the technical report)

Because the true `registrable_domain` field was not distributed, we cannot
prove that the model generalizes to *unseen* domains — only that it
generalizes to unseen *capture files*, which is a weaker but real form of
leakage control. This must be stated explicitly in the technical report's
limitations section, per the brief's "claims must match evidence"
requirement.
