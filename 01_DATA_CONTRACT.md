# Data Contract (fill in once you have the real files, then freeze it)

Both of you code against this file, not against each other. If reality differs
from the organizer's actual schema, update this doc first, then both branches.

## To fill in after downloading the organizer package
- [ ] Snapshot version: ____
- [ ] Archive SHA-256: ____
- [ ] Location of split manifest file: ____
- [ ] Location of feature tables (stateless): ____
- [ ] Location of feature tables (stateful): ____
- [ ] Label column name: ____
- [ ] Label values (e.g. benign / light / heavy, or 0/1/2): ____

## Required columns for the split/session/day boundaries (rename to actual names)
| Purpose | Expected column name (placeholder) |
|---|---|
| Row/query unique ID | `row_id` |
| Capture/PCAP file ID | `capture_id` |
| Exfiltration session ID | `session_id` |
| Collection day | `collection_day` |
| Partition assignment (train/val/test, from organizer manifest) | `split` |
| Registrable benign domain (for domain-leakage control) | `registrable_domain` |
| Label | `label` |

## Excluded / masked fields (must NOT be model inputs — leakage risk)
List every field here as you find it in the schema. Minimum known from the brief:
- `collection_day` / any date-time column
- `capture_id` / capture filename
- `session_id` (usable for grouping/splitting, never as a feature)
- raw `sld`, full query name, any hard-coded testbed IP/domain list
- row order / index
- the `label` column itself, obviously

> Rule of thumb: if a field could tell you the label without describing the
> DNS *behavior*, it's excluded. Behavior = query length, entropy, request
> rate, response codes, TTLs, record types, timing/window stats, etc.

## Mocked schema for Person B (use until real split loader is ready)
Person B should generate a small synthetic CSV with these exact column names
and dtypes so their training script isn't blocked on Person A's loader:

```
row_id: int
capture_id: str
session_id: str
collection_day: int (1-5)
split: str ("train" | "val" | "test")
registrable_domain: str
label: int (0=benign, 1=light, 2=heavy)  # collapse to binary for Track 1 scoring
<...14 stateless feature columns, placeholder names feat_s1..feat_s14, float>
<...16 stateful feature columns, placeholder names feat_d1..feat_d16, float>
```

Swap placeholder feature names for real ones the moment the real schema is
confirmed — update this file and grep-replace.

## Binary label convention for Track 1
- `label == 0` → benign (negative class)
- `label in {1, 2}` → exfiltration (positive class)
- Keep the original 3-way label around as `label_detail` so light/heavy can be
  scored separately per the brief's requirement.

## Interfaces both sides must honor
- **A's split loader** returns train/val/test DataFrames with `registrable_domain`
  and `session_id` never overlapping across partitions (tested in
  `test_no_leakage.py`).
- **B's feature prep** takes a DataFrame with the schema above, fits only on the
  train partition, and returns transformed train/val/test + the fitted
  transformer object (for later reuse in the results bundle).
- **A's metrics module** takes `(y_true, y_score, label_detail, split_name)` and
  returns a dict with PR-AUC, precision/recall/FPR at threshold, confusion
  matrix, per-class (light/heavy/combined) breakdown, FPR per 10k benign.
