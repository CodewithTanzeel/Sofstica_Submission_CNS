# Test Specs (write these together in Hour 1, before splitting)

Philosophy: each test encodes one mandatory rule from the brief. If a test is
red, the pipeline is not submittable — these are acceptance criteria, not
nice-to-haves. Write them as failing stubs first, then build code to pass them.

Suggested repo layout:
```
/data_contract.md        (copy of 01_DATA_CONTRACT.md, kept in sync)
/src/
  load.py                # A: split loading
  leakage.py             # A: leakage guards
  baseline.py            # A: non-ML rule
  metrics.py             # A: scoring
  latency.py             # A: timing harness
  features.py            # B: feature prep
  model.py               # B: training + inference
  explain.py             # B: reason codes
/tests/
  test_no_leakage.py
  test_pipeline_io.py
  test_metrics.py
  test_baseline.py
  test_model_contract.py
/notebooks/
  01_eda.ipynb
  02_results.ipynb
/reports/
  technical_report.md
  data_model_statement.md
/results/
  predictions.csv
  scoring_output.json
```

## `test_no_leakage.py` (Person A)
```python
def test_no_session_split_across_partitions(loaded_split):
    """A session_id must appear in exactly one of train/val/test."""

def test_no_domain_split_across_partitions(loaded_split):
    """A registrable_domain must appear in exactly one of train/val/test."""

def test_no_day_boundary_crossed_in_windows(loaded_split):
    """If windows are used, no window's rows span two collection_day values."""

def test_excluded_fields_not_in_feature_matrix(feature_matrix, excluded_fields):
    """None of collection_day, capture_id, session_id, raw domain/query,
    row order, label are present as columns in the model input matrix."""

def test_split_matches_organizer_manifest(loaded_split, manifest):
    """Row IDs assigned to each partition match the organizer's manifest,
    not a random re-split."""
```

## `test_pipeline_io.py` (Person A, contract test between A and B)
```python
def test_load_split_returns_expected_columns(loaded_split, data_contract):
    """train/val/test DataFrames contain every column named in the data
    contract (01_DATA_CONTRACT.md)."""

def test_feature_prep_fits_on_train_only(mock_split):
    """Fitting the transformer on train and transform(val) must not raise
    and must not internally re-fit when called on val/test."""

def test_model_predict_returns_score_in_unit_interval(mock_split, model):
    """model.predict_proba (or equivalent) returns scores in [0, 1] for every
    row in val/test."""
```

## `test_metrics.py` (Person A)
```python
def test_metrics_reports_required_fields(y_true, y_score):
    """Output dict contains pr_auc, precision, recall, fpr, confusion_matrix,
    false_alerts_per_10k_benign."""

def test_metrics_split_by_attack_class(y_true, y_score, label_detail):
    """Metrics are computed separately for light, heavy, and combined,
    not just combined."""

def test_threshold_selected_on_validation_not_test(pipeline_run):
    """Confirm the threshold value used for test-set reporting was chosen
    using only the validation split's scores/labels."""
```

## `test_baseline.py` (Person A)
```python
def test_baseline_rule_is_documented(baseline_config):
    """Rule and threshold values are stored/loadable, not hardcoded inline
    with no record — needed for the technical report."""

def test_baseline_beats_random(y_true, baseline_scores):
    """Sanity check: baseline PR-AUC > class prevalence (better than chance)."""
```

## `test_model_contract.py` (Person B)
```python
def test_model_outputs_reason_codes(model, sample_row):
    """For a given flagged row, explain() returns a ranked list of feature
    names that are all present in the (non-excluded) feature set — i.e. no
    invented/hallucinated feature names."""

def test_model_beats_baseline_on_validation(baseline_metrics, model_metrics):
    """Model PR-AUC on validation >= baseline PR-AUC. If not, flag it — don't
    silently ship a worse model."""
```

## Order of implementation (strict TDD loop)
1. Write the test stub (red).
2. Write the minimum code to pass it (green).
3. Refactor only if time allows.
4. Move to the next test.

Do not write `load.py`, `model.py`, etc. before their corresponding test
exists, even as a stub — the point of the 12 hours is that both of you can
trust the other's module without reading its internals, because the tests
describe the contract.
