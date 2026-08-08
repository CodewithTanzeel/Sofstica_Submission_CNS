import pandas as pd
import pytest


@pytest.fixture
def sample_manifest():
    return pd.DataFrame(
        [
            {"row_id": 1, "split": "train"},
            {"row_id": 2, "split": "train"},
            {"row_id": 3, "split": "val"},
            {"row_id": 4, "split": "val"},
            {"row_id": 5, "split": "test"},
            {"row_id": 6, "split": "test"},
        ]
    )


@pytest.fixture
def feature_table():
    return pd.DataFrame(
        [
            {"row_id": 1, "capture_id": "cap-1", "session_id": "sess-1", "collection_day": 1, "registrable_domain": "alpha.com", "label": 0, "query_length": 20, "entropy": 2.7, "feat_s1": 0.1, "feat_s2": 0.2, "feat_d1": 0.5, "feat_d2": 0.6},
            {"row_id": 2, "capture_id": "cap-1", "session_id": "sess-2", "collection_day": 1, "registrable_domain": "beta.com", "label": 1, "query_length": 80, "entropy": 4.5, "feat_s1": 0.9, "feat_s2": 0.1, "feat_d1": 0.2, "feat_d2": 0.8},
            {"row_id": 3, "capture_id": "cap-2", "session_id": "sess-3", "collection_day": 2, "registrable_domain": "gamma.com", "label": 0, "query_length": 22, "entropy": 2.9, "feat_s1": 0.3, "feat_s2": 0.4, "feat_d1": 0.7, "feat_d2": 0.3},
            {"row_id": 4, "capture_id": "cap-2", "session_id": "sess-4", "collection_day": 2, "registrable_domain": "delta.com", "label": 2, "query_length": 110, "entropy": 5.0, "feat_s1": 0.6, "feat_s2": 0.7, "feat_d1": 0.4, "feat_d2": 0.9},
            {"row_id": 5, "capture_id": "cap-3", "session_id": "sess-5", "collection_day": 3, "registrable_domain": "epsilon.com", "label": 0, "query_length": 18, "entropy": 2.1, "feat_s1": 0.2, "feat_s2": 0.3, "feat_d1": 0.5, "feat_d2": 0.2},
            {"row_id": 6, "capture_id": "cap-3", "session_id": "sess-6", "collection_day": 3, "registrable_domain": "zeta.com", "label": 1, "query_length": 90, "entropy": 4.8, "feat_s1": 0.8, "feat_s2": 0.5, "feat_d1": 0.6, "feat_d2": 0.7},
        ]
    )


@pytest.fixture
def loaded_split(sample_manifest, feature_table):
    from src.load import load_split

    return load_split(sample_manifest, feature_table)


@pytest.fixture
def excluded_fields():
    return ["collection_day", "capture_id", "session_id", "registrable_domain", "label", "row_id"]


@pytest.fixture
def mock_split(loaded_split):
    return loaded_split


@pytest.fixture
def baseline_config():
    return {"rule": "query_length + entropy", "threshold": 0.55}
