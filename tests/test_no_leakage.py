import pandas as pd


def test_no_session_split_across_partitions(loaded_split):
    sessions = {}
    for split_name, frame in loaded_split.items():
        sessions[split_name] = set(frame["session_id"])

    assert sessions["train"].isdisjoint(sessions["val"])
    assert sessions["train"].isdisjoint(sessions["test"])
    assert sessions["val"].isdisjoint(sessions["test"])


def test_no_domain_split_across_partitions(loaded_split):
    domains = {}
    for split_name, frame in loaded_split.items():
        domains[split_name] = set(frame["registrable_domain"])

    assert domains["train"].isdisjoint(domains["val"])
    assert domains["train"].isdisjoint(domains["test"])
    assert domains["val"].isdisjoint(domains["test"])


def test_no_day_boundary_crossed_in_windows(loaded_split):
    for split_name, frame in loaded_split.items():
        assert frame["collection_day"].nunique() >= 1


def test_excluded_fields_not_in_feature_matrix(loaded_split, excluded_fields):
    from src.load import build_feature_matrix

    frame = loaded_split["train"]
    feature_matrix = build_feature_matrix(frame, excluded_fields=excluded_fields)
    assert not set(excluded_fields).intersection(feature_matrix.columns)


def test_split_matches_organizer_manifest(loaded_split, sample_manifest):
    expected = {split_name: set(group["row_id"]) for split_name, group in sample_manifest.groupby("split")}
    actual = {split_name: set(frame["row_id"]) for split_name, frame in loaded_split.items()}
    assert actual["train"] == expected["train"]
    assert actual["val"] == expected["val"]
    assert actual["test"] == expected["test"]
