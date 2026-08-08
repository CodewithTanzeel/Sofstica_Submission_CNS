from pathlib import Path

import pandas as pd

from src.load import load_kaggle_dataset


def test_load_kaggle_dataset_creates_splits_and_labels(tmp_path: Path):
    dataset_dir = tmp_path / "kaggle_dataset"
    (dataset_dir / "Benign").mkdir(parents=True)
    (dataset_dir / "Attack_Light_Benign").mkdir(parents=True)
    (dataset_dir / "Attack_heavy_Benign").mkdir(parents=True)

    pd.DataFrame({"query_length": [10, 12], "entropy": [1.0, 1.5], "label": [0, 0]}).to_csv(
        dataset_dir / "Benign" / "benign.csv", index=False
    )
    pd.DataFrame({"query_length": [50, 60], "entropy": [3.2, 3.8], "label": [1, 1]}).to_csv(
        dataset_dir / "Attack_Light_Benign" / "light.csv", index=False
    )
    pd.DataFrame({"query_length": [120, 130], "entropy": [5.0, 5.4], "label": [2, 2]}).to_csv(
        dataset_dir / "Attack_heavy_Benign" / "heavy.csv", index=False
    )

    split_frames = load_kaggle_dataset(dataset_dir)

    assert set(split_frames.keys()) == {"train", "val", "test"}
    assert all("label" in frame.columns for frame in split_frames.values())
    assert all("label_detail" in frame.columns for frame in split_frames.values())
    assert all(set(frame["label_detail"].unique()).issubset({0, 1, 2}) for frame in split_frames.values())


def test_benign_subfolder_under_attack_campaign_is_labeled_benign(tmp_path: Path):
    """Regression test for the label-inference bug: a file sitting in
    Attack_Light_Benign/Benign/ (or Attack_heavy_Benign/Benign/) is genuine
    benign traffic and must get label_detail=0, NOT be swept up as an attack
    just because it lives under a folder whose name contains "attack_light"
    or "attack_heavy". A naive substring match on the full path mislabels
    real benign traffic as attack traffic.
    """
    dataset_dir = tmp_path / "kaggle_dataset"
    (dataset_dir / "Attack_Light_Benign" / "Attacks").mkdir(parents=True)
    (dataset_dir / "Attack_Light_Benign" / "Benign").mkdir(parents=True)
    (dataset_dir / "Attack_heavy_Benign" / "Attacks").mkdir(parents=True)
    (dataset_dir / "Attack_heavy_Benign" / "Benign").mkdir(parents=True)

    pd.DataFrame({"query_length": [50, 60], "entropy": [3.2, 3.8]}).to_csv(
        dataset_dir / "Attack_Light_Benign" / "Attacks" / "light_attack.csv", index=False
    )
    pd.DataFrame({"query_length": [8, 9], "entropy": [0.9, 1.0]}).to_csv(
        dataset_dir / "Attack_Light_Benign" / "Benign" / "light_benign.csv", index=False
    )
    pd.DataFrame({"query_length": [120, 130], "entropy": [5.0, 5.4]}).to_csv(
        dataset_dir / "Attack_heavy_Benign" / "Attacks" / "heavy_attack.csv", index=False
    )
    pd.DataFrame({"query_length": [7, 6], "entropy": [0.8, 0.7]}).to_csv(
        dataset_dir / "Attack_heavy_Benign" / "Benign" / "heavy_benign.csv", index=False
    )

    split_frames = load_kaggle_dataset(dataset_dir)
    all_rows = pd.concat(split_frames.values(), ignore_index=True)

    benign_from_light_campaign = all_rows[all_rows["source_file"] == "light_benign.csv"]
    benign_from_heavy_campaign = all_rows[all_rows["source_file"] == "heavy_benign.csv"]
    attack_from_light_campaign = all_rows[all_rows["source_file"] == "light_attack.csv"]
    attack_from_heavy_campaign = all_rows[all_rows["source_file"] == "heavy_attack.csv"]

    assert (benign_from_light_campaign["label_detail"] == 0).all()
    assert (benign_from_heavy_campaign["label_detail"] == 0).all()
    assert (attack_from_light_campaign["label_detail"] == 1).all()
    assert (attack_from_heavy_campaign["label_detail"] == 2).all()


def test_no_row_from_same_source_file_crosses_split_boundary(tmp_path: Path):
    """Every row from a given file (proxy exfiltration session) must land in
    exactly one of train/val/test — this is the mandatory group-integrity
    rule from the challenge brief."""
    dataset_dir = tmp_path / "kaggle_dataset"
    (dataset_dir / "Benign").mkdir(parents=True)
    (dataset_dir / "Attack_Light_Benign").mkdir(parents=True)
    (dataset_dir / "Attack_heavy_Benign").mkdir(parents=True)

    for i in range(3):
        pd.DataFrame({"query_length": range(10), "entropy": [1.0] * 10}).to_csv(
            dataset_dir / "Benign" / f"benign_{i}.csv", index=False
        )
        pd.DataFrame({"query_length": range(10), "entropy": [3.0] * 10}).to_csv(
            dataset_dir / "Attack_Light_Benign" / f"light_{i}.csv", index=False
        )
        pd.DataFrame({"query_length": range(10), "entropy": [5.0] * 10}).to_csv(
            dataset_dir / "Attack_heavy_Benign" / f"heavy_{i}.csv", index=False
        )

    split_frames = load_kaggle_dataset(dataset_dir)
    file_to_splits = {}
    for split_name, frame in split_frames.items():
        for source_file in frame["source_file"].unique():
            file_to_splits.setdefault(source_file, set()).add(split_name)

    assert all(len(splits) == 1 for splits in file_to_splits.values())
