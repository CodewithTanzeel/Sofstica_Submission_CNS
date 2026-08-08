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
