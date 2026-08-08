from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_split(manifest: pd.DataFrame, feature_table: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Load a manifest-based split and join it with the feature table."""
    if "row_id" not in manifest.columns or "split" not in manifest.columns:
        raise ValueError("manifest must contain row_id and split")

    if "row_id" not in feature_table.columns:
        raise ValueError("feature_table must contain row_id")

    merged = manifest[["row_id", "split"]].merge(feature_table, on="row_id", how="inner")
    result: Dict[str, pd.DataFrame] = {}

    for split_name in ["train", "val", "test"]:
        subset = merged.loc[merged["split"] == split_name].copy()
        subset = subset.sort_values("row_id").reset_index(drop=True)
        result[split_name] = subset

    return result


def _infer_label_from_path(path: Path, dataset_root: Path) -> int:
    relative_parts = [part.lower() for part in path.relative_to(dataset_root).parts]
    relative_str = "-".join(relative_parts)
    if "attack_light_benign" in relative_str or ("attack" in relative_str and "light" in relative_str):
        return 1
    if "attack_heavy_benign" in relative_str or ("attack" in relative_str and "heavy" in relative_str):
        return 2
    return 0


def _normalize_feature_key(path: Path) -> str:
    name = path.name.lower()
    for prefix in ["stateful_features-", "stateless_features-"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace(".pcap.csv", "").replace(".csv", "").lstrip("_")


def load_kaggle_dataset(dataset_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """Load the Kaggle-style folder structure into train/val/test splits."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    csv_paths = sorted(dataset_dir.rglob("*.csv"))
    if not csv_paths:
        raise ValueError(f"No CSV files found under {dataset_dir}")

    grouped_files: Dict[str, Dict[str, Path]] = {}
    for csv_path in csv_paths:
        if any(part in {".git", ".venv", "__pycache__"} for part in csv_path.parts):
            continue
        if csv_path.name.lower().startswith("stateful_features"):
            feature_type = "stateful"
        elif csv_path.name.lower().startswith("stateless_features"):
            feature_type = "stateless"
        else:
            feature_type = "single"
        grouped_files.setdefault(_normalize_feature_key(csv_path), {})[feature_type] = csv_path

    frames: List[pd.DataFrame] = []
    for key, feature_files in grouped_files.items():
        stateful_path = feature_files.get("stateful")
        stateless_path = feature_files.get("stateless")
        single_path = feature_files.get("single")

        if stateful_path is not None and stateless_path is not None:
            source_path = stateful_path
            stateful = pd.read_csv(stateful_path)
            stateless = pd.read_csv(stateless_path)
            if len(stateful) != len(stateless):
                min_len = min(len(stateful), len(stateless))
                stateful = stateful.iloc[:min_len].reset_index(drop=True)
                stateless = stateless.iloc[:min_len].reset_index(drop=True)
            combined = pd.concat([stateful.reset_index(drop=True), stateless.reset_index(drop=True)], axis=1)
        elif single_path is not None:
            source_path = single_path
            combined = pd.read_csv(single_path)
        else:
            continue

        combined["label_detail"] = _infer_label_from_path(source_path, dataset_dir)
        combined["label"] = combined["label_detail"].apply(lambda value: 0 if value == 0 else 1)
        combined["source_file"] = source_path.name
        frames.append(combined)

    if not frames:
        raise ValueError(f"No compatible feature file pairs found under {dataset_dir}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.reset_index().rename(columns={"index": "row_id"})
    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

    split_size = len(combined)
    train_end = int(split_size * 0.7)
    val_end = int(split_size * 0.85)

    train = combined.iloc[:train_end].copy()
    val = combined.iloc[train_end:val_end].copy()
    test = combined.iloc[val_end:].copy()

    for frame in [train, val, test]:
        frame["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"

    return {"train": train, "val": val, "test": test}


def build_feature_matrix(frame: pd.DataFrame, excluded_fields: List[str] | None = None) -> pd.DataFrame:
    """Build a feature matrix by dropping excluded columns and keeping numeric features."""
    excluded_fields = excluded_fields or []
    excluded = set(excluded_fields)
    retained = [col for col in frame.columns if col not in excluded]
    feature_matrix = frame[retained].copy()
    non_numeric = feature_matrix.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric:
        feature_matrix = feature_matrix.drop(columns=non_numeric)
    return feature_matrix
