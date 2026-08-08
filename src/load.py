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
    """Infer label_detail (0=benign, 1=light attack, 2=heavy attack) from folder structure.

    CRITICAL: the immediate parent directory ("Attacks" vs "Benign") is the
    authoritative signal, NOT the top-level campaign folder name. A naive
    substring check on the full joined path (the previous implementation)
    mislabels every file under Attack_Light_Benign/Benign/* and
    Attack_heavy_Benign/Benign/* as an attack, because the campaign folder
    name itself contains "attack_light_benign" / "attack_heavy_benign" even
    though the file lives in the Benign/ subfolder. That bug silently
    flipped ~91.8k genuinely benign rows to attack-labeled rows.
    """
    relative_parts = [part.lower() for part in path.relative_to(dataset_root).parts]
    if len(relative_parts) < 2:
        raise ValueError(f"Cannot infer label, path too shallow: {path}")

    parent_dir = relative_parts[-2]
    top_dir = relative_parts[0]

    if parent_dir == "benign":
        return 0
    if parent_dir == "attacks":
        if "heavy" in top_dir:
            return 2
        if "light" in top_dir:
            return 1
        raise ValueError(f"Attacks folder with unrecognized campaign: {path}")

    # Fallback for flat layouts without an Attacks/Benign subfolder.
    if top_dir == "benign":
        return 0
    if "heavy" in top_dir:
        return 2
    if "light" in top_dir:
        return 1
    raise ValueError(f"Cannot infer label for path: {path}")


def _normalize_feature_key(path: Path) -> str:
    name = path.name.lower()
    for prefix in ["stateful_features-", "stateless_features-"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace(".pcap.csv", "").replace(".csv", "").lstrip("_")


def load_kaggle_dataset(dataset_dir: str | Path, random_state: int = 42) -> Dict[str, pd.DataFrame]:
    """Load the Kaggle-style folder structure into train/val/test splits.

    IMPORTANT — split methodology:
    The organizer brief prohibits random row-level splits and requires
    grouping by exfiltration session (and controlling domain leakage). The
    raw CIC-Bell-DNS-EXF-2021 snapshot distributed here does NOT include
    session_id / registrable_domain / collection_day columns — each CSV file
    IS the unit of capture (one file = one exfiltration-session / benign
    capture). We therefore treat each paired (stateful+stateless) file group
    as a single "session" and split at the FILE-GROUP level: every row from
    a given file stays together in exactly one of train/val/test. This is
    documented in the technical report as a proxy for the organizer's true
    session/day manifest, which was not distributed with this snapshot.

    The split is stratified by label_detail (0=benign, 1=light, 2=heavy) so
    every partition contains benign, light-attack, and heavy-attack file
    groups, which the brief's required per-class metrics depend on.
    """
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

    groups: Dict[str, pd.DataFrame] = {}
    group_label: Dict[str, int] = {}
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

        label_detail = _infer_label_from_path(source_path, dataset_dir)
        combined["label_detail"] = label_detail
        combined["label"] = 0 if label_detail == 0 else 1
        combined["source_file"] = source_path.name
        # Session/domain proxy required by src/leakage.py and the brief's
        # group-integrity rule. Real session_id / registrable_domain columns
        # do not exist in this snapshot, so the file-group key stands in for
        # both, guaranteeing (by construction of the split below) that no
        # "session" or "domain" ever crosses a train/val/test boundary.
        combined["session_id"] = key
        combined["registrable_domain"] = key
        combined["collection_day"] = "unknown"  # not available in this snapshot; documented limitation

        groups[key] = combined
        group_label[key] = label_detail

    if not groups:
        raise ValueError(f"No compatible feature file pairs found under {dataset_dir}")

    # Stratified group split: for each label_detail class, shuffle its file
    # groups deterministically and slice ~70/15/15, keeping at least one
    # group in val/test whenever the class has multiple groups. No row from
    # any given file ever appears in more than one partition.
    rng = __import__("numpy").random.RandomState(random_state)
    split_of_group: Dict[str, str] = {}
    for label_value in sorted(set(group_label.values())):
        class_keys = sorted([k for k, v in group_label.items() if v == label_value])
        rng.shuffle(class_keys)
        n = len(class_keys)
        if n == 1:
            split_of_group[class_keys[0]] = "train"
            continue
        n_val = max(1, round(n * 0.15))
        n_test = max(1, round(n * 0.15))
        n_val = min(n_val, n - 1)
        n_test = min(n_test, n - n_val - 1) if n - n_val - 1 >= 1 else max(0, n - n_val - 1)
        n_train = n - n_val - n_test
        for k in class_keys[:n_train]:
            split_of_group[k] = "train"
        for k in class_keys[n_train:n_train + n_val]:
            split_of_group[k] = "val"
        for k in class_keys[n_train + n_val:]:
            split_of_group[k] = "test"

    frames: List[pd.DataFrame] = []
    for key, frame in groups.items():
        frame = frame.copy()
        frame["split"] = split_of_group[key]
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.reset_index().rename(columns={"index": "row_id"})

    result: Dict[str, pd.DataFrame] = {}
    for split_name in ["train", "val", "test"]:
        subset = combined.loc[combined["split"] == split_name].copy()
        subset = subset.sort_values("row_id").reset_index(drop=True)
        result[split_name] = subset

    return result


def group_manifest(split_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a small file-group -> split manifest, useful for the leakage audit doc."""
    rows = []
    for split_name, frame in split_frames.items():
        for key, label_detail in frame.groupby("session_id")["label_detail"].first().items():
            rows.append({"session_id": key, "split": split_name, "label_detail": int(label_detail), "n_rows": int((frame["session_id"] == key).sum())})
    return pd.DataFrame(rows).sort_values(["split", "session_id"]).reset_index(drop=True)


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
