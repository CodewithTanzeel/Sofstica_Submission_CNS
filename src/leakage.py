from __future__ import annotations

import pandas as pd


def validate_leakage(split_frames: dict[str, pd.DataFrame]) -> None:
    """Basic leakage checks for session/domain uniqueness across splits."""
    for split_name, frame in split_frames.items():
        if "session_id" not in frame.columns or "registrable_domain" not in frame.columns:
            raise ValueError(f"{split_name} is missing required leakage columns")

    session_sets = {name: set(frame["session_id"]) for name, frame in split_frames.items()}
    domain_sets = {name: set(frame["registrable_domain"]) for name, frame in split_frames.items()}

    for left_name, left_set in session_sets.items():
        for right_name, right_set in session_sets.items():
            if left_name != right_name and left_set.intersection(right_set):
                raise ValueError(f"session overlap between {left_name} and {right_name}")

    for left_name, left_set in domain_sets.items():
        for right_name, right_set in domain_sets.items():
            if left_name != right_name and left_set.intersection(right_set):
                raise ValueError(f"domain overlap between {left_name} and {right_name}")
