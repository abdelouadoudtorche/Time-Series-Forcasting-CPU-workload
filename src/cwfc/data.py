"""Load VM readings, build the two aggregate targets, split train/holdout."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HOLDOUT_STEPS = 24


def load_readings(data_dir) -> pd.DataFrame:
    path = Path(data_dir) / "vm_cpu_readings.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    return pd.read_csv(path)


def build_targets(readings: pd.DataFrame) -> pd.DataFrame:
    """One row per timestamp: active_vm_count and aggregate_cpu_utilization.

    Rules:
    - each (timestamp, vm_id) pair is counted once;
    - a reading is valid if avg_cpu is within [0, 100] (NaN is invalid);
    - invalid readings are dropped, never filled in.
    """
    df = readings[["timestamp", "vm_id", "avg_cpu"]].drop_duplicates(
        ["timestamp", "vm_id"], keep="first"
    )
    all_ts = np.sort(df["timestamp"].unique())
    valid = df[df["avg_cpu"].between(0, 100)]
    g = valid.groupby("timestamp")
    out = pd.DataFrame(
        {
            "active_vm_count": g["vm_id"].nunique(),
            "aggregate_cpu_utilization": g["avg_cpu"].mean(),
        }
    ).reindex(all_ts)
    out["active_vm_count"] = out["active_vm_count"].fillna(0).astype(int)
    out.index.name = "timestamp"
    return out.reset_index()


def split_train_holdout(series: pd.DataFrame, holdout_steps: int = HOLDOUT_STEPS):
    """Chronological split: the last `holdout_steps` rows are the holdout."""
    if holdout_steps <= 0 or holdout_steps >= len(series):
        raise ValueError("holdout_steps must be between 1 and len(series) - 1")
    s = series.sort_values("timestamp").reset_index(drop=True)
    return s.iloc[:-holdout_steps].copy(), s.iloc[-holdout_steps:].copy()