"""Generate predictions.csv.

Run: python -m cwfc.predict --data challenge_data --output predictions.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .baselines import persistence_forecast
from .data import HOLDOUT_STEPS, build_targets, load_readings, split_train_holdout

COLUMNS = ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]


def make_predictions(train: pd.DataFrame, holdout_timestamps: pd.Series) -> pd.DataFrame:
    """Persistence forecast for both targets.

    Only the holdout TIMESTAMPS are used, never its target values.
    """
    template = pd.DataFrame({"timestamp": holdout_timestamps.to_numpy()})
    out = template.copy()
    for target in ("active_vm_count", "aggregate_cpu_utilization"):
        history = train.dropna(subset=[target])  # last valid value if the final row is NaN
        out[target] = persistence_forecast(history, template, target).to_numpy()
    out["active_vm_count"] = out["active_vm_count"].clip(lower=0)
    out["aggregate_cpu_utilization"] = out["aggregate_cpu_utilization"].clip(0, 100)
    return out[COLUMNS]


def validate_predictions(pred: pd.DataFrame, expected_rows: int) -> None:
    if list(pred.columns) != COLUMNS:
        raise ValueError(f"columns must be {COLUMNS}")
    if len(pred) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {len(pred)}")
    if pred["timestamp"].duplicated().any():
        raise ValueError("duplicate timestamps")
    if not pred["timestamp"].is_monotonic_increasing:
        raise ValueError("timestamps must be in chronological order")
    if not np.isfinite(pred[COLUMNS[1:]].to_numpy(dtype=float)).all():
        raise ValueError("predictions must be finite")
    if (pred["active_vm_count"] < 0).any():
        raise ValueError("active_vm_count must be >= 0")
    if not pred["aggregate_cpu_utilization"].between(0, 100).all():
        raise ValueError("aggregate_cpu_utilization must be in [0, 100]")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Write predictions.csv for the final holdout.")
    p.add_argument("--data", required=True, help="folder with vm_cpu_readings.csv.gz")
    p.add_argument("--output", default="predictions.csv")
    p.add_argument("--holdout-steps", type=int, default=HOLDOUT_STEPS)
    a = p.parse_args(argv)

    series = build_targets(load_readings(a.data))
    train, holdout = split_train_holdout(series, a.holdout_steps)
    pred = make_predictions(train, holdout["timestamp"])
    validate_predictions(pred, a.holdout_steps)
    pred.to_csv(a.output, index=False)
    print(f"wrote {len(pred)} rows to {a.output}")


if __name__ == "__main__":
    main()