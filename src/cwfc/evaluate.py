"""Rolling-origin evaluation by row position (leakage-safe)."""
from __future__ import annotations

import pandas as pd

from .metrics import mae, rmse


def index_splits(n: int, horizon: int = 24, min_train: int = 50, step: int = 12):
    """Expanding windows. Each split is (train_end, val_end):
    fit on rows[:train_end], validate on rows[train_end:val_end]."""
    if horizon <= 0 or min_train <= 0 or step <= 0:
        raise ValueError("horizon, min_train and step must be positive")
    out, end = [], min_train
    while end + horizon <= n:
        out.append((end, end + horizon))
        end += step
    return out


def rolling_origin_eval(train, target, forecasters, horizon=24, min_train=50, step=12):
    """forecasters: {name: fn(fit_df, template_df, target) -> Series}."""
    rows = []
    for k, (tr_end, val_end) in enumerate(index_splits(len(train), horizon, min_train, step)):
        fit = train.iloc[:tr_end]
        val = train.iloc[tr_end:val_end]
        for name, fn in forecasters.items():
            pred = fn(fit, val, target)
            rows.append(
                {
                    "fold": k,
                    "model": name,
                    "mae": mae(val[target], pred),
                    "rmse": rmse(val[target], pred),
                }
            )
    return pd.DataFrame(rows)