"""Candidate forecaster: simple exponential smoothing (flat forecast)."""
from __future__ import annotations

import pandas as pd


def ses_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str, alpha: float) -> pd.Series:
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    values = train[target].astype(float).to_numpy()
    if len(values) == 0:
        raise ValueError("train must not be empty")
    level = values[0]
    for v in values[1:]:
        level = alpha * v + (1 - alpha) * level
    return pd.Series(float(level), index=template.index, dtype=float)