import numpy as np
import pandas as pd

from cwfc.data import build_targets, split_train_holdout


def test_duplicate_vm_counted_once():
    r = pd.DataFrame(
        {
            "timestamp": [0, 0, 0, 300],
            "vm_id": ["a", "a", "b", "a"],
            "avg_cpu": [10.0, 10.0, 30.0, 50.0],
        }
    )
    t = build_targets(r)
    assert list(t["active_vm_count"]) == [2, 1]
    assert list(t["aggregate_cpu_utilization"]) == [20.0, 50.0]


def test_invalid_readings_dropped_not_filled():
    r = pd.DataFrame(
        {
            "timestamp": [0, 0, 0, 300],
            "vm_id": ["a", "b", "c", "a"],
            "avg_cpu": [10.0, np.nan, 150.0, 40.0],
        }
    )
    t = build_targets(r)
    assert list(t["active_vm_count"]) == [1, 1]
    assert list(t["aggregate_cpu_utilization"]) == [10.0, 40.0]


def test_timestamp_with_no_valid_reading():
    r = pd.DataFrame(
        {"timestamp": [0, 300], "vm_id": ["a", "a"], "avg_cpu": [10.0, np.nan]}
    )
    t = build_targets(r)
    assert list(t["active_vm_count"]) == [1, 0]
    assert pd.isna(t["aggregate_cpu_utilization"].iloc[1])


def test_split_is_chronological_110_24():
    s = pd.DataFrame(
        {
            "timestamp": np.arange(134) * 300,
            "active_vm_count": 10000,
            "aggregate_cpu_utilization": 1.0,
        }
    )
    train, hold = split_train_holdout(s)
    assert (len(train), len(hold)) == (110, 24)
    assert train["timestamp"].max() < hold["timestamp"].min()
    assert list(hold["timestamp"]) == list(s["timestamp"].iloc[-24:])