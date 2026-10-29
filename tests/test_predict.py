import numpy as np
import pandas as pd
import pytest

from cwfc.predict import COLUMNS, main, make_predictions, validate_predictions


def _train(n=10):
    return pd.DataFrame(
        {
            "timestamp": np.arange(n) * 300,
            "active_vm_count": [10] * n,
            "aggregate_cpu_utilization": np.linspace(5, 9, n),
        }
    )


def test_forecast_is_last_train_value():
    pred = make_predictions(_train(), pd.Series([3000, 3300, 3600]))
    assert list(pred.columns) == COLUMNS
    assert list(pred["timestamp"]) == [3000, 3300, 3600]
    assert (pred["aggregate_cpu_utilization"] == 9.0).all()
    assert (pred["active_vm_count"] == 10).all()


def test_nan_at_end_falls_back_to_last_valid_value():
    train = _train()
    train.loc[train.index[-1], "aggregate_cpu_utilization"] = np.nan
    pred = make_predictions(train, pd.Series([3000]))
    assert pred["aggregate_cpu_utilization"].iloc[0] == pytest.approx(np.linspace(5, 9, 10)[-2])


def test_validate_rejects_out_of_range():
    pred = make_predictions(_train(), pd.Series([3000, 3300]))
    pred.loc[0, "aggregate_cpu_utilization"] = 120.0
    with pytest.raises(ValueError):
        validate_predictions(pred, 2)


def test_cli_end_to_end_on_tiny_dataset(tmp_path):
    rows = [
        {"timestamp": t * 300, "vm_id": f"vm{v}", "min_cpu": 1.0, "max_cpu": 9.0, "avg_cpu": 5.0 + t * 0.1}
        for t in range(40)
        for v in range(3)
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False)
    out = tmp_path / "predictions.csv"
    main(["--data", str(tmp_path), "--output", str(out), "--holdout-steps", "5"])
    pred = pd.read_csv(out)
    assert list(pred.columns) == COLUMNS
    assert list(pred["timestamp"]) == [t * 300 for t in range(35, 40)]
    assert (pred["active_vm_count"] == 3).all()
    assert pred["aggregate_cpu_utilization"].iloc[0] == pytest.approx(8.4)