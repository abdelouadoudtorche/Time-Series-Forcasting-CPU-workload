import pandas as pd

from cwfc.baselines import moving_average_forecast
from cwfc.evaluate import index_splits, rolling_origin_eval
from cwfc.baselines import persistence_forecast


def test_index_splits_expected_for_110_rows():
    assert index_splits(110, horizon=24, min_train=50, step=12) == [
        (50, 74), (62, 86), (74, 98), (86, 110)
    ]


def test_splits_never_use_future_rows():
    for tr_end, val_end in index_splits(110):
        assert tr_end < val_end <= 110


def test_moving_average_uses_last_window_only():
    train = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0]})
    template = pd.DataFrame({"y": [0, 0]}, index=[4, 5])
    pred = moving_average_forecast(train, template, "y", window=2)
    assert list(pred) == [3.5, 3.5]


def test_rolling_eval_runs_and_has_one_row_per_fold_model():
    df = pd.DataFrame({"y": [float(i) for i in range(80)]})
    res = rolling_origin_eval(df, "y", {"p": persistence_forecast}, horizon=10, min_train=40, step=10)
    assert set(res["model"]) == {"p"}
    assert len(res) == len(index_splits(80, 10, 40, 10))