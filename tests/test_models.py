import pandas as pd
import pytest

from cwfc.baselines import persistence_forecast
from cwfc.models import ses_forecast


def test_alpha_one_equals_persistence():
    train = pd.DataFrame({"y": [1.0, 5.0, 3.0, 4.0]})
    tpl = pd.DataFrame({"y": [0, 0]}, index=[4, 5])
    assert list(ses_forecast(train, tpl, "y", 1.0)) == list(persistence_forecast(train, tpl, "y"))


def test_ses_known_value():
    train = pd.DataFrame({"y": [10.0, 20.0]})
    tpl = pd.DataFrame({"y": [0]}, index=[2])
    assert ses_forecast(train, tpl, "y", 0.5).iloc[0] == 15.0


def test_bad_alpha_rejected():
    train = pd.DataFrame({"y": [1.0, 2.0]})
    tpl = pd.DataFrame({"y": [0]}, index=[2])
    with pytest.raises(ValueError):
        ses_forecast(train, tpl, "y", 0.0)