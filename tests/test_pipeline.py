import pandas as pd
import pytest

from src.config import MIN_LAG_HOURS, TZ
from src.evaluate import window
from src.features.build import assert_no_leakage
from src.models.baseline import SeasonalBaseline
from src.predict import check_submission, to_submission


def test_window_matches_submission():
    cutoff = pd.Timestamp("2026-11-05 23:00", tz=TZ).tz_convert("UTC")
    start, end = window(cutoff)
    assert start.tz_convert(TZ).strftime("%Y-%m-%d %H:%M") == "2026-11-07 00:00"
    assert end.tz_convert(TZ).strftime("%Y-%m-%d %H:%M") == "2026-11-11 23:00"
    assert (end - cutoff) / pd.Timedelta(hours=1) == MIN_LAG_HOURS


def test_short_lags_are_rejected():
    with pytest.raises(ValueError):
        assert_no_leakage([24])
    assert_no_leakage([168])


def test_baseline_submission_format():
    hours = pd.date_range("2026-09-01", "2026-11-05 22:00", freq="h", tz="UTC")
    traffic = pd.concat([
        pd.DataFrame({"arc": arc, "datetime": hours, "debit_horaire": 100.0, "taux_occupation": 5.0, "is_valid": True})
        for arc in ["Champs-Elysées", "Convention", "Saint-Pères"]
    ])
    cutoff = hours[-1]
    start = pd.Timestamp("2026-11-07 00:00", tz=TZ).tz_convert("UTC")
    end = pd.Timestamp("2026-11-11 23:00", tz=TZ).tz_convert("UTC")
    pred = SeasonalBaseline().fit(traffic, cutoff).predict(traffic, cutoff, start, end)
    check_submission(to_submission(pred))


def test_export_checks_block_a_broken_dataset():
    from src.config import ARCS, TARGETS
    from src.data.export_dataset import check
    hours = pd.date_range("2026-11-01 23:00", "2026-11-11 22:00", freq="h", tz="UTC")
    window = hours >= pd.Timestamp("2026-11-06 23:00", tz="UTC")
    df = pd.concat([pd.DataFrame({"arc": arc, "datetime": hours, "debit_horaire": 100.0, "taux_occupation": 5.0,
                                  "is_valid": True, "partie": "historique", "hour": hours.hour})
                    for arc in ARCS], ignore_index=True)
    in_window = df["datetime"] >= pd.Timestamp("2026-11-06 23:00", tz="UTC")
    df.loc[in_window, "partie"] = "a_predire"
    df.loc[in_window, TARGETS] = None
    df.loc[in_window, "is_valid"] = False
    assert window.sum() == 120
    check(df, ["hour"])
    with pytest.raises(AssertionError):
        check(pd.concat([df, df.iloc[[0]]]), ["hour"])                   # doublon
    with pytest.raises(AssertionError):
        check(df.assign(taux_occupation=df["taux_occupation"] * 30), ["hour"])   # occupation > 100 %
