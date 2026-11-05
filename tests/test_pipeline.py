import pandas as pd
import pytest

from src.config import MIN_LAG_HOURS, TZ
from src import evaluate
from src.evaluate import holiday_cutoffs, window
from src.features.build import assert_no_leakage, lag_features, yearly_lag_features
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


def test_short_or_fractional_lags_are_rejected():
    with pytest.raises(ValueError):
        assert_no_leakage([144])    # 143 h réelles après le passage à l'heure d'été
    with pytest.raises(ValueError):
        assert_no_leakage([170])    # pas un nombre entier de jours


def _traffic(start, end, arc="Convention"):
    """Trafic factice dont la valeur est l'heure locale : un lag correct renvoie la même heure locale."""
    hours = pd.date_range(start, end, freq="h", tz="UTC")
    local_hour = hours.tz_convert(TZ).hour.astype(float)
    return pd.DataFrame({"arc": arc, "datetime": hours, "debit_horaire": local_hour,
                         "taux_occupation": local_hour, "is_valid": True})


def test_lags_follow_paris_local_time_across_dst():
    """Après le passage à l'heure d'hiver (25/10/2026), lagN doit viser la même heure locale."""
    traffic = _traffic("2026-09-20", "2026-11-12")
    lags = lag_features(traffic)
    after_dst = traffic["datetime"].dt.tz_convert(TZ) >= pd.Timestamp("2026-10-26", tz=TZ)
    for lag in (168, 336, 504):
        got = lags.loc[after_dst, f"debit_horaire_lag{lag}"].dropna()
        assert len(got) > 100
        assert (got == traffic.loc[got.index, "debit_horaire"]).all(), f"lag{lag} décalé d'une heure"


def test_yearly_lag_follows_paris_local_time():
    traffic = _traffic("2025-09-01", "2026-11-12")
    got = yearly_lag_features(traffic)["debit_horaire_lag52s"]
    ok = got.notna()
    assert ok.sum() > 1000
    assert (got[ok] == traffic.loc[ok, "debit_horaire"]).all()


def test_lag_on_duplicated_and_missing_dst_hours():
    traffic = _traffic("2026-03-01", "2026-04-20")
    lags = lag_features(traffic)["debit_horaire_lag168"]
    local = traffic["datetime"].dt.tz_convert(TZ).dt.tz_localize(None)
    # 05/04 02:00 locale existe, mais 7 jours avant = 29/03 02:00 locale, qui n'existe pas -> NaN
    assert pd.isna(lags[local == pd.Timestamp("2026-04-05 02:00")]).all()
    # Heure en double : 25/10/2026 02:00 existe deux fois, le lag d'une semaine plus tard est leur moyenne
    traffic = _traffic("2026-10-01", "2026-11-05")
    traffic.loc[traffic["datetime"] == pd.Timestamp("2026-10-25 00:00", tz="UTC"), "debit_horaire"] = 10.0
    traffic.loc[traffic["datetime"] == pd.Timestamp("2026-10-25 01:00", tz="UTC"), "debit_horaire"] = 20.0
    lags = lag_features(traffic)["debit_horaire_lag168"]
    local = traffic["datetime"].dt.tz_convert(TZ).dt.tz_localize(None)
    assert (lags[local == pd.Timestamp("2026-11-01 02:00")] == 15.0).all()


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


def test_holiday_folds_are_saturday_to_wednesday_with_a_holiday(monkeypatch):
    # Fériés à date fixe écrits en dur : le test ne dépend d'aucune donnée téléchargée.
    feries = pd.DataFrame({"date": [f"{y}-{md}" for y in range(2021, 2027)
                                    for md in ("01-01", "05-01", "05-08", "07-14", "08-15", "11-01", "11-11", "12-25")]})
    monkeypatch.setattr(evaluate.calendrier, "load", lambda name: feries)
    hours = pd.date_range("2021-06-21", "2026-11-12", freq="h", tz="UTC")
    traffic = pd.DataFrame({"datetime": hours, "is_valid": True})
    cutoffs = holiday_cutoffs(traffic)
    assert len(cutoffs) > 20
    for cutoff in cutoffs:
        assert cutoff.tz_convert(TZ).day_of_week == 3 and cutoff.tz_convert(TZ).hour == 23
        start, end = window(cutoff)
        assert start.tz_convert(TZ).day_of_week == 5 and end.tz_convert(TZ).day_of_week == 2
    assert pd.Timestamp("2025-11-06 23:00", tz=TZ).tz_convert("UTC") in cutoffs   # 11/11/2025


def test_predict_fails_clearly_when_a_source_is_missing(monkeypatch, tmp_path):
    from src import predict
    monkeypatch.setattr(predict, "DATA_DIR", tmp_path)
    with pytest.raises(SystemExit, match="Données manquantes"):
        predict.check_sources()


def test_predict_without_refresh_only_needs_the_exported_dataset(monkeypatch, tmp_path):
    from src import predict
    monkeypatch.setattr(predict, "DATASET_MODELE_PATH", tmp_path / "absent.parquet")
    with pytest.raises(SystemExit, match="introuvable"):
        predict.predict("xgb", "test")


def test_manifestation_lowers_only_the_given_day_from_noon():
    from src.predict import MANIF_FACTORS, apply_manifestation
    start = pd.Timestamp("2026-11-07 00:00", tz=TZ).tz_convert("UTC")
    end = pd.Timestamp("2026-11-11 23:00", tz=TZ).tz_convert("UTC")
    hours = pd.date_range(start, end, freq="h")
    pred = pd.concat([pd.DataFrame({"arc": arc, "datetime": hours, "debit_horaire": 100.0, "taux_occupation": 10.0})
                      for arc in MANIF_FACTORS])
    out = apply_manifestation(pred, "2026-11-07")
    local = out["datetime"].dt.tz_convert(TZ)
    hit = (local.dt.day == 7) & (local.dt.hour >= 12)
    assert (out.loc[hit, "debit_horaire"] == 100 * out.loc[hit, "arc"].map(MANIF_FACTORS)).all()
    assert (out.loc[~hit, ["debit_horaire", "taux_occupation"]] == [100.0, 10.0]).all().all()
    check_submission(to_submission(out))
    with pytest.raises(SystemExit, match="aucune heure"):
        apply_manifestation(pred, "2026-11-12")
