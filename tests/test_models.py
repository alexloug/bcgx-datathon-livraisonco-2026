import numpy as np
import pandas as pd

from src.config import ARCS, TARGETS, TZ
from src.models.preprocessing import Preprocessor, feature_columns
from src.models.xgb import XGBModel


def fake_dataset(weeks=8, seed=0):
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2026-08-03", periods=24 * 7 * weeks, freq="h", tz="UTC")
    frames = []
    for i, arc in enumerate(ARCS):
        local = hours.tz_convert(TZ)
        base = 300 + 200 * np.sin(np.pi * local.hour / 24) + 100 * i
        df = pd.DataFrame({"arc": arc, "datetime": hours, "debit_horaire": base + rng.normal(0, 10, len(hours)),
                           "taux_occupation": base / 100, "hour": local.hour, "dow": local.day_of_week,
                           "ferie": False, "meteo_obs_temperature_2m": 10.0, "meteo_temperature_2m": 10.0,
                           "source_cible": "mesure", "partie": "historique"})
        for col in TARGETS:
            for lag in [168, 336, 504]:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
            df[f"{col}_lag_mean"] = df[[f"{col}_lag{lag}" for lag in [168, 336, 504]]].mean(axis=1)
            df[f"{col}_niveau28j"] = df[col].shift(144).rolling(24 * 7, min_periods=24).mean()
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["is_valid"] = True
    return out


def test_features_exclude_targets_ids_and_observed_weather():
    cols = feature_columns(fake_dataset(weeks=4))
    assert not set(cols) & {"arc", "datetime", "is_valid", "source_cible", "partie", *TARGETS}
    assert not any(c.startswith("meteo_obs_") for c in cols)
    assert "meteo_temperature_2m" in cols


def test_preprocessor_leaves_no_nan_and_keeps_columns_stable():
    df = fake_dataset()
    cutoff = df["datetime"].max() - pd.Timedelta(days=7)
    prep = Preprocessor().fit(df[df["datetime"] <= cutoff])
    X_train = prep.transform(df[df["datetime"] <= cutoff])
    X_new = prep.transform(df[df["datetime"] > cutoff].assign(debit_horaire_lag168=np.nan))
    assert not X_train.isna().any().any() and not X_new.isna().any().any()
    assert list(X_train.columns) == list(X_new.columns)
    assert (X_new["nb_lags_manquants"] >= 1).all()


def test_xgb_fits_and_predicts_bounded_values():
    df = fake_dataset()
    cutoff = df["datetime"].max() - pd.Timedelta(days=6)
    start, end = cutoff + pd.Timedelta(hours=25), df["datetime"].max()
    pred = XGBModel(n_estimators=20).fit(df, cutoff).predict(df, cutoff, start, end)
    assert len(pred) == len(ARCS) * len(pd.date_range(start, end, freq="h"))
    assert not pred[TARGETS].isna().any().any()
    assert (pred[TARGETS] >= 0).all().all() and (pred["taux_occupation"] <= 100).all()
