"""Stream B : construction des features, sans fuite par rapport au cutoff.

Règle : une feature pour l'heure t ne peut utiliser que des valeurs observées à t - MIN_LAG_HOURS
ou avant (144 h pour la version notée). `assert_no_leakage` le vérifie sur les lags.
"""

import pandas as pd

from src.config import MIN_LAG_HOURS, TARGETS, TZ

LAGS_HOURS = [168, 336, 504]   # même heure à S-1, S-2, S-3


def calendar_features(datetimes: pd.Series) -> pd.DataFrame:
    local = datetimes.dt.tz_convert(TZ)
    return pd.DataFrame({
        "hour": local.dt.hour,
        "dow": local.dt.dayofweek,
        "month": local.dt.month,
        "is_weekend": local.dt.dayofweek >= 5,
    }, index=datetimes.index)


def lag_features(traffic: pd.DataFrame, lags=LAGS_HOURS) -> pd.DataFrame:
    """Lags par arc sur la grille horaire continue ; les valeurs invalides sont masquées."""
    assert_no_leakage(lags)
    values = traffic[TARGETS].where(traffic["is_valid"])
    out = pd.DataFrame(index=traffic.index)
    for lag in lags:
        shifted = values.groupby(traffic["arc"]).shift(lag)
        for col in TARGETS:
            out[f"{col}_lag{lag}"] = shifted[col]
    for col in TARGETS:
        out[f"{col}_lag_mean"] = out[[f"{col}_lag{lag}" for lag in lags]].mean(axis=1)
    return out


def assert_no_leakage(lags) -> None:
    too_short = [lag for lag in lags if lag < MIN_LAG_HOURS]
    if too_short:
        raise ValueError(f"Lags {too_short} < {MIN_LAG_HOURS} h : indisponibles au moment de la prédiction")


def make_features(traffic: pd.DataFrame, external: pd.DataFrame | None = None) -> pd.DataFrame:
    """`traffic` doit contenir aussi les heures à prédire (cibles à NaN) pour que les lags s'y propagent."""
    traffic = traffic.sort_values(["arc", "datetime"]).reset_index(drop=True)
    feats = pd.concat([traffic[["arc", "datetime"]], calendar_features(traffic["datetime"]), lag_features(traffic)], axis=1)
    if external is not None:
        feats = feats.merge(external, on="datetime", how="left")
    return feats
