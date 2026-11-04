"""Stream B : construction des features, sans fuite par rapport au cutoff.

Règle : une feature pour l'heure t ne peut utiliser que des valeurs observées à t - MIN_LAG_HOURS
ou avant (144 h pour la version notée). `assert_no_leakage` le vérifie sur les lags.
"""

from __future__ import annotations

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


def local_lag(traffic: pd.DataFrame, lag_hours: int) -> pd.DataFrame:
    """Cibles à « même heure locale, lag_hours/24 jours avant », par arc (valeurs invalides masquées).

    Le trafic suit l'heure de Paris. Un décalage de N lignes sur la grille UTC tombe sur la mauvaise heure
    locale après un changement d'heure (ex. 25/10/2026 : tout lag de 7 à 14 jours pour la semaine du 07/11).
    Heure en double (passage à l'heure d'hiver) : moyenne des deux mesures ; heure inexistante (passage à
    l'heure d'été) : NaN.
    """
    values = traffic[TARGETS].where(traffic["is_valid"])
    wall = traffic["datetime"].dt.tz_convert(TZ).dt.tz_localize(None)   # heure murale, sans fuseau
    by_wall = values.groupby([traffic["arc"], wall.rename("wall")]).mean()
    key = pd.MultiIndex.from_arrays([traffic["arc"], wall - pd.Timedelta(hours=lag_hours)], names=["arc", "wall"])
    shifted = by_wall.reindex(key)
    shifted.index = traffic.index
    return shifted


def lag_features(traffic: pd.DataFrame, lags=LAGS_HOURS) -> pd.DataFrame:
    """Lags hebdomadaires en heure de Paris : `lag168` = même heure locale, 7 jours plus tôt."""
    assert_no_leakage(lags)
    out = pd.DataFrame(index=traffic.index)
    for lag in lags:
        shifted = local_lag(traffic, lag)
        for col in TARGETS:
            out[f"{col}_lag{lag}"] = shifted[col]
    for col in TARGETS:
        out[f"{col}_lag_mean"] = out[[f"{col}_lag{lag}" for lag in lags]].mean(axis=1)
    return out


def level_features(traffic: pd.DataFrame, offset: int = MIN_LAG_HOURS, window_days: int = 28) -> pd.DataFrame:
    """Niveau récent de l'arc : moyenne des `window_days` jours qui s'arrêtent `offset` heures avant t.

    Suit les ruptures de niveau (JO 2024, baisse de Convention) que les lags hebdomadaires ne voient qu'à moitié.
    """
    assert_no_leakage([offset], local=False)   # décalage réel : une moyenne sur 28 jours ne souffre pas d'1 h de biais
    values = traffic[TARGETS].where(traffic["is_valid"])
    out = pd.DataFrame(index=traffic.index)
    for col in TARGETS:
        shifted = values[col].groupby(traffic["arc"]).shift(offset)
        out[f"{col}_niveau{window_days}j"] = shifted.groupby(traffic["arc"]).transform(
            lambda s: s.rolling(24 * window_days, min_periods=24 * 7).mean())
    return out


def yearly_lag_features(traffic: pd.DataFrame, lag: int = 364 * 24) -> pd.DataFrame:
    """Même heure, même jour de semaine, 52 semaines plus tôt (utile pour le 11 novembre, les vacances…)."""
    assert_no_leakage([lag])
    return local_lag(traffic, lag).add_suffix("_lag52s")


def assert_no_leakage(lags, local: bool = True) -> None:
    """Lags en heure locale : jours entiers, et assez longs même quand un changement d'heure raccourcit l'écart d'1 h.

    `local=False` : décalage réel en heures (niveau 28 j), il suffit que lag >= MIN_LAG_HOURS.
    """
    if not local:
        if any(lag < MIN_LAG_HOURS for lag in lags):
            raise ValueError(f"Lags {lags} < {MIN_LAG_HOURS} h : indisponibles au moment de la prédiction")
        return
    not_days = [lag for lag in lags if lag % 24]
    if not_days:
        raise ValueError(f"Lags {not_days} : en heure locale, un lag doit être un nombre entier de jours")
    too_short = [lag for lag in lags if lag - 1 < MIN_LAG_HOURS]
    if too_short:
        raise ValueError(f"Lags {too_short} : moins de {MIN_LAG_HOURS} h après un changement d'heure, "
                         "donc indisponibles au moment de la prédiction")


def make_features(traffic: pd.DataFrame, external: pd.DataFrame | None = None) -> pd.DataFrame:
    """`traffic` doit contenir aussi les heures à prédire (cibles à NaN) pour que les lags s'y propagent."""
    traffic = traffic.sort_values(["arc", "datetime"]).reset_index(drop=True)
    feats = pd.concat([traffic[["arc", "datetime"]], calendar_features(traffic["datetime"]), lag_features(traffic)], axis=1)
    if external is not None:
        feats = feats.merge(external, on="datetime", how="left")
    return feats
