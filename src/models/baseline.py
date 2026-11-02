"""Baseline saisonnière : moyenne des N dernières observations valides au même jour de semaine et à la même heure."""

import pandas as pd

from src.config import ARCS, TARGETS, TZ


class SeasonalBaseline:
    def __init__(self, n_weeks: int = 4):
        self.n_weeks = n_weeks

    def fit(self, traffic: pd.DataFrame, cutoff: pd.Timestamp) -> "SeasonalBaseline":
        past = traffic[(traffic["datetime"] <= cutoff) & traffic["is_valid"]].copy()
        local = past["datetime"].dt.tz_convert(TZ)
        past["dow"], past["hour"] = local.dt.dayofweek, local.dt.hour
        recent = past.sort_values("datetime").groupby(["arc", "dow", "hour"]).tail(self.n_weeks)
        self.profile_ = recent.groupby(["arc", "dow", "hour"])[TARGETS].mean()
        return self

    def predict(self, traffic, cutoff, start, end) -> pd.DataFrame:
        hours = pd.date_range(start, end, freq="h")
        grid = pd.MultiIndex.from_product([list(ARCS), hours], names=["arc", "datetime"]).to_frame(index=False)
        local = grid["datetime"].dt.tz_convert(TZ)
        grid["dow"], grid["hour"] = local.dt.dayofweek, local.dt.hour
        out = grid.join(self.profile_, on=["arc", "dow", "hour"])
        return out[["arc", "datetime", *TARGETS]]
