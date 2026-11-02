"""Stream C : backtest reproduisant le décalage du rendu (cutoff -> 1 jour sauté -> 5 jours prédits).

Usage : python -m src.evaluate --model baseline --n-folds 8
Sorties : outputs/backtest_<model>.csv (RMSE par arc x cible x jour d'horizon) + résumé à l'écran.
"""

import argparse

import numpy as np
import pandas as pd

from src.config import GAP_DAYS, HORIZON_DAYS, OUTPUTS_DIR, TARGETS, TRAFFIC_PATH, TZ
from src.models.registry import MODELS


def window(cutoff: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Fenêtre prédite pour un cutoff donné (dernière heure connue, ex. 05/11 23:00 heure de Paris)."""
    day = cutoff.tz_convert(TZ).tz_localize(None).normalize() + pd.Timedelta(days=GAP_DAYS + 1)
    start = day.tz_localize(TZ).tz_convert("UTC")
    end = (day + pd.Timedelta(days=HORIZON_DAYS, hours=-1)).tz_localize(TZ).tz_convert("UTC")
    return start, end


def default_cutoffs(traffic: pd.DataFrame, n_folds: int) -> list[pd.Timestamp]:
    """Un cutoff par semaine, le jeudi 23:00 (heure de Paris) comme pour le rendu, en remontant depuis la fin."""
    # Dates calendaires naïves puis localisation, pour ne pas glisser d'une heure aux changements d'heure.
    last_day = traffic["datetime"].max().tz_convert(TZ).tz_localize(None).normalize()
    days = pd.date_range(end=last_day - pd.Timedelta(days=GAP_DAYS + HORIZON_DAYS + 1), periods=7 * n_folds + 7, freq="D")
    thursdays = [d for d in days if d.day_of_week == 3][-n_folds:]
    return [(d + pd.Timedelta(hours=23)).tz_localize(TZ).tz_convert("UTC") for d in thursdays]


def backtest(model_name: str, traffic: pd.DataFrame, cutoffs) -> pd.DataFrame:
    rows = []
    for cutoff in cutoffs:
        start, end = window(cutoff)
        model = MODELS[model_name]().fit(traffic, cutoff)
        pred = model.predict(traffic, cutoff, start, end)
        truth = traffic[(traffic["datetime"] >= start) & (traffic["datetime"] <= end) & traffic["is_valid"]]
        merged = truth.merge(pred, on=["arc", "datetime"], suffixes=("", "_pred"))
        merged["horizon_day"] = (merged["datetime"] - start).dt.days + 1
        merged["cutoff"] = cutoff
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def rmse_table(results: pd.DataFrame, by=("arc", "horizon_day")) -> pd.DataFrame:
    out = {}
    for col in TARGETS:
        err2 = (results[col] - results[f"{col}_pred"]) ** 2
        out[col] = err2.groupby([results[b] for b in by]).mean().pow(0.5)
    return pd.DataFrame(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="baseline", choices=list(MODELS))
    parser.add_argument("--n-folds", type=int, default=8)
    args = parser.parse_args()

    traffic = pd.read_parquet(TRAFFIC_PATH)
    cutoffs = default_cutoffs(traffic, args.n_folds)
    results = backtest(args.model, traffic, cutoffs)

    table = rmse_table(results)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    table.to_csv(OUTPUTS_DIR / f"backtest_{args.model}.csv")
    print(f"Cutoffs : {[c.tz_convert(TZ).strftime('%Y-%m-%d %H:%M') for c in cutoffs]}")
    print("\nRMSE par arc :")
    print(rmse_table(results, by=("arc",)).round(2).to_string())
    print("\nRMSE par jour d'horizon :")
    print(rmse_table(results, by=("horizon_day",)).round(2).to_string())
    n_missing = results[[f"{c}_pred" for c in TARGETS]].isna().any(axis=1).sum()
    if n_missing:
        print(f"\nAttention : {n_missing} heures sans prédiction (ignorées dans la RMSE).")
    print(f"\n-> {OUTPUTS_DIR / f'backtest_{args.model}.csv'}")
