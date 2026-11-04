"""Stream C : backtest reproduisant le décalage du rendu (cutoff -> 1 jour sauté -> 5 jours prédits).

Entrée : data/processed/dataset.parquet (python -m src.data.build_dataset). Toutes ses features sont décalées
d'au moins 144 h : pour un cutoff c, les lignes prédites (c + 25 h -> c + 144 h) n'utilisent que des valeurs <= c.

Usage :
    python -m src.evaluate --model xgb --n-folds 8
    python -m src.evaluate --model xgb --cutoffs 2024-11-07 2025-11-06     # jeudis, 23:00 heure de Paris
    python -m src.evaluate --model xgb --dataset data/processed/dataset_modele.parquet
Sorties : outputs/backtest_<model>.csv (RMSE sur les 24 heures, par arc x cible x jour d'horizon) + résumé à l'écran.
"""

import argparse

import numpy as np
import pandas as pd

from src.config import DATASET_PATH, GAP_DAYS, HORIZON_DAYS, OUTPUTS_DIR, TARGETS, TZ
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
    last_day = traffic.loc[traffic["is_valid"], "datetime"].max().tz_convert(TZ).tz_localize(None).normalize()
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


def parse_cutoffs(days: list[str]) -> list[pd.Timestamp]:
    return [(pd.Timestamp(d) + pd.Timedelta(hours=23)).tz_localize(TZ).tz_convert("UTC") for d in days]


def rmse_table(results: pd.DataFrame, by=("arc", "horizon_day")) -> pd.DataFrame:
    out = {}
    for col in TARGETS:
        err2 = (results[col] - results[f"{col}_pred"]) ** 2
        out[col] = err2.groupby([results[b] for b in by]).mean().pow(0.5)
    return pd.DataFrame(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="baseline", choices=list(MODELS))
    parser.add_argument("--n-folds", type=int, default=8, help="nombre de jeudis consécutifs, en remontant depuis la fin")
    parser.add_argument("--cutoffs", nargs="+", help="jeudis explicites (AAAA-MM-JJ), à la place de --n-folds")
    parser.add_argument("--dataset", default=DATASET_PATH, help="dataset d'entrée (ex. le dataset livré au modèle)")
    args = parser.parse_args()

    traffic = pd.read_parquet(args.dataset)
    cutoffs = parse_cutoffs(args.cutoffs) if args.cutoffs else default_cutoffs(traffic, args.n_folds)
    results = backtest(args.model, traffic, cutoffs)

    table = rmse_table(results)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    table.to_csv(OUTPUTS_DIR / f"backtest_{args.model}.csv")
    print(f"Cutoffs : {[c.tz_convert(TZ).strftime('%Y-%m-%d %H:%M') for c in cutoffs]}")
    print("\nRMSE par arc (24 heures, comme la note du brief) :")
    print(rmse_table(results, by=("arc",)).round(2).to_string())
    if "source_cible" in results:
        print("\nRMSE par origine de la cible (proxy = Champs-Elysées reconstruit) :")
        print(rmse_table(results, by=("arc", "source_cible")).round(2).to_string())
    print("\nRMSE par jour d'horizon :")
    print(rmse_table(results, by=("horizon_day",)).round(2).to_string())
    n_missing = results[[f"{c}_pred" for c in TARGETS]].isna().any(axis=1).sum()
    if n_missing:
        print(f"\nAttention : {n_missing} heures sans prédiction (ignorées dans la RMSE).")
    print(f"\n-> {OUTPUTS_DIR / f'backtest_{args.model}.csv'}")
