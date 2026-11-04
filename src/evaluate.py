"""Stream C : backtest reproduisant le décalage du rendu (cutoff -> 1 jour sauté -> 5 jours prédits).

Entrée : data/processed/dataset.parquet (python -m src.data.build_dataset). Toutes ses features sont décalées
d'au moins 144 h : pour un cutoff c, les lignes prédites (c + 25 h -> c + 144 h) n'utilisent que des valeurs <= c.

Usage :
    python -m src.evaluate --model xgb --n-folds 8
    python -m src.evaluate --model xgb --cutoffs 2024-11-07 2025-11-06     # jeudis, 23:00 heure de Paris
    python -m src.evaluate --model xgb --dataset data/processed/dataset_modele.parquet
    python -m src.evaluate --model xgb --folds all    # 8 jeudis récents (regular) + semaines avec férié (holiday)
Le modèle est toujours comparé à la baseline sur exactement les mêmes heures (gain en %, par type de fold).
Sorties : outputs/backtest_<model>.csv (RMSE par arc x cible x jour d'horizon) + résumé à l'écran ;
outputs/backtest_runs.csv : une ligne par run x fold x arc x cible, ajoutée à chaque run (historique).
"""

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import DATASET_PATH, GAP_DAYS, HORIZON_DAYS, OUTPUTS_DIR, TARGETS, TZ
from src.data.sources import calendrier
from src.models.registry import MODELS

BASELINE = "baseline"
RUNS_PATH = OUTPUTS_DIR / "backtest_runs.csv"
HOLIDAY_SINCE = "2022-01-01"


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


def holiday_cutoffs(traffic: pd.DataFrame, since: str = HOLIDAY_SINCE) -> list[pd.Timestamp]:
    """Jeudis depuis `since` dont la fenêtre prédite (samedi -> mercredi) contient un jour férié et est connue."""
    feries = pd.to_datetime(calendrier.load("jours_feries")["date"])
    last = traffic.loc[traffic["is_valid"], "datetime"].max()
    cutoffs = []
    for thursday in pd.date_range(since, last.tz_convert(TZ).tz_localize(None), freq="W-THU"):
        cutoff = (thursday + pd.Timedelta(hours=23)).tz_localize(TZ).tz_convert("UTC")
        start, end = window(cutoff)
        days = pd.date_range(start.tz_convert(TZ).tz_localize(None), periods=HORIZON_DAYS, freq="D")
        if end <= last and days.isin(feries).any():
            cutoffs.append(cutoff)
    return cutoffs


def select_folds(traffic: pd.DataFrame, folds: str, n_folds: int, manual: list[str] | None) -> dict:
    """{cutoff: type de fold}. Un jeudi à la fois récent et férié reste « regular »."""
    if manual:
        return {c: "manual" for c in parse_cutoffs(manual)}
    out = {}
    if folds in ("all", "holiday"):
        out.update({c: "holiday" for c in holiday_cutoffs(traffic)})
    if folds in ("all", "regular"):
        out.update({c: "regular" for c in default_cutoffs(traffic, n_folds)})
    return dict(sorted(out.items()))


def backtest(model_name: str, traffic: pd.DataFrame, cutoffs, model_params=None) -> pd.DataFrame:
    rows = []
    for cutoff in cutoffs:
        start, end = window(cutoff)
        model = MODELS[model_name](**(model_params or {})).fit(traffic, cutoff)
        pred = model.predict(traffic, cutoff, start, end)
        truth = traffic[(traffic["datetime"] >= start) & (traffic["datetime"] <= end) & traffic["is_valid"]]
        if pred.duplicated(["arc", "datetime"]).any():
            raise ValueError("Prédictions dupliquées pour un arc et une heure")
        merged = truth.merge(pred, on=["arc", "datetime"], suffixes=("", "_pred"), how="left", validate="one_to_one")
        if merged.empty or not np.isfinite(merged[[f"{c}_pred" for c in TARGETS]].to_numpy()).all():
            raise ValueError(f"Backtest vide ou prédictions manquantes/non finies au cutoff {cutoff}")
        merged["horizon_day"] = (merged["datetime"] - start).dt.days + 1
        merged["cutoff"] = cutoff
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def parse_cutoffs(days: list[str]) -> list[pd.Timestamp]:
    return [(pd.Timestamp(d) + pd.Timedelta(hours=23)).tz_localize(TZ).tz_convert("UTC") for d in days]


def compare(model_name: str, traffic: pd.DataFrame, fold_types: dict) -> pd.DataFrame:
    """Modèle et baseline sur les mêmes folds et les mêmes heures : colonnes <cible>_pred et <cible>_base."""
    cutoffs = list(fold_types)
    results = backtest(model_name, traffic, cutoffs)
    preds = [f"{c}_pred" for c in TARGETS]
    bases = [f"{c}_base" for c in TARGETS]
    base = results if model_name == BASELINE else backtest(BASELINE, traffic, cutoffs)
    base = base[["cutoff", "arc", "datetime", *preds]].rename(columns=dict(zip(preds, bases)))
    results = results.merge(base, on=["cutoff", "arc", "datetime"], how="left")
    both = results[preds + bases].notna().all(axis=1)
    if (~both).any():
        print(f"Attention : {(~both).sum()} heures sans prédiction (modèle ou baseline), retirées des deux RMSE.")
    results = results[both].copy()
    results["fold_type"] = results["cutoff"].map(fold_types)
    return results


def rmse_table(results: pd.DataFrame, by=("arc", "horizon_day"), suffix="pred") -> pd.DataFrame:
    out = {}
    for col in TARGETS:
        err2 = (results[col] - results[f"{col}_{suffix}"]) ** 2
        out[col] = err2.groupby([results[b] for b in by]).mean().pow(0.5)
    return pd.DataFrame(out)


def gain_table(results: pd.DataFrame, by=("fold_type", "arc")) -> pd.DataFrame:
    """Une ligne par groupe x cible : RMSE, RMSE de la baseline, gain en % (positif = mieux que la baseline)."""
    out = pd.concat([rmse_table(results, by).stack().rename("rmse"),
                     rmse_table(results, by, "base").stack().rename("rmse_baseline")], axis=1)
    out.index = out.index.set_names([*by, "target"])
    out["gain_pct"] = 100 * (1 - out["rmse"] / out["rmse_baseline"])
    out["n_hours"] = results.groupby(list(by)).size().reindex(out.index.droplevel("target")).to_numpy()
    return out


def log_run(model_name: str, results: pd.DataFrame) -> None:
    """Ajoute à outputs/backtest_runs.csv une ligne par fold x arc x cible."""
    per_fold = gain_table(results, by=("fold_type", "cutoff", "arc")).reset_index()
    per_fold["cutoff"] = per_fold["cutoff"].dt.tz_convert(TZ).dt.strftime("%Y-%m-%d")
    per_fold.insert(0, "model", model_name)
    per_fold.insert(0, "run_at", datetime.now().strftime("%Y-%m-%d %H:%M"))
    OUTPUTS_DIR.mkdir(exist_ok=True)
    per_fold.round(3).to_csv(RUNS_PATH, mode="a", header=not RUNS_PATH.exists(), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="baseline", choices=list(MODELS))
    parser.add_argument("--n-folds", type=int, default=8, help="nombre de jeudis consécutifs, en remontant depuis la fin")
    parser.add_argument("--cutoffs", nargs="+", help="jeudis explicites (AAAA-MM-JJ), à la place de --n-folds")
    parser.add_argument("--dataset", default=DATASET_PATH, help="dataset d'entrée (ex. le dataset livré au modèle)")
    parser.add_argument("--folds", default="regular", choices=["regular", "holiday", "all"],
                        help="regular : --n-folds derniers jeudis ; holiday : jeudis depuis 2022 dont sam->mer contient un férié")
    parser.add_argument("--no-log", action="store_true", help="ne pas écrire dans outputs/backtest_runs.csv")
    args = parser.parse_args()

    traffic = pd.read_parquet(args.dataset)
    fold_types = select_folds(traffic, args.folds, args.n_folds, args.cutoffs)
    cutoffs = list(fold_types)
    results = compare(args.model, traffic, fold_types)

    table = rmse_table(results)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    table.to_csv(OUTPUTS_DIR / f"backtest_{args.model}.csv")
    print(f"Cutoffs : {[c.tz_convert(TZ).strftime('%Y-%m-%d %H:%M') for c in cutoffs]}")
    print("\nRMSE par arc (24 heures, comme la note du brief) :")
    print(rmse_table(results, by=("arc",)).round(2).to_string())
    if "source_cible" in results:
        print("\nRMSE par origine de la cible (proxy = Champs-Elysées reconstruit) :")
        print(rmse_table(results, by=("arc", "source_cible")).round(2).to_string())
    print(f"\n{args.model} contre la baseline, par type de fold et par arc (gain > 0 : mieux que la baseline) :")
    print(gain_table(results).round({"rmse": 2, "rmse_baseline": 2, "gain_pct": 1}).to_string())
    print("\nRMSE par jour d'horizon :")
    print(rmse_table(results, by=("horizon_day",)).round(2).to_string())
    n_missing = results[[f"{c}_pred" for c in TARGETS]].isna().any(axis=1).sum()
    if n_missing:
        print(f"\nAttention : {n_missing} heures sans prédiction (ignorées dans la RMSE).")
    if not args.no_log:
        log_run(args.model, results)
        print(f"-> {RUNS_PATH} (historique)")
    print(f"\n-> {OUTPUTS_DIR / f'backtest_{args.model}.csv'}")
