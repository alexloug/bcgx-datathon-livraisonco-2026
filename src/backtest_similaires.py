"""Compare les variantes « arcs similaires » au XGBoost de référence, sur les mêmes cutoffs et les mêmes heures.

Référence : `xgb` sur dataset_modele.parquet. Variantes : src/models/registry.py (xgb_rel, xgb_groupe, xgb_pool…)
sur dataset_similaires.parquet. Les lignes des 3 arcs du brief sont identiques dans les deux fichiers.

Usage :
    python -m src.backtest_similaires                                   # 8 derniers jeudis + 3 semaines du 11/11
    python -m src.backtest_similaires --models xgb xgb_rel --jeux recents
Sorties : outputs/backtest_similaires_<jeu>.csv (RMSE par modèle, arc, cible) + résumé à l'écran.
"""

import argparse
import time

import pandas as pd

from src.config import DATASET_MODELE_PATH, OUTPUTS_DIR, TARGETS, TZ
from src.data.build_dataset_similaires import OUT as DATASET_SIM
from src.evaluate import backtest, default_cutoffs, parse_cutoffs

JEUX = {
    "recents": None,   # 8 derniers jeudis, comme docs/livraison_dataset.md
    "novembre": ["2023-11-09", "2024-11-07", "2025-11-06"],
    # Un jeudi tous les 2 mois sur 2023-2025 : saisons, vacances, fériés de printemps.
    "etales": ["2023-01-12", "2023-05-11", "2023-09-14", "2024-03-14", "2024-05-02", "2024-12-19",
               "2025-03-13", "2025-05-15", "2025-07-10", "2025-12-18", "2026-02-12", "2026-05-07"],
}
MODELES = ["xgb", "xgb_rel", "xgb_groupe", "xgb_rel_groupe", "xgb_pool", "xgb_pool_w5", "xgb_pool_groupe"]


def rmse(results: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for col in TARGETS:
        err2 = (results[col] - results[f"{col}_pred"]) ** 2
        out[col] = err2.groupby(results["arc"]).mean().pow(0.5)
    return pd.DataFrame(out)


def main(models: list[str], jeux: list[str]) -> None:
    base = pd.read_parquet(DATASET_MODELE_PATH)
    sim = pd.read_parquet(DATASET_SIM)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    for jeu in jeux:
        cutoffs = parse_cutoffs(JEUX[jeu]) if JEUX[jeu] else default_cutoffs(base, 8)
        print(f"\n=== {jeu} : {[c.tz_convert(TZ).strftime('%Y-%m-%d') for c in cutoffs]}", flush=True)
        tables, preds = [], []
        for m in models:
            t0 = time.time()
            res = backtest(m, base if m == "xgb" else sim, cutoffs)
            res = res[res["arc"].isin(base["arc"].unique())]
            tables.append(rmse(res).assign(modele=m))
            preds.append(res[["arc", "datetime", "cutoff", *TARGETS, *[f"{c}_pred" for c in TARGETS]]].assign(modele=m))
            print(f"{m:16s} {time.time() - t0:5.0f} s  " + "  ".join(
                f"{a[:5]} {r.debit_horaire:6.1f}/{r.taux_occupation:5.2f}" for a, r in tables[-1].iterrows()), flush=True)
        table = pd.concat(tables).reset_index().rename(columns={"index": "arc"})
        table.to_csv(OUTPUTS_DIR / f"backtest_similaires_{jeu}.csv", index=False)
        pd.concat(preds).to_parquet(OUTPUTS_DIR / f"backtest_similaires_{jeu}_predictions.parquet", index=False)
        ref = table[table["modele"] == "xgb"].set_index("arc")[TARGETS]
        gain = table.set_index(["modele", "arc"])[TARGETS].div(ref, level="arc").sub(1).mul(100).round(1)
        print(f"\nÉcart de RMSE à xgb, en % (négatif = mieux) — {jeu}")
        print(gain.unstack("arc").to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELES)
    parser.add_argument("--jeux", nargs="+", default=["recents", "novembre"], choices=list(JEUX))
    args = parser.parse_args()
    main(args.models, args.jeux)
