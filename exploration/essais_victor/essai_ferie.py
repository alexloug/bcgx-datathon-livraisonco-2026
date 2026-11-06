"""Essai : traiter à part les jours fériés (dont le 11 novembre) ou le jour de la semaine.

Variantes, évaluées sur les 3 semaines contenant un 11 novembre (2023, 2024, 2025) :
    global        : modèle actuel
    par_jour      : un modèle par jour de la semaine (7 modèles)
    modele_ferie  : global pour les jours normaux + un modèle entraîné sur les seuls jours fériés pour les fériés
    poids_ferie   : global, avec un poids x10 sur les heures fériées à l'entraînement
Erreur mesurée sur le 11/11 seul, et sur les autres jours de la fenêtre (pour vérifier qu'on n'abîme rien).

Usage : python -m exploration.essais_victor.essai_ferie
"""

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from src.config import TARGETS, TZ
from exploration.essais_victor.essai_xgb import DATASET, PARAMS, prepare, rmse
from src.evaluate import parse_cutoffs, window
from src.models.preprocessing import feature_columns

CUTOFFS = ["2023-11-09", "2024-11-07", "2025-11-06"]
POIDS_FERIE = 10


def main():
    df = prepare(pd.read_parquet(DATASET))
    cat_cols, num_cols = ["arc"], feature_columns(df)
    feats = cat_cols + num_cols
    preprocessor = make_column_transformer((OneHotEncoder(handle_unknown="ignore"), cat_cols),
                                           ("passthrough", num_cols))

    def fit(tr, cible, poids=None):
        return make_pipeline(preprocessor, XGBRegressor(**PARAMS)).fit(
            tr[feats], tr[cible], xgbregressor__sample_weight=poids)

    rows = []
    for cutoff in parse_cutoffs(CUTOFFS):
        start, end = window(cutoff)
        train = df[(df["datetime"] <= cutoff) & df["is_valid"]]
        test = df[(df["datetime"] >= start) & (df["datetime"] <= end) & df["is_valid"]]
        jour = test["datetime"].dt.tz_convert(TZ)
        est_11nov = ((jour.dt.month == 11) & (jour.dt.day == 11)).to_numpy()
        ferie_tr, ferie_te = train["ferie"] == 1, test["ferie"] == 1
        print(f"  jeudi {cutoff.date()} : {int(ferie_tr.sum() / 3 / 24)} jours fériés environ à l'entraînement")

        for cible in TARGETS:
            preds = {}
            glob = fit(train, cible)
            preds["global"] = glob.predict(test[feats])

            p = np.full(len(test), np.nan)
            for d in test["dow"].unique():
                m = (test["dow"] == d).to_numpy()
                p[m] = fit(train[train["dow"] == d], cible).predict(test[feats][m])
            preds["par_jour"] = p

            p = preds["global"].copy()
            if ferie_te.any():
                p[ferie_te.to_numpy()] = fit(train[ferie_tr], cible).predict(test[feats][ferie_te])
            preds["modele_ferie"] = p

            preds["poids_ferie"] = fit(train, cible, np.where(ferie_tr, POIDS_FERIE, 1.0)).predict(test[feats])

            for nom, y_hat in preds.items():
                for arc in test["arc"].unique():
                    for zone, mz in [("11/11", est_11nov), ("autres jours", ~est_11nov)]:
                        m = (test["arc"] == arc).to_numpy() & mz
                        if m.any():
                            rows.append({"cutoff": cutoff.date(), "variante": nom, "arc": arc, "cible": cible,
                                         "zone": zone, "rmse": rmse(test[cible].to_numpy()[m], y_hat[m])})

    res = pd.DataFrame(rows)
    ordre = ["global", "par_jour", "modele_ferie", "poids_ferie"]
    for zone in ["11/11", "autres jours"]:
        t = res[res["zone"] == zone].groupby(["cible", "arc", "variante"])["rmse"].mean().unstack()[ordre]
        g = (1 - t.div(t["global"], axis=0)) * 100
        print(f"\n=== {zone} : RMSE moyen sur les 3 années")
        print(t.round(2).to_string())
        print(f"Gain en % par rapport au global (> 0 = mieux) :")
        print(g.round(1).to_string())


if __name__ == "__main__":
    main()
