"""Essai : donner plus de poids aux données récentes améliore-t-il XGBoost ?
(Voir aussi src/essai_proxy.py : poids des heures reconstruites des Champs-Elysées.)

Poids d'une heure d'entraînement = 0,5 ** (âge en jours / demi-vie), l'âge étant compté depuis le jeudi (cutoff).
Demi-vie de 365 j : une heure d'il y a 1 an compte moitié moins qu'une heure d'hier ; 2 ans, un quart.

Usage (depuis la racine, venv actif) :
    python -m exploration.essais_victor.essai_poids
    python -m exploration.essais_victor.essai_poids --demi-vies 0 730 365 180 90      # 0 = sans pondération (référence)
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from src.config import TARGETS
from exploration.essais_victor.essai_xgb import DATASET, MODES, PARAMS, prepare, rmse
from src.evaluate import parse_cutoffs, window
from src.models.preprocessing import feature_columns

# Jeudis testés : automne 2024-2025 (1 sur 2) + octobre 2026 (validation d'Alex)
CUTOFFS = MODES["automne"][::2] + ["2026-10-01", "2026-10-08", "2026-10-15", "2026-10-22"]


def main(demi_vies):
    df = prepare(pd.read_parquet(DATASET))
    cat_cols, num_cols = ["arc"], feature_columns(df)
    feats = cat_cols + num_cols
    preprocessor = make_column_transformer((OneHotEncoder(handle_unknown="ignore"), cat_cols),
                                           ("passthrough", num_cols))
    rows = []
    for cutoff in parse_cutoffs(CUTOFFS):
        start, end = window(cutoff)
        train = df[(df["datetime"] <= cutoff) & df["is_valid"]]
        test = df[(df["datetime"] >= start) & (df["datetime"] <= end) & df["is_valid"]]
        if test.empty:
            continue
        age_jours = (cutoff - train["datetime"]).dt.total_seconds().to_numpy() / 86400

        for demi_vie in demi_vies:
            poids = np.ones(len(train)) if demi_vie == 0 else 0.5 ** (age_jours / demi_vie)
            for cible in TARGETS:
                model = make_pipeline(preprocessor, XGBRegressor(**PARAMS))
                model.fit(train[feats], train[cible], xgbregressor__sample_weight=poids)
                y_hat = model.predict(test[feats])
                for arc in test["arc"].unique():
                    m = (test["arc"] == arc).to_numpy()
                    rows.append({"cutoff": cutoff.date(), "demi_vie": demi_vie, "arc": arc, "cible": cible,
                                 "rmse": rmse(test[cible].to_numpy()[m], y_hat[m])})
        print(f"  jeudi {cutoff.date()} fait")

    res = pd.DataFrame(rows)
    table = res.groupby(["cible", "arc", "demi_vie"])["rmse"].mean().unstack("demi_vie")
    gain = (1 - table.div(table[0], axis=0)) * 100
    print(f"\nRMSE moyen sur {res['cutoff'].nunique()} jeudis (colonnes = demi-vie en jours, 0 = sans poids) :")
    print(table.round(2).to_string())
    print("\nGain en % par rapport à « sans poids » (> 0 = mieux) :")
    print(gain.round(1).to_string())
    # Combien de jeudis la pondération gagne-t-elle ? (stabilité)
    piv = res.pivot_table(index=["cutoff", "arc", "cible"], columns="demi_vie", values="rmse")
    print("\nPart des (jeudi, rue, cible) où la pondération fait mieux que sans poids :")
    print(pd.Series({d: (piv[d] < piv[0]).mean() for d in demi_vies if d != 0}).map("{:.0%}".format).to_string())
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demi-vies", nargs="+", type=int, default=[0, 730, 365, 180])
    args = parser.parse_args()
    main(args.demi_vies)
