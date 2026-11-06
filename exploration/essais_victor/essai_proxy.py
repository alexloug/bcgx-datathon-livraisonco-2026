"""Essai : moins de poids sur les heures reconstruites (proxy) des Champs-Elysées améliore-t-il le modèle ?

Depuis la panne du capteur 4264 (19/11/2025), la cible des Champs-Elysées est reconstruite à partir d'un capteur
voisin (`source_cible == "proxy"`), plus quelques trous avant. On teste un poids de 1 (actuel), 0,5, 0,25 et 0
(heures reconstruites ignorées) à l'entraînement.
La note n'est calculée que sur les vraies mesures : sinon on mesurerait l'accord avec le proxy, pas avec la réalité.

Usage : python -m exploration.essais_victor.essai_proxy
"""

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

CUTOFFS = MODES["automne"][::2]      # automne 2024-2025 : le capteur des Champs fonctionne encore
POIDS = [1.0, 0.5, 0.25, 0.0]


def main():
    df = prepare(pd.read_parquet(DATASET))
    cat_cols, num_cols = ["arc"], feature_columns(df)
    feats = cat_cols + num_cols
    preprocessor = make_column_transformer((OneHotEncoder(handle_unknown="ignore"), cat_cols),
                                           ("passthrough", num_cols))
    rows = []
    for cutoff in parse_cutoffs(CUTOFFS):
        start, end = window(cutoff)
        train = df[(df["datetime"] <= cutoff) & df["is_valid"]]
        test = df[(df["datetime"] >= start) & (df["datetime"] <= end) & df["is_valid"]
                  & (df["source_cible"] == "mesure")]                    # vraies mesures seulement
        if test.empty:
            continue
        proxy = (train["source_cible"] == "proxy").to_numpy()
        for w in POIDS:
            poids = np.where(proxy, w, 1.0)
            for cible in TARGETS:
                model = make_pipeline(preprocessor, XGBRegressor(**PARAMS))
                y_hat = model.fit(train[feats], train[cible], xgbregressor__sample_weight=poids).predict(test[feats])
                for arc in test["arc"].unique():
                    m = (test["arc"] == arc).to_numpy()
                    rows.append({"cutoff": cutoff.date(), "poids_proxy": w, "arc": arc, "cible": cible,
                                 "rmse": rmse(test[cible].to_numpy()[m], y_hat[m])})
        print(f"  jeudi {cutoff.date()} fait ({proxy.sum()} heures proxy à l'entraînement)")

    res = pd.DataFrame(rows)
    table = res.groupby(["cible", "arc", "poids_proxy"])["rmse"].mean().unstack()[POIDS]
    print(f"\nRMSE moyen sur {res['cutoff'].nunique()} jeudis, vraies mesures seulement (colonnes = poids du proxy) :")
    print(table.round(2).to_string())
    print("\nGain en % par rapport au poids 1 (> 0 = mieux) :")
    print(((1 - table.div(table[1.0], axis=0)) * 100).round(1).to_string())


if __name__ == "__main__":
    main()
