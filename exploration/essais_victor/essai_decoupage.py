"""Essai : un seul modèle global, ou un modèle par arc / par tranche horaire / par heure ?

Chaque découpage entraîne un XGBoost séparé sur chaque groupe de lignes (ex. une heure donnée), puis prédit
les lignes de test de ce groupe. Référence : le modèle global actuel (les 3 arcs ensemble, l'arc en colonne).

Usage (depuis la racine, venv actif) :
    python -m exploration.essais_victor.essai_decoupage
"""

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from src.config import TARGETS
from exploration.essais_victor.essai_poids import CUTOFFS          # automne 2024-2025 (1 sur 2) + octobre 2026
from exploration.essais_victor.essai_xgb import DATASET, PARAMS, prepare, rmse
from src.evaluate import parse_cutoffs, window
from src.models.preprocessing import feature_columns


def tranche(heure):
    """Tranches horaires (heure de Paris) : nuit, pointe du matin, journée, pointe du soir, soirée."""
    return pd.cut(heure, bins=[-1, 5, 9, 15, 19, 23], labels=["nuit", "matin", "journee", "soir", "soiree"])


DECOUPAGES = {
    "global": lambda d: pd.Series("tout", index=d.index),
    "par_arc": lambda d: d["arc"],
    "par_tranche": lambda d: tranche(d["hour"]).astype(str),
    "par_heure": lambda d: d["hour"],
}


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
        test = df[(df["datetime"] >= start) & (df["datetime"] <= end) & df["is_valid"]]
        if test.empty:
            continue
        for nom, groupe in DECOUPAGES.items():
            g_train, g_test = groupe(train), groupe(test)
            for cible in TARGETS:
                y_hat = pd.Series(np.nan, index=test.index)
                for g in g_test.unique():                    # un modèle par groupe
                    tr, te = train[g_train == g], test[g_test == g]
                    model = make_pipeline(preprocessor, XGBRegressor(**PARAMS)).fit(tr[feats], tr[cible])
                    y_hat[te.index] = model.predict(te[feats])
                for arc in test["arc"].unique():
                    m = test["arc"] == arc
                    rows.append({"cutoff": cutoff.date(), "decoupage": nom, "arc": arc, "cible": cible,
                                 "rmse": rmse(test.loc[m, cible], y_hat[m])})
        print(f"  jeudi {cutoff.date()} fait")

    res = pd.DataFrame(rows)
    table = res.groupby(["cible", "arc", "decoupage"])["rmse"].mean().unstack("decoupage")[list(DECOUPAGES)]
    gain = (1 - table.div(table["global"], axis=0)) * 100
    print(f"\nRMSE moyen sur {res['cutoff'].nunique()} jeudis :")
    print(table.round(2).to_string())
    print("\nGain en % par rapport au modèle global (> 0 = mieux) :")
    print(gain.round(1).to_string())
    piv = res.pivot_table(index=["cutoff", "arc", "cible"], columns="decoupage", values="rmse")
    print("\nPart des (jeudi, rue, cible) où le découpage fait mieux que le global :")
    print(pd.Series({d: (piv[d] < piv["global"]).mean() for d in DECOUPAGES if d != "global"})
          .map("{:.0%}".format).to_string())


if __name__ == "__main__":
    main()
