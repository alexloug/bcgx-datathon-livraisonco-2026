"""Deux essais sur l'entraînement, avec 3 graines moyennées par variante (le bruit d'une graine est de 3-5 %).

    python -m exploration.essais_victor.essai_nettoyage perte       # fonction de perte : quadratique, Huber, absolue
    python -m exploration.essais_victor.essai_nettoyage anomalies   # retirer de l'entraînement les jours anormaux (travaux, déviations)
    python -m exploration.essais_victor.essai_nettoyage debut       # date de début de l'entraînement : 2021, 2023, 18 ou 12 derniers mois

Perte : la note finale reste le RMSE (erreur quadratique), mais on peut minimiser autre chose à l'entraînement.
    quadratique : actuel, très sensible aux grosses erreurs (un jour de déviation pèse énormément)
    huber       : quadratique pour les petites erreurs, linéaire au-delà de `delta` (moins sensible aux aberrants)
    absolue     : erreur absolue partout (vise la médiane plutôt que la moyenne)
Anomalies : détection dans src/anomalies.py, recalculée à chaque jeudi avec les seules données passées.
Les jours anormaux restent dans le TEST : au rendu, on ne pourra pas les retirer.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from exploration.essais_victor.anomalies import jours_anormaux, masque
from src.config import TARGETS
from exploration.essais_victor.essai_poids import CUTOFFS          # automne 2024-2025 (1 sur 2) + octobre 2026
from exploration.essais_victor.essai_xgb import DATASET, PARAMS, prepare, rmse
from src.evaluate import parse_cutoffs, window
from src.models.preprocessing import feature_columns

GRAINES = [0, 1, 2]
DELTA = {"debit_horaire": {"huber_30": 30, "huber_100": 100},        # en véhicules/heure
         "taux_occupation": {"huber_30": 1, "huber_100": 3}}         # en points d'occupation


def variantes_perte(cible, train):
    """{nom: (params XGBoost, lignes d'entraînement gardées)}"""
    tout = np.ones(len(train), bool)
    median = float(train[cible].median())           # point de départ, utile pour Huber et l'erreur absolue
    v = {"quadratique": ({}, tout)}
    for nom, delta in DELTA[cible].items():
        v[nom.replace("_30", "_petit").replace("_100", "_grand")] = (
            {"objective": "reg:pseudohubererror", "huber_slope": delta, "base_score": median}, tout)
    v["absolue"] = ({"objective": "reg:absoluteerror", "base_score": median}, tout)
    return v


def variantes_anomalies(cible, train, anormaux):
    return {nom: ({}, ~masque(train, a)) for nom, a in anormaux.items()}


def variantes_debut(train, cutoff):
    """Même modèle, historique d'entraînement plus ou moins long (coupé, pas pondéré)."""
    t = train["datetime"]
    return {"depuis_2021": ({}, np.ones(len(train), bool)),
            "depuis_2023": ({}, (t >= pd.Timestamp("2023-01-01", tz="UTC")).to_numpy()),
            "18_derniers_mois": ({}, (t > cutoff - pd.DateOffset(months=18)).to_numpy()),
            "12_derniers_mois": ({}, (t > cutoff - pd.DateOffset(months=12)).to_numpy())}


def main(essai):
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
        if essai == "anomalies":
            passe = df[df["datetime"] <= cutoff]                  # détection sans regarder le futur
            a20, a30 = jours_anormaux(passe, 0.20), jours_anormaux(passe, 0.30)
            connus = a30[a30["cause"] != "détecté"]
            anormaux = {"aucun_retrait": a30.iloc[:0], "episodes_connus": connus,
                        "retrait_30%": a30, "retrait_20%": a20}
            print(f"  jeudi {cutoff.date()} : {len(connus)} / {len(a30)} / {len(a20)} jours retirés "
                  "(connus / 30 % / 20 %)")

        for cible in TARGETS:
            variantes = (variantes_perte(cible, train) if essai == "perte"
                         else variantes_debut(train, cutoff) if essai == "debut"
                         else variantes_anomalies(cible, train, anormaux))
            for nom, (params, garde) in variantes.items():
                tr = train[garde]
                preds = [make_pipeline(preprocessor, XGBRegressor(**{**PARAMS, **params, "random_state": g}))
                         .fit(tr[feats], tr[cible]).predict(test[feats]) for g in GRAINES]
                y_hat = np.mean(preds, axis=0)
                for arc in test["arc"].unique():
                    m = (test["arc"] == arc).to_numpy()
                    rows.append({"cutoff": cutoff.date(), "variante": nom, "arc": arc, "cible": cible,
                                 "rmse": rmse(test[cible].to_numpy()[m], y_hat[m])})
        if essai != "anomalies":
            print(f"  jeudi {cutoff.date()} fait")

    res = pd.DataFrame(rows)
    ordre = list(dict.fromkeys(res["variante"]))
    ref = ordre[0]
    table = res.groupby(["cible", "arc", "variante"])["rmse"].mean().unstack()[ordre]
    print(f"\nRMSE moyen sur {res['cutoff'].nunique()} jeudis (3 graines moyennées) :")
    print(table.round(2).to_string())
    print(f"\nGain en % par rapport à « {ref} » (> 0 = mieux) :")
    print(((1 - table.div(table[ref], axis=0)) * 100).round(1).to_string())
    piv = res.pivot_table(index=["cutoff", "arc", "cible"], columns="variante", values="rmse")
    print(f"\nPart des (jeudi, rue, cible) où la variante fait mieux que « {ref} » :")
    print(pd.Series({v: (piv[v] < piv[ref]).mean() for v in ordre if v != ref}).map("{:.0%}".format).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "perte")
