"""Essai simple : RMSE de XGBoost contre une référence naïve, sur quelques jeudis (cutoffs).

Usage (depuis la racine, venv actif) :
    python -m exploration.essais_victor.essai_xgb                          # 4 jeudis de septembre 2026 (réglage d'Alex)
    python -m exploration.essais_victor.essai_xgb --mode reguliers         # 1 jeudi toutes les 4 semaines, 2022-2024 (toutes saisons)
    python -m exploration.essais_victor.essai_xgb --mode automne           # tous les jeudis de sept. à nov. 2024 et 2025
    python -m exploration.essais_victor.essai_xgb --cutoffs 2026-09-03 2026-09-10
    python -m exploration.essais_victor.essai_xgb --mode automne --fin-train 2023-12-31   # entraînement gelé fin 2023

Principe, pour chaque jeudi (cutoff) :
    1. train = toutes les lignes valides jusqu'au jeudi 23h (jamais après : pas de fuite du futur)
    2. test  = les 5 jours à prédire, après 1 jour sauté (même découpage que le rendu)
    3. on entraîne un XGBoost par cible, on prédit le test, on calcule le RMSE
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from src.config import TARGETS
from src.evaluate import parse_cutoffs, window
from src.models.preprocessing import feature_columns

DATASET = "data/processed/dataset_modele.parquet"

# Hyperparamètres : ce qu'on va faire varier
PARAMS = dict(n_estimators=600, learning_rate=0.05, max_depth=6, min_child_weight=5,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=42)


def rmse(y, y_hat):
    return np.sqrt(np.mean((y - y_hat) ** 2))


def prepare(df):
    """Booléens -> 0/1 (XGBoost gère seul les valeurs manquantes)."""
    out = df.copy()
    bool_cols = out.select_dtypes("bool").columns
    out[bool_cols] = out[bool_cols].astype(int)
    return out


def jeudis(debut, fin, pas_jours=7):
    """Jeudis (AAAA-MM-JJ) de `debut` à `fin`, tous les `pas_jours` jours."""
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(debut, fin, freq=f"{pas_jours}D")]


MODES = {
    "septembre": jeudis("2026-09-03", "2026-09-24"),
    "reguliers": jeudis("2022-01-06", "2024-12-26", pas_jours=28),
    "automne": jeudis("2024-09-05", "2024-11-28") + jeudis("2025-09-04", "2025-11-27"),
}


def main(cutoffs, depart, params, fin_train=None):
    df = pd.read_parquet(DATASET)
    df = prepare(df)

    cat_cols = ["arc"]
    num_cols = feature_columns(df)                     # toutes les features, sauf identifiants et cibles
    feats = cat_cols + num_cols

    # Préparation : arc en one-hot, le reste tel quel
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("passthrough", num_cols))

    rows = []
    for cutoff in parse_cutoffs(cutoffs):
        start, end = window(cutoff)
        limite = cutoff if fin_train is None else min(cutoff, pd.Timestamp(fin_train, tz="UTC") + pd.Timedelta(hours=23))
        train = df[(df["datetime"] >= pd.Timestamp(depart, tz="UTC")) & (df["datetime"] <= limite) & df["is_valid"]]
        test = df[(df["datetime"] >= start) & (df["datetime"] <= end) & df["is_valid"]]
        if test.empty:                                  # trou de capteur : rien à noter ce jeudi-là
            print(f"  jeudi {cutoff.date()} ignoré : aucune heure valide à tester")
            continue

        # Référence naïve : moyenne du même créneau aux semaines S-1, S-2, S-3 (déjà dans le dataset)
        # (si les 3 lags manquent : médiane de l'entraînement pour le même arc, jour de semaine et heure)
        profil = train.groupby(["arc", "dow", "hour"])[TARGETS].median()

        for cible in TARGETS:
            model = make_pipeline(preprocessor, XGBRegressor(**params))
            model.fit(train[feats], train[cible])
            y_hat = np.clip(model.predict(test[feats]), 0, 100 if cible == "taux_occupation" else None)

            repli = test[["arc", "dow", "hour"]].merge(profil[cible].rename("p"), left_on=["arc", "dow", "hour"],
                                                       right_index=True, how="left")["p"].to_numpy()
            naif = test[f"{cible}_lag_mean"].fillna(pd.Series(repli, index=test.index)).to_numpy()

            for arc, sous in test.assign(xgb=y_hat, naif=naif).groupby("arc"):
                rows.append({"cutoff": cutoff.date(), "annee": cutoff.year, "arc": arc, "cible": cible,
                             "rmse_naif": rmse(sous[cible], sous["naif"]),
                             "rmse_xgb": rmse(sous[cible], sous["xgb"]), "n_heures": len(sous)})

    res = pd.DataFrame(rows)
    if len(res) <= 30:
        print("\nRMSE par jeudi :")
        print(res.round(2).to_string(index=False))

    # Par année
    par_an = res.groupby(["annee", "arc", "cible"])[["rmse_naif", "rmse_xgb"]].mean()
    par_an["gain_%"] = (1 - par_an["rmse_xgb"] / par_an["rmse_naif"]) * 100
    print(f"\nPar année ({res['cutoff'].nunique()} jeudis au total) :")
    print(par_an.round(2).to_string())

    # Moyenne des RMSE sur les jeudis, par arc et par cible
    resume = res.groupby(["arc", "cible"])[["rmse_naif", "rmse_xgb"]].mean()
    resume["gain_%"] = (1 - resume["rmse_xgb"] / resume["rmse_naif"]) * 100
    print("\nMoyenne sur les jeudis (gain > 0 : XGBoost fait mieux que la référence) :")
    print(resume.round(2).to_string())
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(MODES), default="septembre", help="jeu de jeudis prédéfini")
    parser.add_argument("--cutoffs", nargs="+", help="jeudis AAAA-MM-JJ (23h Paris), à la place de --mode")
    parser.add_argument("--fin-train", help="gèle l'entraînement à cette date (AAAA-MM-JJ) au lieu de suivre chaque jeudi")
    parser.add_argument("--depart", default="2021-06-21", help="début de l'historique d'entraînement")
    args = parser.parse_args()
    main(args.cutoffs or MODES[args.mode], args.depart, PARAMS, args.fin_train)
