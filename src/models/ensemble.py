"""Modèle final : moyenne de 3 XGBoost et 3 LightGBM, avec l'occupation nettoyée des dérives du capteur.

Même interface que xgb.py (fit / predict, voir base.py) : `--model ensemble` dans evaluate.py et predict.py.
Par rapport à xgb.py, deux changements, retenus après les essais de exploration/essais_victor/ :
1. pour l'occupation, on retire de l'entraînement les jours où le capteur dérive : l'occupation dépasse de plus de
   60 % l'habituel alors que le débit reste normal (à moins de 20 %) ; plus 3 épisodes repérés à la main ;
2. la prédiction est la moyenne de 6 modèles : XGBoost et LightGBM, 3 graines chacun. Deux boostings voisins font des
   erreurs un peu différentes, et la moyenne dépend moins du hasard d'une seule graine.
Les NaN sont laissés à XGBoost et LightGBM, qui savent les gérer (même score que l'imputation, docs/3_modele.md).
"""

import numpy as np
import pandas as pd

# Préparation
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

# Modèles
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.config import TARGETS, TZ
from src.models.preprocessing import feature_columns

GRAINES = [0, 1, 2]
XGB_PARAMS = dict(n_estimators=600, learning_rate=0.05, max_depth=6, min_child_weight=5,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", n_jobs=-1)
LGBM_PARAMS = dict(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbose=-1)

# Dérives du capteur d'occupation repérées à la main (occupation +60 à +240 % à débit normal)
DERIVES_CONNUES = [("Champs-Elysées", "2023-10-29", "2023-11-15"),
                   ("Champs-Elysées", "2021-11-20", "2021-11-23"),
                   ("Convention", "2022-10-03", "2022-10-06")]


# ---------- Préparation

def colonnes(dataset):
    """Rue + features du dataset, booléens convertis en 0/1."""
    X = dataset[["arc"] + feature_columns(dataset)].copy()
    bools = X.select_dtypes("bool").columns
    X[bools] = X[bools].astype(int)
    return X


def jours_derive_occupation(passe):
    """Jours (rue, date) où le capteur d'occupation dérive, d'après les seules données de `passe`.
    « Habituel » = médiane des mêmes jours de la semaine, 4 semaines avant et 4 semaines après."""
    v = passe[passe["is_valid"]].copy()
    v["date"] = v["datetime"].dt.tz_convert(TZ).dt.date
    jour = v.groupby(["arc", "date"]).agg(occ=("taux_occupation", "mean"), debit=("debit_horaire", "mean"),
                                         n=("debit_horaire", "size"), dow=("dow", "first"))
    jour = jour[jour["n"] >= 18].reset_index()                     # au moins 18 heures mesurées
    jour["date"] = pd.to_datetime(jour["date"])
    morceaux = []
    for _, g in jour.groupby(["arc", "dow"]):
        g = g.set_index("date").sort_index()
        for col in ("occ", "debit"):
            voisins = [g[col].shift(k, freq="7D") for k in (-4, -3, -2, -1, 1, 2, 3, 4)]
            habituel = pd.concat(voisins, axis=1, sort=True).reindex(g.index).median(axis=1)
            g[f"ecart_{col}"] = g[col] / habituel - 1
        morceaux.append(g.reset_index())
    jour = pd.concat(morceaux)
    detectes = jour[(jour["ecart_occ"] > 0.60) & (jour["ecart_debit"].abs() < 0.20)][["arc", "date"]]
    connus = pd.concat([pd.DataFrame({"arc": arc, "date": pd.date_range(debut, fin)})
                        for arc, debut, fin in DERIVES_CONNUES])
    return pd.concat([detectes, connus]).drop_duplicates()


def lignes_entrainement(dataset, cutoff, cible):
    """Heures mesurées jusqu'au `cutoff`. Pour l'occupation, sans les jours où le capteur dérive."""
    train = dataset[(dataset["datetime"] <= cutoff) & dataset["is_valid"]]
    if cible == "taux_occupation":
        derives = jours_derive_occupation(dataset[dataset["datetime"] <= cutoff])
        cle = train["arc"] + "|" + train["datetime"].dt.tz_convert(TZ).dt.strftime("%Y-%m-%d")
        a_retirer = set(derives["arc"] + "|" + derives["date"].dt.strftime("%Y-%m-%d"))
        train = train[~cle.isin(a_retirer)]
    return train


# ---------- Modèles

def creer_modele(nom, graine, num_cols):
    """Pipeline sklearn : rue en indicatrices (one-hot), autres colonnes telles quelles, puis le régresseur.
    Pas de StandardScaler : les arbres n'en ont pas besoin."""
    preprocessor = make_column_transformer((OneHotEncoder(handle_unknown="ignore"), ["arc"]),
                                           ("passthrough", num_cols))
    if nom == "xgb":
        regresseur = XGBRegressor(**XGB_PARAMS, random_state=graine)
    else:
        regresseur = LGBMRegressor(**LGBM_PARAMS, random_state=graine)
    return make_pipeline(preprocessor, regresseur)


class EnsembleModel:
    def __init__(self, modeles=("xgb", "lgbm"), graines=GRAINES):
        self.modeles = modeles
        self.graines = graines

    def fit(self, dataset: pd.DataFrame, cutoff: pd.Timestamp) -> "EnsembleModel":
        num_cols = feature_columns(dataset)
        self.pipelines_ = {}
        for cible in TARGETS:
            train = lignes_entrainement(dataset, cutoff, cible)
            self.pipelines_[cible] = [creer_modele(nom, graine, num_cols).fit(colonnes(train), train[cible])
                                      for nom in self.modeles for graine in self.graines]
        return self

    def predict(self, dataset: pd.DataFrame, cutoff: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        rows = dataset[(dataset["datetime"] >= start) & (dataset["datetime"] <= end)]
        X = colonnes(rows)
        out = rows[["arc", "datetime"]].copy()
        for cible, pipelines in self.pipelines_.items():
            y_hat = np.mean([p.predict(X) for p in pipelines], axis=0)       # moyenne des 6 modèles
            out[cible] = np.clip(y_hat, 0, 100 if cible == "taux_occupation" else None)
        return out.reset_index(drop=True)
