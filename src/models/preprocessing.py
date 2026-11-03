"""Prétraitement des features du dataset (src/data/build_dataset.py) avant un modèle tabulaire.

`Preprocessor` s'apprend sur les seules lignes d'entraînement (datetime <= cutoff), puis s'applique aux lignes à
prédire : aucune statistique ne vient du futur.

Traitement des NaN, du plus informatif au plus grossier :
1. lags hebdomadaires (S-1, S-2, S-3) : remplacés par la moyenne des lags disponibles (`<cible>_lag_mean`) ;
2. lags et niveaux encore manquants (dont lag 52 semaines, niveau 28 j) : profil médian de la cible par
   (arc, jour de semaine, heure) appris sur l'entraînement ;
3. toute autre colonne numérique : médiane de la colonne sur l'entraînement.
Un indicateur `nb_lags_manquants` garde la trace de ce qui a été imputé.
"""

import numpy as np
import pandas as pd

from src.config import ARCS, TARGETS

ID_COLS = ["arc", "datetime", "is_valid", "source_cible", "partie"]
WEEKLY_LAGS = [168, 336, 504]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Colonnes utilisables comme features : ni identifiants, ni cibles, ni météo observée (inconnue à la prédiction)."""
    return [c for c in df.columns
            if c not in ID_COLS + TARGETS and not c.startswith("meteo_obs_")]


class Preprocessor:
    def __init__(self, impute: bool = True):
        self.impute = impute

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        self.columns_ = feature_columns(df)
        valid = df[df["is_valid"]]
        key = [valid["arc"], valid["dow"], valid["hour"]]
        self.profile_ = valid[TARGETS].groupby(key).median()
        self.profile_.index.names = ["arc", "dow", "hour"]
        X = self._encode(df[self.columns_ + ["arc"]])
        self.medians_ = X.median(numeric_only=True)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.columns_].copy()
        lag_cols = [c for c in X.columns if "_lag" in c or "_niveau" in c]
        X["nb_lags_manquants"] = X[lag_cols].isna().sum(axis=1)
        if self.impute:
            profile = self.profile_.reindex(pd.MultiIndex.from_arrays([df["arc"], df["dow"], df["hour"]])).to_numpy()
            for j, col in enumerate(TARGETS):
                for lag in WEEKLY_LAGS:
                    X[f"{col}_lag{lag}"] = X[f"{col}_lag{lag}"].fillna(X[f"{col}_lag_mean"])
                for c in [c for c in lag_cols if c.startswith(col)]:
                    X[c] = X[c].fillna(pd.Series(profile[:, j], index=X.index))
        X = self._encode(X.assign(arc=df["arc"]))
        if self.impute:
            X = X.fillna(self.medians_)
        return X

    @staticmethod
    def _encode(X: pd.DataFrame) -> pd.DataFrame:
        """Booléens en 0/1, arc en indicatrices (ordre fixe : mêmes colonnes en fit et en predict)."""
        X = X.copy()
        for arc in ARCS:
            X[f"arc_{arc}"] = (X["arc"] == arc).astype(np.int8)
        X = X.drop(columns="arc")
        bools = X.select_dtypes(include="bool").columns
        X[bools] = X[bools].astype(np.int8)
        return X.astype(np.float32)
