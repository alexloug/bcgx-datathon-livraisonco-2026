"""Modèle LightGBM global : un régresseur par cible, les 3 arcs ensemble (l'arc est une feature).

Même pipeline que src/models/xgb.py (Preprocessor -> régresseur -> bornes), pour comparer les deux boostings à
données identiques. Paramètres équivalents à ceux de XGBoost, sans réglage (exploration/docs/comparaison_modeles.md).

Note macOS : LightGBM et XGBoost ont besoin d'OpenMP (`brew install libomp`).
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.config import TARGETS
from src.models.preprocessing import Preprocessor

PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "num_leaves": 63,          # ~ profondeur 6 de XGBoost
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "random_state": 0,
    "verbose": -1,
}


class GBMModel:
    def __init__(self, impute: bool = True, **params):
        self.impute = impute
        self.params = {**PARAMS, **params}

    def fit(self, dataset: pd.DataFrame, cutoff: pd.Timestamp) -> "GBMModel":
        train = dataset[(dataset["datetime"] <= cutoff) & dataset["is_valid"]]
        self.prep_ = Preprocessor(self.impute).fit(dataset[dataset["datetime"] <= cutoff])
        X = self.prep_.transform(train)
        self.models_ = {col: LGBMRegressor(**self.params).fit(X, train[col]) for col in TARGETS}
        return self

    def predict(self, dataset: pd.DataFrame, cutoff: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        rows = dataset[(dataset["datetime"] >= start) & (dataset["datetime"] <= end)]
        X = self.prep_.transform(rows)
        out = rows[["arc", "datetime"]].copy()
        for col, model in self.models_.items():
            out[col] = np.clip(model.predict(X), 0, 100 if col == "taux_occupation" else None)
        return out.reset_index(drop=True)
