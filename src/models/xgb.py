"""Modèle XGBoost global : un régresseur par cible, les 3 arcs ensemble (l'arc est une feature).

Entrée : le dataset de src/data/build_dataset.py (cibles + features déjà décalées d'au moins 144 h), passé à
`fit`/`predict` à la place de `traffic` (même interface que src/models/base.py).
Pipeline : Preprocessor (NaN, encodage) -> XGBRegressor. Prédictions bornées à [0, +inf) et occupation <= 100 %.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.config import TARGETS
from src.models.preprocessing import Preprocessor

PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 0,
}


class XGBModel:
    def __init__(self, impute: bool = True, proxy_weight: float = 1.0, **params):
        """`proxy_weight` : poids des heures Champs-Elysées reconstruites par proxy (1 = comme une mesure)."""
        self.impute = impute
        self.proxy_weight = proxy_weight
        self.params = {**PARAMS, **params}

    def fit(self, dataset: pd.DataFrame, cutoff: pd.Timestamp) -> "XGBModel":
        train = dataset[(dataset["datetime"] <= cutoff) & dataset["is_valid"]]
        self.prep_ = Preprocessor(self.impute).fit(dataset[dataset["datetime"] <= cutoff])
        X = self.prep_.transform(train)
        weight = np.where(train["source_cible"] == "proxy", self.proxy_weight, 1.0)
        self.models_ = {col: XGBRegressor(**self.params).fit(X, train[col], sample_weight=weight) for col in TARGETS}
        return self

    def predict(self, dataset: pd.DataFrame, cutoff: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        rows = dataset[(dataset["datetime"] >= start) & (dataset["datetime"] <= end)]
        X = self.prep_.transform(rows)
        out = rows[["arc", "datetime"]].copy()
        for col, model in self.models_.items():
            out[col] = np.clip(model.predict(X), 0, 100 if col == "taux_occupation" else None)
        return out.reset_index(drop=True)

    def feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame({col: pd.Series(m.feature_importances_, index=m.get_booster().feature_names)
                             for col, m in self.models_.items()}).sort_values(TARGETS[0], ascending=False)
