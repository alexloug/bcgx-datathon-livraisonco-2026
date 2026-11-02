"""Stream B : modèle gradient boosting global (un modèle par cible, l'arc en feature).

À implémenter en respectant l'interface de src/models/base.py :
    - fit : make_features(traffic) -> lignes valides avec datetime <= cutoff -> un régresseur par cible
    - predict : ajouter à `traffic` les heures [start, end] avec cibles à NaN, make_features, prédire

Note macOS : LightGBM et XGBoost ont besoin d'OpenMP (`brew install libomp`).
"""

import pandas as pd


class GBMModel:
    def fit(self, traffic: pd.DataFrame, cutoff: pd.Timestamp) -> "GBMModel":
        raise NotImplementedError("Stream B")

    def predict(self, traffic, cutoff, start, end) -> pd.DataFrame:
        raise NotImplementedError("Stream B")
