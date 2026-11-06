"""XGBoost avec les arcs similaires (dataset de src/data/build_dataset_similaires.py).

Trois options, combinables, pour mesurer chacune séparément en backtest :
- `relative` : on prédit cible / niveau 28 j de l'arc, puis on remultiplie. Les lags sont aussi ajoutés rapportés
  au niveau. Indispensable pour mélanger des arcs de niveaux différents (Bd Brune ~400 véh/h, Quai des Tuileries
  ~1 300) ; testé seul pour séparer son effet de celui de l'augmentation.
- `groupe` : garde les features `sim_*` (comportement passé des arcs similaires, décalé d'au moins 144 h).
- `pool` : ajoute les lignes des arcs similaires à l'entraînement ; `brief_weight` = poids des lignes des 3 arcs du
  brief (1 = autant qu'une ligne d'arc similaire).
On ne prédit que les 3 arcs du brief : la RMSE du backtest porte sur eux seuls.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.config import TARGETS
from src.models.preprocessing import Preprocessor
from src.models.xgb import PARAMS

EPS = {"debit_horaire": 1.0, "taux_occupation": 0.1}
REL_LAGS = ["lag168", "lag336", "lag504", "lag_mean", "lag52s"]


class XGBSimModel:
    def __init__(self, relative: bool = True, groupe: bool = False, pool: bool = False, brief_weight: float = 1.0,
                 **params):
        self.relative, self.groupe, self.pool, self.brief_weight = relative, groupe, pool, brief_weight
        self.params = {**PARAMS, **params}

    def _rows(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if "role" in dataset and not self.pool:
            dataset = dataset[dataset["role"] == "brief"]
        if not self.groupe:
            dataset = dataset.drop(columns=[c for c in dataset if c.startswith("sim_")])
        return dataset

    def _X(self, rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        X = self.prep_.transform(rows)
        dens = {}
        if self.relative:
            for col in TARGETS:
                dens[col] = X[f"{col}_niveau28j"].clip(lower=EPS[col]).to_numpy()
                for lag in REL_LAGS:
                    X[f"{col}_{lag}_rel"] = X[f"{col}_{lag}"] / dens[col]
        return X, dens

    def fit(self, dataset: pd.DataFrame, cutoff: pd.Timestamp) -> "XGBSimModel":
        d = self._rows(dataset)
        d = d[d["datetime"] <= cutoff]
        self.prep_ = Preprocessor().fit(d.drop(columns=["role", "groupe"], errors="ignore"))
        train = d[d["is_valid"]]
        X, dens = self._X(train)
        weight = np.ones(len(train))
        if "role" in train:
            weight = np.where(train["role"] == "brief", self.brief_weight, 1.0)
        self.models_ = {}
        for col in TARGETS:
            y = train[col].to_numpy() / dens[col] if self.relative else train[col]
            self.models_[col] = XGBRegressor(**self.params).fit(X, y, sample_weight=weight)
        return self

    def predict(self, dataset: pd.DataFrame, cutoff: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        d = self._rows(dataset)
        if "role" in d:
            d = d[d["role"] == "brief"]
        rows = d[(d["datetime"] >= start) & (d["datetime"] <= end)]
        X, dens = self._X(rows)
        out = rows[["arc", "datetime"]].copy()
        for col, model in self.models_.items():
            p = model.predict(X) * (dens[col] if self.relative else 1)
            out[col] = np.clip(p, 0, 100 if col == "taux_occupation" else None)
        return out.reset_index(drop=True)

    def feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame({col: pd.Series(m.feature_importances_, index=m.get_booster().feature_names)
                             for col, m in self.models_.items()}).sort_values(TARGETS[0], ascending=False)
