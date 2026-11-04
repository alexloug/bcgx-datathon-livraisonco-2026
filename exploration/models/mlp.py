"""Réseau de neurones (MLP, PyTorch) global : les 2 cibles prédites ensemble, les 3 arcs ensemble.

Mêmes entrées que XGBoost (Preprocessor), pour mesurer l'effet du modèle à données identiques. Pas de réglage :
architecture et entraînement standards (exploration/docs/comparaison_modeles.md).

Pipeline : Preprocessor -> standardisation des entrées et des cibles (statistiques de l'entraînement) ->
MLP 3 couches (ReLU, dropout) -> sortie à 2 neurones -> déstandardisation et bornes.
Arrêt précoce sur les 4 dernières semaines avant le cutoff (validation temporelle, jamais après le cutoff).
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.config import TARGETS
from src.models.preprocessing import Preprocessor

# PyTorch et XGBoost/LightGBM embarquent chacun un runtime OpenMP. Sur macOS, PyTorch multithread se bloque si un
# boosting est chargé avant lui, et XGBoost plante si PyTorch est chargé avant lui. Un seul thread PyTorch évite
# les deux (le réseau est petit : quelques secondes par cutoff).
torch.set_num_threads(1)

PARAMS = {
    "hidden": (256, 128, 64),
    "dropout": 0.1,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 512,
    "max_epochs": 60,
    "patience": 6,            # époques sans amélioration de la validation avant arrêt
    "val_days": 28,           # validation = dernières semaines avant le cutoff
    "seed": 0,
}


def build_net(n_in: int, hidden, dropout: float) -> nn.Sequential:
    layers, width = [], n_in
    for h in hidden:
        layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
        width = h
    return nn.Sequential(*layers, nn.Linear(width, len(TARGETS)))


class MLPModel:
    def __init__(self, **params):
        self.params = {**PARAMS, **params}

    def fit(self, dataset: pd.DataFrame, cutoff: pd.Timestamp) -> "MLPModel":
        p = self.params
        torch.manual_seed(p["seed"])
        train = dataset[(dataset["datetime"] <= cutoff) & dataset["is_valid"]]
        self.prep_ = Preprocessor().fit(dataset[dataset["datetime"] <= cutoff])
        X = self.prep_.transform(train).to_numpy(np.float32)
        y = train[TARGETS].to_numpy(np.float32)
        # Une colonne constante à l'entraînement (ex. `jo_2024` avant les JO) garde un écart-type de 1 : sinon une
        # valeur nouvelle à la prédiction serait divisée par ~0.
        std = X.std(0)
        self.x_mean_, self.x_std_ = X.mean(0), np.where(std > 1e-6, std, 1.0).astype(np.float32)
        self.y_mean_, self.y_std_ = y.mean(0), y.std(0)
        X, y = (X - self.x_mean_) / self.x_std_, (y - self.y_mean_) / self.y_std_

        is_val = (train["datetime"] > cutoff - pd.Timedelta(days=p["val_days"])).to_numpy()
        Xt, yt = torch.from_numpy(X[~is_val]), torch.from_numpy(y[~is_val])
        Xv, yv = torch.from_numpy(X[is_val]), torch.from_numpy(y[is_val])

        self.net_ = build_net(X.shape[1], p["hidden"], p["dropout"])
        opt = torch.optim.Adam(self.net_.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        loss_fn = nn.MSELoss()
        best, best_state, wait = np.inf, None, 0
        for _ in range(p["max_epochs"]):
            self.net_.train()
            for idx in torch.randperm(len(Xt)).split(p["batch_size"]):
                opt.zero_grad()
                loss_fn(self.net_(Xt[idx]), yt[idx]).backward()
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                val = loss_fn(self.net_(Xv), yv).item()
            if val < best:
                best, wait = val, 0
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                wait += 1
                if wait >= p["patience"]:
                    break
        self.net_.load_state_dict(best_state)
        self.net_.eval()
        return self

    def predict(self, dataset: pd.DataFrame, cutoff: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        rows = dataset[(dataset["datetime"] >= start) & (dataset["datetime"] <= end)]
        X = (self.prep_.transform(rows).to_numpy(np.float32) - self.x_mean_) / self.x_std_
        with torch.no_grad():
            y = self.net_(torch.from_numpy(X)).numpy() * self.y_std_ + self.y_mean_
        out = rows[["arc", "datetime"]].copy()
        for j, col in enumerate(TARGETS):
            out[col] = np.clip(y[:, j], 0, 100 if col == "taux_occupation" else None)
        return out.reset_index(drop=True)
