"""Interface commune à tous les modèles (baseline, GBM, ...), utilisée par evaluate.py et predict.py."""

from typing import Protocol

import pandas as pd


class TrafficModel(Protocol):
    def fit(self, traffic: pd.DataFrame, cutoff: pd.Timestamp) -> "TrafficModel":
        """Entraîne sur les lignes de `traffic` avec datetime <= cutoff (UTC, tz-aware)."""

    def predict(self, traffic: pd.DataFrame, cutoff: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Renvoie arc, datetime (UTC), debit_horaire, taux_occupation pour chaque heure de [start, end].

        `traffic` n'est utilisé que jusqu'au cutoff : aucune valeur postérieure ne doit être lue.
        """
