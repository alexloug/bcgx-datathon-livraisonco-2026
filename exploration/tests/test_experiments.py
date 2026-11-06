"""Contrôles du protocole : couverture des prédictions et score entre unités distinctes."""
import pandas as pd
import pytest

from src.config import ARCS, TARGETS
from src.evaluate import backtest
from exploration.experiments.run import relative_score, select_features


def test_relative_score_is_invariant_to_target_units():
    baseline = pd.DataFrame({"debit_horaire": [100., 200., 300.], "taux_occupation": [1., 2., 3.]}, index=list(ARCS))
    candidate = baseline * .8
    assert relative_score(candidate, baseline) == pytest.approx(.8)
    candidate["debit_horaire"] *= 1000
    baseline["debit_horaire"] *= 1000
    assert relative_score(candidate, baseline) == pytest.approx(.8)


def test_ablation_preserves_targets_and_identifiers():
    df = pd.DataFrame({"arc": ["Convention"], "datetime": [pd.Timestamp("2026-01-01", tz="UTC")],
                       "debit_horaire": [100.], "taux_occupation": [2.], "hour": [0], "meteo_pluie": [0]})
    selected = select_features(df, {"drop_prefixes": ["meteo_", "debit", "arc"]})
    assert "meteo_pluie" not in selected
    assert set(["arc", "datetime", *TARGETS]) <= set(selected)


def test_backtest_rejects_missing_predictions(monkeypatch):
    class IncompleteModel:
        def fit(self, data, cutoff):
            return self

        def predict(self, data, cutoff, start, end):
            return pd.DataFrame({"arc": [], "datetime": pd.DatetimeIndex([], tz="UTC"),
                                 "debit_horaire": [], "taux_occupation": []})

    monkeypatch.setitem(__import__("src.evaluate", fromlist=["MODELS"]).MODELS, "incomplete", IncompleteModel)
    df = pd.DataFrame({"arc": ["Convention"], "datetime": [pd.Timestamp("2026-10-03", tz="UTC")],
                       "is_valid": [True], "debit_horaire": [100.], "taux_occupation": [2.]})
    with pytest.raises(ValueError, match="prédictions manquantes"):
        backtest("incomplete", df, [pd.Timestamp("2026-10-01 21:00", tz="UTC")])
