"""Tests de LightGBM et du MLP, écartés du rendu (voir exploration/README.md). Lancer : python -m pytest exploration/tests."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("module, cls, params", [
    ("exploration.models.gbm", "GBMModel", "{'n_estimators': 20, 'n_jobs': 2}"),
    ("exploration.models.mlp", "MLPModel", "{'max_epochs': 3}"),
])
def test_model_fits_and_predicts_bounded_values(module, cls, params):
    # Les runtimes OpenMP de PyTorch et des boostings doivent rester dans des processus distincts.
    code = f"""
import runpy
import pandas as pd
from src.config import ARCS, TARGETS
from {module} import {cls}
df = runpy.run_path('tests/test_models.py')['fake_dataset']()
cutoff = df['datetime'].max() - pd.Timedelta(days=6)
start, end = cutoff + pd.Timedelta(hours=25), df['datetime'].max()
pred = {cls}(**{params}).fit(df, cutoff).predict(df, cutoff, start, end)
assert len(pred) == len(ARCS) * len(pd.date_range(start, end, freq='h'))
assert not pred[TARGETS].isna().any().any()
assert (pred[TARGETS] >= 0).all().all() and (pred['taux_occupation'] <= 100).all()
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
