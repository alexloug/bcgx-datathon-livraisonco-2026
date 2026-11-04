# Tests

```bash
python -m pytest -q tests
```

- `test_dataset.py` : nettoyage, proxys et features externes.
- `test_models.py` : prétraitement et apprentissage des trois modèles sur données synthétiques.
- `test_pipeline.py` : horizon, absence de lags courts et contrôles du rendu/export.
- `test_experiments.py` : ablations, score sans mélange d'unités et refus des prédictions absentes.

Les tests de modèles nécessitent XGBoost, LightGBM, PyTorch et OpenMP.
Voir `docs/stream2_modelisation.md` pour l'environnement macOS local.
