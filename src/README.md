# src

Code du pipeline, organisé par stream. Toutes les commandes se lancent depuis la racine du repo.

| Fichier | Stream | Rôle |
|---|---|---|
| `config.py` | tous | Arcs (nœuds amont/aval), chemins, horizon (`MIN_LAG_HOURS = 144`) |
| `data/update.py` | A | Met à jour toutes les sources open data → `data/<source>/` (voir `data/README.md`) |
| `data/sources/*.py` | A | Un connecteur par source : trafic, météo, calendrier, événements, chantiers |
| `data/build_dataset.py` | A | Dataset unifié cibles + features → `data/processed/dataset.parquet` (doc : `docs/dataset_entrainement.md`) |
| `data/build_traffic.py` | A | `data/trafic/` (archives + API) → `data/processed/traffic_hourly.parquet` |
| `data/build_external.py` | A | Fériés, confinements, JO → `data/processed/external.parquet` (météo, vacances à ajouter) |
| `features/build.py` | B | Features calendaires et lags ≥ 144 h, refus des lags plus courts |
| `models/base.py` | B | Interface `fit(traffic, cutoff)` / `predict(traffic, cutoff, start, end)` |
| `models/preprocessing.py` | B | Sélection des features, retraitement des NaN, encodage (appris ≤ cutoff) |
| `models/xgb.py` | B | XGBoost, un modèle par cible (`--model xgb`), doc : `docs/modele_xgboost.md` |
| `models/baseline.py` | B | Moyenne des 4 dernières semaines, même jour et même heure |
| `models/gbm.py` | B | Modèle gradient boosting, à implémenter |
| `evaluate.py` | C | Backtest sur `dataset.parquet` : cutoff le jeudi 23:00, 1 jour sauté, 5 jours prédits (`--cutoffs` pour des semaines choisies) |
| `predict.py` | C | CSV de rendu depuis `dataset.parquet`, contrôlé avec le test du brief |

## Pipeline

```bash
python -m src.data.update                     # ajouter --history 2010 2025 pour les archives trafic
python -m src.data.build_dataset              # dataset d'entraînement unifié
python -m src.data.build_traffic
python -m src.data.build_external
python -m src.evaluate --model xgb --n-folds 8
python -m src.predict --model xgb --group <nom_du_groupe>
python -m pytest -q tests
```

Pour ajouter un modèle : une classe qui respecte `models/base.py`, enregistrée dans `models/registry.py`.
Elle est alors utilisable avec `--model <nom>` dans `evaluate.py` et `predict.py`.

## Conventions

- `datetime` en UTC (tz-aware) partout ; conversion en heure de Paris seulement pour le CSV de rendu.
- Une feature pour l'heure t n'utilise que des données à t - 144 h ou avant.
- Les données téléchargées ou générées (`data/**/*.parquet`, `outputs/models/`) ne sont pas versionnées.
- macOS : LightGBM et XGBoost ont besoin d'OpenMP (`brew install libomp`).
