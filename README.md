# LivraisonCo — Datathon BCG X 2026

Prévision horaire du débit et de l'occupation sur Champs-Elysées, Convention et Saint-Pères,
du 7 au 11 novembre 2026 (360 lignes).

## Organisation

| Dossier | Contenu |
|---|---|
| `data/<source>/` | Sources téléchargées, non versionnées |
| `data/processed/` | Dataset complet et export final `dataset_modele.parquet` |
| `src/data/`, `src/features/` | Stream 1 : collecte, nettoyage et construction des features |
| `src/models/` | Stream 2 : modèles et prétraitement partagé |
| `src/experiments/`, `configs/` | Stream 2 : protocole et configurations reproductibles |
| `src/evaluate.py`, `src/predict.py` | Stream 3 : backtests et CSV de rendu |
| `docs/` | Méthode, livraison, résultats et limites |
| `notebooks/` | Analyses de qualité et construction du dataset |
| `outputs/experiments/` | Résultats locaux de chaque campagne (non versionnés) |
| `archive/` | Ancienne approche CSV, conservée hors du pipeline actif |
| `tests/` | Contrôles du dataset, modèles et protocole |

## Préparer les données

Depuis la racine, avec l'environnement virtuel activé :

```bash
pip install -r requirements.txt
python -m src.data.update --history 2020 2025 &&
python -m src.data.build_dataset &&
python -m src.data.export_dataset
```

Si le dataset final est déjà disponible, passer directement aux expériences.
Description : [livraison du dataset](docs/livraison_dataset.md), [features](docs/features.md).

## Stream 2 : comparer et régler les modèles

Sur le macOS local sans Homebrew, OpenMP a été installé dans `.venv/lib/`.
Avant d'entraîner ou de lancer tous les tests, exécuter :

```bash
export DYLD_LIBRARY_PATH="$PWD/.venv/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

```bash
python -m src.experiments.run --config configs/stream2.json
python -m pytest -q tests
```

Le lanceur sélectionne une configuration sur les semaines de réglage, puis évalue seulement la baseline,
XGBoost de référence et cette configuration sur une validation temporelle distincte et les semaines du 11 novembre.
Voir le [protocole stream 2](docs/stream2_modelisation.md) avant d'interpréter les résultats.
Les paramètres sélectionnés sont conservés dans `selected.json` ; ils ne remplacent pas automatiquement le modèle de rendu.

## Rendu

```bash
python -m src.predict --model xgb --group NOM_DU_GROUPE
```

Cette commande utilise les paramètres XGBoost par défaut et le dataset complet. Le rendu définitif doit attendre
l'actualisation du 6 novembre, les contrôles de qualité et la décision finale de l'équipe.

Index des documents : [docs/README.md](docs/README.md).
