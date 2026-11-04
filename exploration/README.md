# Exploration : ce qu'on a testé et pourquoi on ne l'a pas retenu

Tout ce dossier est hors du chemin du rendu (`src/`). Il garde la trace des pistes écartées, avec leurs chiffres.
Le code déplacé ici n'est plus branché sur `src/models/registry.py` (qui ne connaît que `baseline` et `xgb`) :
il sert de référence, pas de commande à relancer telle quelle. Pour rejouer un modèle, l'ajouter au registre
(`"gbm": _factory("exploration.models.gbm", "GBMModel")`) et installer `lightgbm` ou `torch`.
Ses tests : `python -m pytest exploration/tests`.

## Contenu

| Dossier | Quoi |
|---|---|
| `models/` | LightGBM (`gbm.py`) et réseau dense PyTorch (`mlp.py`) |
| `experiments/`, `configs/` | Protocole de réglage / validation du stream 2 et ses configurations |
| `notebooks/` | EDA, qualité des données, construction du dataset |
| `data/` | Anciens constructeurs `build_traffic.py` et `build_external.py`, remplacés par `src/data/build_dataset.py` |
| `docs/` | Les 13 documents d'origine, avant leur fusion en 4 dans `docs/` à la racine |

## Ce qu'on a écarté

### LightGBM et réseau de neurones : XGBoost suffit
Backtest du 04/11, mêmes données et mêmes features pour tous, RMSE de 6 h à 20 h, débit (véh/h) / occupation (points) :

| Jeu de cutoffs | Arc | Baseline | XGBoost | LightGBM | MLP |
|---|---|---|---|---|---|
| Jeudis réguliers 2022-2024 | Champs-Elysées | 213 / 11,1 | 186 / 9,5 | 184 / 9,3 | 199 / 9,6 |
| | Convention | 106 / 5,5 | 87 / 4,8 | 88 / 4,7 | 97 / 4,8 |
| 2e semaine après reprise | Champs-Elysées | 201 / 10,5 | 161 / 8,1 | 161 / 8,2 | 175 / 9,5 |
| | Convention | 125 / 4,9 | 70 / 4,3 | 67 / 4,4 | 77 / 4,7 |

- LightGBM ≈ XGBoost (écart de 0 à 3 %, il gagne 53 % des cutoffs × arcs sur le débit : du bruit) et 3 fois plus lent (9,5 s contre 3 s par cutoff).
- Le MLP perd 5 à 15 % de RMSE débit sur les jeudis réguliers et les reprises, et varie de ±5 % selon la graine.
- Moyenner les trois modèles gagne 1 à 2 % de RMSE (183 / 9,2 contre 186 / 9,5 sur les Champs-Elysées) : pas assez pour justifier trois modèles.
- Aucun modèle n'était réglé : on mesure l'effet du modèle, pas son optimisation.

### Réglages XGBoost, météo prévue, événements, pondération du proxy
Campagne `stream2_initial` : réglage sur septembre 2026, validation sur octobre. Score = moyenne des ratios RMSE / baseline (1 = baseline, plus petit = mieux).

| Candidat | Réglage (sept.) | Validation (oct.) | Semaines du 11/11 |
|---|---|---|---|
| baseline | 1,000 | 1,000 | 1,000 |
| xgb_reference | 0,771 | 0,868 | 0,840 |
| xgb_proxy_half (proxy pondéré 0,5) | 0,767 | 0,871 | 0,828 |
| xgb_depth4 | 0,772 | | |
| xgb_regularized | 0,771 | | |
| xgb_without_weather | 0,772 | | |
| xgb_without_events | 0,785 | | |

- Profondeur 4 et régularisation : pas mieux que la référence. Retirer la météo prévue ne change rien (0,772 contre 0,771) ; retirer les événements dégrade (0,785).
- Le proxy pondéré à 0,5 gagne 0,46 % au réglage, perd 0,25 % en validation : pas de gain stable, non adopté.
- Limite : une recherche ciblée sur peu de semaines, pas une optimisation exhaustive.

### Chantiers
Emprises de chantier de la Ville, 4 features (`chantier_*`), testées sur 39 jeudis de 2022 à 2024 (RMSE débit avec / sans) : Champs-Elysées 193 / 193, Convention 91 / 89, Saint-Pères 79 / 78. Les chantiers gagnent un fold sur deux.
Une emprise ne dit pas si une voie est fermée ni où part la déviation, et la moitié des jours ont une emprise à moins de 150 m.
Les colonnes restent générées dans `dataset.parquet` (si `data/chantiers/` existe), mais le modèle les ignore. Aucune emprise de voirie à moins de 150 m des 3 arcs sur la fenêtre du 07-11/11.

### « Que faire à Paris »
Agenda culturel de la Ville, téléchargé mais jamais intégré aux features : les événements retenus sont la liste manuelle `data/evenements/evenements_manuels.csv`. Source disponible avec `python -m src.data.update --all`.

### Anciennes chaînes de données
`data/build_traffic.py` et `data/build_external.py` ont précédé `src/data/build_dataset.py`. Seule la liste de périodes `EVENTS` a été reprise (dans `src/config.py`).
