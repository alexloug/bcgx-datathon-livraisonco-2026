# Expériences modèle (branche `leopold/modele`)

Journal des expériences, toutes évaluées sur les mêmes semaines de test et comparées à la baseline
(moyenne des 4 dernières semaines, même jour et même heure). Historique brut : `outputs/backtest_runs.csv`.

## Étape 0 — Mise en place et chiffres de référence

Environnement : Windows, Python 3.11, xgboost 3.2.0, lightgbm 4.7.0, pyarrow 18.1 (pyarrow ≥ 19 exige NumPy 2).
`unzip` n'est pas dans le PATH de PowerShell : ajouter `C:\Program Files\Git\usr\bin` à la session avant
`python -m src.data.update --history 2020 2025`. L'archive 2021 (Deflate64) est bien chargée : 446 505 lignes,
du 01/01 au 31/12/2021. La source `chantiers` échoue au téléchargement, sans effet : le dataset ne l'utilise pas.

Données au 03/11/2026 (dernière heure observée : 03/11 23:00 UTC) ; dataset de 141 843 lignes, comme dans
`docs/dataset_entrainement.md`.

### Référence A : 8 derniers jeudis (03/09 → 22/10/2026), RMSE sur toute la fenêtre

| Arc | Débit baseline | Débit XGBoost | Occ. baseline | Occ. XGBoost |
|---|---|---|---|---|
| Champs-Elysées (proxy) | 192,8 | 182,8 | 10,88 | 9,39 |
| Convention | 81,9 | 65,7 | 2,88 | 2,53 |
| Saint-Pères | 159,8 | 101,9 | 3,72 | 2,91 |

### Référence B : semaines du 11/11 (cutoffs 2023-11-09, 2024-11-07, 2025-11-06)

Sur toute la fenêtre sam → mer :

| Arc | Débit baseline | Débit XGBoost | Occ. baseline | Occ. XGBoost |
|---|---|---|---|---|
| Champs-Elysées | 216,1 | 200,1 | 15,55 | 13,06 |
| Convention | 91,9 | 68,2 | 2,13 | 1,71 |
| Saint-Pères | 108,2 | 88,8 | 2,21 | 1,89 |

Sur le seul 11/11 (150 heures), débit :

| Arc | Baseline | XGBoost |
|---|---|---|
| Champs-Elysées | 344,3 | 269,1 |
| Convention | 153,6 | 63,1 |
| Saint-Pères | 127,5 | 60,0 |

### Écarts avec `docs/modele_xgboost.md`

- Baseline : identique partout.
- XGBoost : identique sauf le débit des Champs-Elysées sur 8 jeudis (182,8 ici contre 193 dans la doc) et
  le 11/11 (269/63/60 ici contre 259/67/62). Le run est déterministe (deux exécutions identiques), donc
  l'écart vient de la version de xgboost ou d'un léger changement de données, pas du hasard. Les conclusions
  (XGBoost bat la baseline partout) ne changent pas. **Ce sont ces chiffres-ci la référence pour la suite.**
