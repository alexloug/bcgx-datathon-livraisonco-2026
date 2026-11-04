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

## Étape 1 — Lags en heure de Paris

Bug : `lag_features` et `yearly_lag_features` décalaient la grille UTC de N lignes. Après le changement d'heure du
25/10/2026, le lag tombait une heure trop tard (ex. 07/11 08:00 visait le S-3 du 17/10 à 09:00, valeur 322 au
lieu de 237). Correction : `local_lag` cherche « même heure locale, n jours avant » ; heure en double
(automne) = moyenne, heure inexistante (printemps) = NaN. `level_features` reste en décalage réel de 144 h
(moyenne sur 28 jours). `assert_no_leakage` exige des jours entiers et `lag - 1 >= 144`.

Vérification sur la fenêtre du rendu (360 h) : 0 écart entre `lag168`, `lag336`, `lag504` et la valeur réelle
à l'heure locale attendue. Les 3 nouveaux tests (traversée du 25/10, heures en double/manquantes, lags
fractionnaires) échouent sur l'ancien code.

RMSE du débit de XGBoost, avant -> après la correction (un seul run, pas de graine différente) :

| Semaines | Champs-Elysées | Convention | Saint-Pères |
|---|---|---|---|
| 8 jeudis, fenêtre entière | 182,8 -> 182,6 | 65,7 -> 65,9 | 101,9 -> 102,0 |
| dont semaine du 24/10 seule | 137,9 -> 139,7 | 65,1 -> 56,8 | 126,2 -> 130,7 |
| 3 semaines 11/11, fenêtre entière | 200,1 -> 196,1 | 68,2 -> 68,0 | 88,8 -> 87,9 |
| dont le 11/11 seul | 269,1 -> 276,9 | 63,1 -> 63,6 | 60,0 -> 57,8 |

Lecture : sur les 8 jeudis, aucun effet (une seule semaine touchée par le changement d'heure). Sur les 3 semaines
du 11/11, léger mieux sur la fenêtre entière, mais mélangé sur le 11/11 seul (Champs-Elysées se dégrade de 3 %,
Saint-Pères s'améliore de 4 %) : ces écarts sont dans le bruit d'un échantillon de 150 heures. La correction
est retenue parce qu'elle est exacte (le lag pointe la bonne heure) et non pour un gain mesuré.

## Étape 2 — Backtest enrichi (`src/evaluate.py`)

Ajouts, sans toucher à `--cutoffs` ni à la RMSE par `source_cible` : `--folds regular|holiday|all` (défaut
`regular`, comme avant) ; la baseline tourne automatiquement sur les mêmes cutoffs, et seules les heures où le
modèle et la baseline ont une prédiction sont gardées ; gain en % par type de fold et par arc ; une ligne par
run x fold x arc x cible ajoutée à `outputs/backtest_runs.csv`.

Semaines `holiday` : 34 jeudis depuis 2022 dont la fenêtre samedi -> mercredi contient un férié (calculé depuis
`data/calendrier/jours_feries`). Un férié tombant un jeudi ou un vendredi n'est donc jamais dans une fenêtre
(ex. le 11/11/2022, un vendredi). `--folds all` = 8 jeudis `regular` + 34 `holiday`.

XGBoost contre baseline, après correction des lags (`python -m src.evaluate --model xgb --folds all`) :

| Fold | Arc | Débit XGB / base (gain) | Occupation XGB / base (gain) | Heures |
|---|---|---|---|---|
| regular | Champs-Elysées* | 182,6 / 192,8 (+5 %) | 9,43 / 10,88 (+13 %) | 902 |
| regular | Convention | 65,9 / 82,0 (+20 %) | 2,53 / 2,88 (+12 %) | 935 |
| regular | Saint-Pères | 102,0 / 159,8 (+36 %) | 2,91 / 3,72 (+22 %) | 933 |
| holiday | Champs-Elysées | 177,7 / 224,8 (+21 %) | 9,39 / 10,25 (+8 %) | 3 695 |
| holiday | Convention | 78,2 / 121,6 (+36 %) | 2,94 / 3,32 (+11 %) | 2 336 |
| holiday | Saint-Pères | 72,1 / 110,6 (+35 %) | 1,85 / 2,41 (+23 %) | 3 060 |

\* Cible partiellement reconstruite par proxy. Le détail mesure / proxy est affiché par `evaluate` (RMSE par origine
de la cible) : sur toutes les semaines, le débit des Champs est à 229,6 en mesure contre 189,5 en proxy pour la baseline.

XGBoost bat la baseline sur les deux familles, pour les 3 arcs et les 2 cibles. Par fold, il fait mieux sur 68 des
82 couples (fold x arc) pour le débit des semaines fériées.

Limites : (1) les folds de 2022 sont appris sur 6 à 12 mois d'historique seulement (le dataset commence en
06/2021), donc plus difficiles que ceux de 2025 ; (2) Convention a moins d'heures (trous de capteur) ; (3) le
« jour d'horizon 6 » affiché est un artefact : une semaine contenant le retour à l'heure d'hiver compte 121 h
réelles (défaut préexistant du calcul d'`horizon_day`, non corrigé).
