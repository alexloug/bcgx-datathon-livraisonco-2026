# Premier modèle : XGBoost

Code : `src/models/preprocessing.py` (retraitement) et `src/models/xgb.py` (modèle), enregistré sous le nom `xgb`.
Données : le dataset de `docs/dataset_entrainement.md`.

```bash
python -m src.data.build_dataset
python -m src.evaluate --model xgb --n-folds 8                                    # 8 derniers jeudis
python -m src.evaluate --model xgb --cutoffs 2023-11-09 2024-11-07 2025-11-06     # semaines du 11 novembre
python -m src.predict --model xgb --group <nom_du_groupe>                          # CSV de rendu
```

## Pipeline

1. **Sélection des features** : tout le dataset sauf identifiants, cibles et météo observée (`meteo_obs_*`, inconnue
   au moment de prédire).
2. **Retraitement des NaN** (`Preprocessor`, appris sur les seules lignes ≤ cutoff) :
   - lag S-1, S-2 ou S-3 manquant → moyenne des lags disponibles ;
   - lag ou niveau encore manquant → profil médian de la cible par (arc, jour de semaine, heure) ;
   - toute autre valeur manquante → médiane de la colonne ;
   - `nb_lags_manquants` garde la trace du nombre de lags imputés.
3. **Encodage** : booléens en 0/1, arc en indicatrices.
4. **Modèle** : un `XGBRegressor` par cible (débit, occupation), les 3 arcs ensemble. Heures valides ≤ cutoff, y
   compris les heures Champs-Elysées reconstruites par proxy (poids réglable avec `proxy_weight`).
   600 arbres, profondeur 6, taux d'apprentissage 0,05, sous-échantillonnage 0,8 (`PARAMS`).
5. **Post-traitement** : prédictions bornées à ≥ 0, occupation ≤ 100 %.

## Backtest (données au 03/11/2026)

Même décalage que le rendu : cutoff le jeudi 23:00, 1 jour sauté, 5 jours prédits. RMSE :

**8 derniers jeudis (03/09 → 22/10/2026)**

| Arc | Débit baseline | Débit XGBoost | Occupation baseline | Occupation XGBoost |
|---|---|---|---|---|
| Champs-Elysées (cible reconstruite) | 193 | 193 | 10,9 | **9,4** |
| Convention | 82 | **65** | 2,9 | **2,5** |
| Saint-Pères | 160 | **102** | 3,7 | **2,9** |

**Semaines contenant le 11 novembre (2023, 2024, 2025), débit sur le seul 11/11**

| Arc | Baseline | XGBoost |
|---|---|---|
| Champs-Elysées | 344 | **259** |
| Convention | 154 | **67** |
| Saint-Pères | 128 | **62** |

La baseline (moyenne des 4 dernières semaines) ignore le jour férié. XGBoost l'apprend grâce à `ferie` et au lag 52 semaines.

**Variantes testées** (8 jeudis + 3 semaines de novembre) : imputation et NaN gérés nativement par XGBoost donnent
des RMSE équivalentes (écarts < 2 %). Même constat pour un poids de 0,5 sur les heures reconstruites par proxy.
On garde l'imputation par défaut, qui rend le pipeline indépendant du modèle.

**Features les plus utilisées** (débit) : moyenne des lags S-1/S-2/S-3 (68 %), lag S-1, lag 52 semaines, férié,
vacances, JO 2024.

## Limites et pistes

- **Champs-Elysées** : l'erreur est dominée par la cible elle-même, reconstruite par proxy (septembre 2026 : trou
  de 4274, seul 4281 disponible). On ne peut pas mieux évaluer cet arc sans mesure réelle.
- Hyperparamètres non optimisés : prochaine étape, une recherche sur les backtests (attention au surapprentissage
  sur les 8 semaines).
- Pistes : prédire l'écart au niveau récent (`cible / niveau28j`) pour mieux suivre les ruptures ; prédire le couple
  de façon cohérente (occupation à partir du débit prédit) ; ajouter des cutoffs sur d'autres périodes de fériés.
