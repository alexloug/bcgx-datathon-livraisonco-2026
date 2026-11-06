# Sélection des features du XGBoost

06/11/2026. Question : le modèle `xgb` reçoit 48 colonnes (les 44 features de `dataset_modele.parquet`, plus
`nb_lags_manquants` et 3 indicatrices d'arc). Certaines ajoutent-elles du bruit ou un biais qu'on éliminerait avec
une short-list ?

**Réponse : non.** Aucune short-list ne fait mieux que le modèle complet hors des cutoffs qui ont servi à la choisir.
On garde les 44 features pour le rendu.

## Méthode

`src/selection_features.py`. Ablation par groupe : on réentraîne `xgb` (mêmes hyperparamètres) sans un groupe de
features et on compare la RMSE à celle du modèle complet, cutoff par cutoff (jeudi 23:00, 1 jour sauté, 5 jours
prédits, 24 h). `XGBModel(exclude=..., own_lags_only=...)` retire les colonnes après le prétraitement.

- **Sélection** sur 15 cutoffs : semaines du 11/11 (2023-2025) et 12 jeudis étalés de 2023 à 2026.
- **Validation** sur les 8 derniers jeudis (03/09 → 22/10/2026), jamais regardés pendant la sélection.
- **Score** = moyenne des écarts relatifs sur les 6 RMSE (2 cibles × 3 arcs), chaque RMSE pesant autant.

```bash
python -m src.selection_features ablation                       # ~25 min
python -m src.selection_features valider vacances meteo_jour    # une short-list contre le modèle complet
```

## Ablation : retirer un groupe à la fois

Écart de RMSE au modèle complet, en %. Positif = le groupe est utile. « Pire » = la plus forte dégradation des
6 RMSE.

| Groupe retiré | Colonnes | Sélection : score | Sélection : pire | Validation : score |
|---|---|---|---|---|
| `ferie` | 1 | **+4,0** | +12,7 (débit Convention) | +0,3 |
| `vacances_detail` | type, jours avant/depuis reprise, jours avant vacances, part du niveau 28 j en vacances | **+2,4** | +5,7 | +0,1 |
| `niveau28j` | 2 | **+1,9** | +3,0 | −0,6 |
| `lag52s` | 2 | +0,6 | +4,4 | −0,2 |
| `evenements` | 3 | +0,5 | +4,8 (débit Champs) | **+1,4** (+9,3 débit Champs) |
| `lag_mean` | 2 | +0,4 | +1,7 | 0,0 |
| `nb_lags_manquants` | 1 | +0,2 | +4,0 | +0,2 |
| `saison` | month, heure_ete | −0,1 | +1,3 | +0,9 |
| `meteo_horaire` | 7 | −0,1 | +2,2 | −0,1 |
| `autour_ferie` | veille, lendemain, pont | −0,2 | +0,7 | +0,5 |
| `autre_cible` | historique de l'autre cible | −0,2 | +2,5 | +0,8 |
| `regimes` | JO 2024, post-JO, 30 km/h | −0,3 | +1,1 | +0,4 |
| `lags_S1_S3` | 6 | −0,6 | +2,8 | +1,0 |
| `meteo_jour` | pluie, neige jour/veille, tmin | −0,9 | +0,4 | +1,0 |
| `vacances` | 1 | −1,7 | −0,4 | +0,3 |

Lecture :
- **3 groupes portent le modèle** au-delà des lags hebdomadaires : le férié, le détail des vacances scolaires et le
  niveau 28 jours. Les événements comptent pour les Champs-Elysées (fermetures).
- **Tous les autres groupes sont à ±2 %**, de l'ordre de la différence entre deux réentraînements sur des données
  voisines. Ceux qui semblent « nuire » en sélection (score négatif) ne nuisent plus en validation : aucun écart
  n'est stable d'un jeu de cutoffs à l'autre.

## Short-lists testées

Short-list = modèle complet moins les groupes dont l'absence améliore le score de sélection sans dégrader aucune
RMSE de plus de 1,5 %.

| Short-list | Groupes retirés | Sélection | Validation | Semaines du 11/11 seules |
|---|---|---|---|---|
| large | `vacances`, `meteo_jour`, `regimes`, `autour_ferie`, `saison` | +0,8 | +1,3 | **+6,1** (+23 % débit Convention) |
| prudente | `vacances`, `meteo_jour` | −0,8 | +0,4 | −1,4 (mais +8 % débit Convention) |

Retirer les groupes ensemble ne fait pas mieux que de les retirer un par un : leurs petits effets ne s'additionnent
pas. La short-list large perd nettement sur les semaines du 11/11, celles qui ressemblent le plus au rendu :
`autour_ferie` et `regimes` y servent. La prudente reste dans le bruit.

## Décision

- **Rendu** : `xgb` avec ses 44 features, sans changement. On ne détecte ni bruit ni biais à retirer : avec
  `colsample_bytree = 0,8` et 108 000 lignes, XGBoost ignore de lui-même les colonnes peu utiles.
- Pour la présentation : les features qui comptent sont l'historique du trafic (moyenne des lags S-1 à S-3 : 64 % du
  gain pour le débit), le férié, les vacances scolaires, le niveau 28 jours et, pour les Champs, les fermetures
  événementielles.
- Les marges de progrès sont ailleurs que dans le choix des features : jours anormaux (travaux, déviations), cible
  des Champs-Elysées reconstruite, réglage des hyperparamètres.

Sorties : `outputs/selection_features_ablation.parquet` et `outputs/selection_features_valider.parquet`
(prédictions heure par heure de chaque variante).
