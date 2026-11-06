# Comparaison des modèles : XGBoost, LightGBM, réseau de neurones

Backtest du 04/11/2026. **XGBoost reste le bon choix.** LightGBM fait jeu égal, 3 fois plus lentement. Le réseau de
neurones (MLP) est 5 à 15 % moins bon sur le débit et proche sur l'occupation, sauf sur les Champs-Elysées
en semaine de reprise (9,5 contre 8,1). Aucun modèle n'a été
réglé : on mesure l'effet du modèle, pas son optimisation.

## Protocole

- **Mêmes données pour tous** : le dataset de `docs/features.md`, sans les colonnes chantiers. Les 48 mêmes features
  prétraitées (`Preprocessor`), apprises sur les seules lignes antérieures au cutoff.
- **Même découpage que le rendu** : cutoff le jeudi 23:00, 1 jour sauté, 5 jours prédits.
- **RMSE de 6 h à 20 h**, heures de livraison supposées.
- **4 jeux de cutoffs** :
  - 39 jeudis réguliers de 2022 à 2024 (un toutes les 4 semaines) ;
  - les 8 derniers jeudis (septembre-octobre 2026) ;
  - 24 deuxièmes semaines après une reprise de vacances, comme la fenêtre du 07-11/11 ;
  - les 3 semaines du 11/11 (2023-2025).

| Modèle | Code | Principe | Temps par cutoff |
| --- | --- | --- | --- |
| Baseline | `baseline` | moyenne des 4 dernières semaines, même jour et même heure | < 0,1 s |
| XGBoost | `xgb` | un modèle par cible, 600 arbres, profondeur 6 | 3 s |
| LightGBM | `gbm` | mêmes réglages traduits (600 arbres, 63 feuilles) | 9,5 s |
| MLP (PyTorch) | `mlp` | 3 couches cachées (256, 128, 64), dropout 0,1, Adam, les 2 cibles ensemble, arrêt précoce sur les 4 dernières semaines avant le cutoff | 2 s |

## Résultats

RMSE débit (véh/h) / occupation (points), 6 h-20 h. En gras, le meilleur modèle. Sur les 8 derniers jeudis, la cible
des Champs-Elysées est reconstruite par proxy.

| Jeu de cutoffs | Arc | Baseline | XGBoost | LightGBM | MLP |
| --- | --- | --- | --- | --- | --- |
| Réguliers 2022-2024 | Champs-Elysées | 213 / 11,1 | 186 / 9,5 | **184 / 9,3** | 199 / 9,6 |
| | Convention | 106 / 5,5 | **87** / 4,8 | 88 / **4,7** | 97 / 4,8 |
| | Saint-Pères | 91 / 2,7 | **77** / 2,5 | **77 / 2,4** | 89 / 2,7 |
| 8 derniers jeudis | Champs-Elysées | 219 / 13,4 | 194 / 11,6 | **191 / 11,5** | 195 / 12,1 |
| | Convention | 92 / 3,6 | **71 / 3,0** | **71** / 3,1 | 74 / 3,1 |
| | Saint-Pères | 162 / 4,5 | 107 / 3,6 | 107 / 3,6 | **105 / 3,5** |
| 2e semaine après reprise | Champs-Elysées | 201 / 10,5 | 161 / **8,1** | **161** / 8,2 | 175 / 9,5 |
| | Convention | 125 / 4,9 | 70 / **4,3** | **67** / 4,4 | 77 / 4,7 |
| | Saint-Pères | 123 / 3,5 | 84 / **3,2** | **83 / 3,2** | 91 / 3,3 |
| Semaines du 11/11 | Champs-Elysées | 242 / 16,6 | **223** / 14,0 | 226 / **13,1** | 237 / 14,2 |
| | Convention | 104 / 2,7 | 69 / 2,4 | **60** / 2,3 | 77 / **2,2** |
| | Saint-Pères | 134 / 2,7 | 111 / **2,3** | **109 / 2,3** | 117 / 2,4 |

Lecture :

1. **Tous les modèles battent nettement la baseline**, sur toutes les périodes.
2. **LightGBM ≈ XGBoost** : écarts de 0 à 3 % en général. LightGBM gagne 53 % des cutoffs × arcs sur le débit et
   60 % sur l'occupation, autrement dit du bruit. Seule exception : le débit de Convention la semaine du 11/11
   (60 contre 69), sur 3 semaines seulement.
3. **Le MLP perd sur le débit** : 5 à 15 % de RMSE en plus sur les jeudis réguliers et les reprises. Il ne bat XGBoost
   que sur 26 % des cutoffs × arcs pour le débit, 37 % pour l'occupation. Il reste proche sur l'occupation (0 à 8 % d'écart, 17 % sur les Champs-Elysées en reprise) et fait jeu
   égal sur les 8 derniers jeudis.
4. **Le MLP est instable** : selon la graine aléatoire, son RMSE débit varie de ±5 % (Convention, jeudis réguliers :
   97 à 106). Les boostings sont stables à graine fixe.
5. **Le 11/11 seul** : le MLP prédit mieux l'occupation de Convention (1,2 contre 3,0 pour XGBoost), mais nettement
   moins bien le débit de Convention et de Saint-Pères (86-90 contre 56-67).

## Moyenne de modèles

Moyenner les prédictions gagne au mieux 1 à 2 % de RMSE : les modèles se trompent sur les mêmes heures. Sur les
jeudis réguliers 2022-2024, la moyenne XGBoost + LightGBM + MLP donne par exemple 183 / 9,2 sur les Champs-Elysées,
contre 186 / 9,5 pour XGBoost seul. Pas assez pour justifier trois modèles au rendu.

## Pourquoi le réseau ne gagne pas ici

- Les données sont **tabulaires et peu nombreuses** : ~100 000 lignes, 48 features hétérogènes (booléens, compteurs,
  lags). C'est le terrain où les boostings dominent les réseaux denses.
- Les ruptures (JO, nouveaux régimes, jours fériés rares) se traduisent par des seuils, que les arbres apprennent
  directement. Le réseau doit les approcher avec des fonctions lisses.
- On n'a rien réglé. Un réseau dépend bien plus que XGBoost de ses hyperparamètres (taille, taux d'apprentissage,
  régularisation) : son écart pourrait se réduire avec du réglage, mais pas au point de compenser son instabilité.

## Pistes si on veut aller plus loin

- **Régler XGBoost** avant tout : c'est le modèle retenu, et le moins coûteux à faire tourner.
- Un réseau séquentiel (LSTM, TCN, Transformer temporel) qui lit directement les dernières semaines heure par heure.
  C'est un autre modèle, qui n'utiliserait pas les mêmes entrées : à tester seulement si on veut sortir du cadre
  « mêmes features ».

## Reproduire

```bash
pip install -r requirements.txt                      # torch ajouté
python -m src.evaluate --model xgb --n-folds 8
python -m src.evaluate --model gbm --n-folds 8
python -m src.evaluate --model mlp --n-folds 8
```

Sur macOS, PyTorch et XGBoost/LightGBM embarquent chacun leur runtime OpenMP. Le MLP force donc PyTorch sur un seul
thread (`src/models/mlp.py`) : sans ce réglage, PyTorch se bloque ou XGBoost plante selon l'ordre des imports.
