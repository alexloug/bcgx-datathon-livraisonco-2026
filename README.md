# LivraisonCo — Datathon BCG X 2026

## Le problème

Prédire, heure par heure, le débit et le taux d'occupation de 3 arcs parisiens (Champs-Elysées, Convention, Saint-Pères)
du samedi 07/11/2026 00 h au mercredi 11/11 23 h : 360 lignes, notées en RMSE sur les 24 heures.
Dernière donnée disponible : le 05/11 à 23 h, puis 1 jour sauté et 5 jours à prédire. Rendu : CSV, code et deck avant le vendredi 06/11, 18 h.

## Démarrage

```bash
pip install -r requirements.txt && python -m src.data.update --history 2021 2025   # 1re fois seulement : archives trafic, ~10 min
python -m src.predict --group <groupe> --refresh     # rendu : mise à jour, dataset, prédiction, contrôle BCG -> outputs/output_<groupe>.csv (~30 s)
python -m src.evaluate --model xgb --folds all       # backtest contre la baseline : 8 jeudis + 34 semaines fériées (~20 min ; sans --folds all, 8 jeudis en ~3 min)
```

Python 3.11 ; `unzip` doit être dans le PATH (archives trafic). Sans `--refresh`, `predict` repart du dataset déjà présent.
Les tests : `python -m pytest -q tests`.

## Démarche

1. **Données** : trafic Paris open data depuis le 21/06/2021, nettoyé sans interpolation ; le capteur muet des Champs-Elysées est reconstruit par le capteur voisin ; calendrier, vacances, fériés, événements saisis à la main, météo prévue ([docs/2_donnees.md](docs/2_donnees.md)).
2. **Features** : calendrier, historique décalé d'au moins 144 h (S-1 à S-3, niveau 28 j, 52 semaines) en heure de Paris, vacances, météo prévue, événements, régimes : 44 colonnes sans fuite possible.
3. **Modèle** : ensemble de 3 XGBoost et 3 LightGBM (réglages par défaut) par cible, occupation nettoyée des dérives du capteur, comparé à une baseline saisonnière ([docs/3_modele.md](docs/3_modele.md)).
4. **Validation** : backtest qui rejoue le rendu (cutoff le jeudi 23 h, 1 jour sauté, 5 jours prédits), RMSE sur 24 h, sur les 8 derniers jeudis et 34 semaines contenant un férié.

## Décisions

| Décision | Pourquoi | Preuve |
|---|---|---|
| Données depuis 06/2021 | avant, ruptures de régime (Covid, capacité des Champs en 2019, capteur Convention en 2020) et pas de prévision météo archivée | débit des Champs ÷ 2 et occupation × 2 en 2018-2019 ; [2_donnees](docs/2_donnees.md) §2 |
| Pas d'interpolation | les trous sont des blocs de plusieurs semaines, qu'aucune interpolation ne rend crédibles | Convention absente 7 mois (11/2022), Saint-Pères 11 mois (11/2024) ; [2_donnees](docs/2_donnees.md) §3 |
| Champs reconstruits par le capteur voisin | le capteur 4264 est muet depuis le 19/11/2025 ; sans lui, l'arc n'aurait ni cible ni lag | débit 46 véh/h d'erreur (6 %), occupation 27 % ; [2_donnees](docs/2_donnees.md) §2 |
| Lags en heure de Paris | un décalage de 24 h × n sur la grille UTC vise une heure fausse après le changement d'heure | 07/11 08:00 visait 17/10 09:00 (322 au lieu de 237) ; 0 écart après correction ; [2_donnees](docs/2_donnees.md) §2 |
| Ensemble XGBoost + LightGBM | les deux boostings font des erreurs un peu différentes, la moyenne gagne 1 à 2 % de façon régulière | 29 semaines de backtest, 71 % des cas gagnés ; [exploration/docs/resultats_essais_victor.md](exploration/docs/resultats_essais_victor.md) |
| Réglages par défaut | profondeur, régularisation et recherche aléatoire d'hyperparamètres ne font pas mieux que le bruit | score relatif 0,772 / 0,771 contre 0,771 ; 44 combinaisons tirées au hasard : les 10 meilleures à moins de 1,7 % les unes des autres ; [3_modele](docs/3_modele.md) §4, [exploration/README](exploration/README.md) |
| Samedi 07/11 corrigé à la main (`--manif`) | grosse manifestation annoncée ; le modèle ne peut pas l'apprendre (événements : 1 % de son importance) | samedis de manifestation d'octobre 2026 : prédictions trop hautes de 12 h à 23 h (rapport jusqu'à 0,42) ; facteurs vérifiés sur l'autre samedi : erreur −22 à −48 % ; [4_rendu](docs/4_rendu.md) §1 |
| Météo observée et chantiers écartés | la météo observée est inconnue à la prédiction ; les chantiers ne gagnent qu'un fold sur deux | débit avec / sans chantiers : 193/193, 91/89, 79/78 ; [3_modele](docs/3_modele.md) §4 |
| Heures « Barré » | à compléter avec Victor | à compléter |

## Résultats

RMSE sur les 24 heures, XGBoost / baseline (moyenne des 4 dernières semaines), débit en véh/h puis occupation en points.
Dernière exécution de `python -m src.evaluate --model xgb --folds all`, données au 03/11/2026.

| Arc | Semaines normales (8 jeudis, 03/09 → 22/10) | Semaines fériées (34 semaines depuis 2022) |
|---|---|---|
| Champs-Elysées | 176 / 192 · 8,3 / 9,6 (gain 9 % · 14 %) * | 177 / 225 · 9,4 / 10,2 (gain 22 % · 8 %) |
| Convention | 66 / 82 · 2,5 / 2,9 (gain 20 % · 13 %) | 75 / 122 · 2,7 / 3,3 (gain 38 % · 18 %) |
| Saint-Pères | 104 / 160 · 2,9 / 3,7 (gain 35 % · 22 %) | 73 / 111 · 1,9 / 2,4 (gain 34 % · 22 %) |

\* Cible des Champs entièrement reconstruite sur les semaines normales : la RMSE mesure l'écart au proxy, pas à un capteur.
XGBoost bat la baseline sur les 6 couples arc × cible dans les deux familles de semaines. Les gains les plus faibles sont sur les Champs
(débit en semaines normales : 9 % ; occupation en semaines fériées : 8 %), l'arc à cible reconstruite. Détail par jour d'horizon et par origine de la cible : sortie de la commande.

Les heures « Barré » (fermetures de rue, dont la cérémonie du 11/11 aux Champs-Elysées) sont exclues de l'entraînement et de l'évaluation : l'erreur des Champs sur ces heures est donc sous-estimée. Correction en cours.

## Structure

```
src/config.py            arcs, chemins, fenêtre, décalage minimum de 144 h
src/data/                collecte (update.py, sources/), dataset complet (build_dataset.py), export contrôlé (export_dataset.py)
src/features/build.py    calendrier et historique décalé, en heure de Paris
src/models/              baseline, xgb, ensemble (XGBoost + LightGBM), prétraitement ; registry.py liste les modèles
src/evaluate.py          backtest 24 h, comparaison à la baseline
src/predict.py           point d'entrée : --refresh, prédiction, contrôle du format BCG
tests/                   tests du dataset, du backtest, des modèles et du format
docs/                    1_probleme, 2_donnees, 3_modele, 4_rendu
roadmap/                 feuille de route de Danny : des prédictions au planning de livraison (hors rendu du modèle)
exploration/             pistes écartées (LightGBM seul, MLP, protocole stream 2, essais de Victor, notebooks, docs d'origine)
data/, outputs/          données téléchargées (non versionnées) ; CSV de rendu et backtests
```

## Contributeurs

| Qui | Rôle |
|---|---|
| Robin | données : sources, nettoyage, proxy des Champs, dataset |
| Pétronille | événements saisis et vérifiés |
| Alex | protocole de validation |
| Danny | feuille de route |
| Victor | modèle : essais d'amélioration de XGBoost, modèle final, semaine du rendu |
| Léopold | backtest et structure du dépôt |
