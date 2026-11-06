# 4. Le rendu : feuille de route et procédure du vendredi 06/11

Rendu : CSV + code + présentation, avant le **vendredi 06/11 à 18 h**.

## 1. Procédure du vendredi

```bash
pip install -r requirements.txt
python -m src.predict --group <nom_du_groupe> --refresh
```

`--refresh` enchaîne : mise à jour du trafic et de la prévision météo, `build_dataset`, `export_dataset` (contrôles
bloquants), prédiction (modèle `ensemble`), test du format BCG, écriture de `outputs/output_<groupe>.csv` (360 lignes, `sep=";"`).
La commande échoue avec un message clair si une source manque, si la mise à jour échoue ou si un contrôle échoue ;
dans ce dernier cas, aucun CSV n'est écrit.

Contrôles à faire avant d'envoyer, dans la sortie de la commande et sur le CSV :

| Contrôle | Attendu |
|---|---|
| Dernière heure observée | 05/11 23:00 sur les 3 arcs (l'API publie en J-1) |
| Features incomplètes sur la fenêtre | aucune (les lags S-1 à S-3 et le niveau 28 j ne sont complets qu'avec les données du 05/11 ; 61 % avec celles du 02/11) |
| Météo de la fenêtre | prévision courante, sans trou |
| Champs-Elysées | les proxys 4274 et 4281 ont mesuré la semaine passée, rapport dans sa plage (débit 1,0-1,15) |
| Saint-Pères | comparer le 31/10 → 05/11 (après le chantier du quai) au niveau 28 j ; un écart > 15 % est à signaler comme risque connu, sans correction manuelle |
| Format | la fonction de test du brief passe (colonnes, types, format de date, 3 arcs, 360 lignes), après notre contrôle « aucun NaN » |
| Profil prédit | cohérent avec les mêmes jours des 3 dernières semaines (contrôle visuel) |
| Fenêtre à la main | manifestations, grèves, cérémonie du 11/11 (déjà saisie), arrêtés de circulation ; voir le tableau ci-dessous |

**La semaine du 07-11/11.** On rejoue des semaines passées et on calcule, jour par jour, débit mesuré / débit prédit
(1 = le modèle tombe juste). Détail : `exploration/docs/resultats_essais_victor.md`.

| Jour | Décision | Pourquoi |
|---|---|---|
| Sam. 07/11, grosse manifestation | `--manif 2026-11-07` : débit et occupation × 0,65 (Champs), × 0,8 (Convention), × 0,65 (Saint-Pères), de 12 h à 23 h | Samedis de manifestation des 17 et 24/10/2026 : le modèle surestime l'après-midi et la soirée (rapport jusqu'à 0,42), le matin est juste. Pas un effet des vacances : premiers samedis de la Toussaint 2023-2025 entre 0,87 et 1,16. Facteurs réglés sur un samedi et vérifiés sur l'autre : erreur −22 à −48 % (débit), −29 à −48 % (occupation). Mais appliqués à tort un samedi normal, ils doublent l'erreur : seulement si la manifestation est confirmée |
| Mar. 10/11, petite manifestation | rien | Mardis d'octobre 2026 : rapport 1,04 / 1,03 / 0,95 (Champs / Convention / Saint-Pères), comme les lundis et mercredis |
| Mer. 11/11, férié | rien de plus | Férié, cérémonie et veille de férié déjà dans le dataset ; sur les 11/11 de 2023 à 2025, pas de biais dans un sens (Champs 0,77 à 1,24, Saint-Pères 0,92 à 1,04) |

Reproduire à la main, étape par étape : `python -m src.data.update` puis `python -m src.data.build_dataset` puis
`python -m src.data.export_dataset` puis `python -m src.predict --group <nom>`. Autres sources optionnelles
(non utilisées par le modèle) : `python -m src.data.update --all`.

## 2. Règle de décision sur les données

Une famille de features n'entre dans le dataset que si, avec XGBoost à réglage inchangé et en RMSE 24 h, elle améliore
les jeudis réguliers 2022-2024, ne dégrade pas les 9 derniers jeudis ni les 3 semaines du 11/11. Résultat : les 48
features actuelles sont figées (voir 3_modele.md §4).

| # | Élément | Pourquoi | Décision |
|---|---|---|---|
| 1 | Mesurer en 24 h | le brief note les 24 heures ; 6 h-20 h est jusqu'à 23 % plus haut | fait : `evaluate` |
| 2 | Cible d'occupation des Champs | débit reconstruit fiable (ratio 4264 / 4274 entre 0,98 et 1,07 en 2025) ; occupation non (1,7 à 2,8) | proxy gardé ; à présenter comme la prédiction la moins sûre |
| 3 | Pas d'indicateur « quai Voltaire » | chantier fini le 30/10 ; un indicateur 10/2024-10/2026 serait confondu avec la ZTL et le trou capteur | rien ; contrôle de vendredi |
| 4 | Jours anormaux connus (déviation Saint-Pères 28/08 → 20/09/2026) | 10 % des jours portent 47 à 71 % de l'erreur | pas une feature (inconnu dans le futur) ; sert à découper les backtests |
| 5 | Pas de colonne business dans le dataset | classe de trafic et indice viennent de la cible à l'heure t : fuite | calculées après la prédiction |
| 6 | Fenêtre 07-11/11 à la main | événements et déviations = plus gros écarts ; aucune source ouverte | veille manuelle jeudi |

## 3. Ce que ça donne côté business

Chaque choix se dit en une phrase à un CEO :
- « On prédit à partir du calendrier, de l'historique récent des rues et de la météo prévue ; ajouter les rues voisines
  n'apporte rien de stable, on ne l'a donc pas fait. »
- « Nos erreurs viennent surtout de quelques jours de travaux ou de déviations, que personne ne publie : c'est la donnée
  à collecter en phase 2. »
- « L'occupation des Champs est la prédiction la moins sûre : le capteur est en panne depuis un an et on la reconstruit
  à partir des capteurs voisins. »

La traduction en créneaux (classe de trafic, indice de temps de parcours, coût jour 10 € / nuit 14 €) se fait à partir
des prédictions (voir 1_probleme.md §2) : une grille jour × heure × arc en 4 couleurs, un créneau étant « sûr » si sa
borne P90 reste fluide.

## 4. Historique du calendrier

| Quand | Quoi |
|---|---|
| Mer. 04/11 | dataset v2 figé et livré à l'équipe modèle ; lags en heure de Paris ; backtest 24 h |
| Jeu. 05/11 | veille de la fenêtre ; backtests XGBoost à réglage fixe ; tutorat du soir (agrégation des 6 RMSE, vérité des Champs, heures comptées) |
| Ven. 06/11 matin | `--refresh`, contrôles, CSV |
| Ven. 06/11 avant 18 h | relecture, envoi CSV + code + présentation |
