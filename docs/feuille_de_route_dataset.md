# Feuille de route du dataset jusqu'au rendu (vendredi 06/11, 18 h)

04/11/2026. But : livrer à l'équipe modèle un dataset figé et justifié, puis le régénérer vendredi matin avec les
dernières données. On ne règle pas XGBoost : on décide **quelles colonnes** il reçoit et **quelles lignes** servent
à l'entraînement. Contexte : [`reformulation_probleme.md`](reformulation_probleme.md), [`features.md`](features.md),
[`determinants_trafic.md`](determinants_trafic.md).

## Règle de décision

Une famille de features entre dans le dataset du modèle si, avec le XGBoost du dépôt **à réglage inchangé** et en
**RMSE sur 24 h** (la métrique du brief), elle :
1. améliore les jeudis réguliers 2022-2024 (25 cutoffs, un toutes les 6 semaines) ;
2. ne dégrade pas les 9 derniers jeudis (09-10/2026, la période la plus proche du rendu) ;
3. ne dégrade pas les 3 semaines du 11/11 (2023-2025).

Une seule condition ratée suffit à refuser. Pourquoi cette règle : on n'a qu'un rendu, une feature qui aide en 2023
mais pas en 2026 est un pari, pas un gain. Elle se justifie en une phrase devant le jury.

## 1. Ce qui a été testé aujourd'hui

Ablation par famille (RMSE 24 h ; débit en véh/h, occupation en points ; en gras, gain ≥ 3 %) :

| Famille ajoutée | Contenu | Réguliers 2022-2024 | 9 derniers jeudis | Semaines du 11/11 | Décision |
|---|---|---|---|---|---|
| **Rue** (arcs voisins) | écart au niveau des autres arcs de la même rue, à S-1 et S-2 ; tendance du niveau de la rue | Champs débit -2 %, **Convention occ. -4 %**, Champs occ. +4 % | 0 à +3 % sur les 6 RMSE | **Convention occ. -9 %**, Champs débit +2 % | refusée (2) |
| **Lags robustes** | lag S-4, médiane de S-1 à S-4 | Saint-Pères débit -2 %, Convention occ. +4 % | Saint-Pères débit +4 % | Convention occ. -9 %, reste neutre | refusée (1, 2) |
| **Régime de congestion** | part d'heures > 15 % d'occupation sur le niveau 28 j, lag S-1 rapporté au niveau | **Champs débit -3 %** | Champs débit +2 % | **Convention débit -15 %** | refusée (2) |
| Les trois ensemble | — | **Champs débit -6 %** | Champs occ. +3 %, Saint-Pères débit +3 % | **Convention occ. -17 %** | refusée (2) |

Référence (dataset actuel) : réguliers 160 / 78 / 65 véh/h et 7,1 / 3,8 / 2,0 points (Champs / Convention /
Saint-Pères) ; 9 derniers jeudis 173 / 65 / 100 et 9,2 / 2,4 / 2,8.

**Lecture** : les features « historique » sont saturées. Le modèle tire déjà des lags et du niveau 28 j ce que les
arcs voisins savent. Les écarts sont de ±3 %, c'est-à-dire du bruit, sauf sur les 11/11 (3 semaines seulement, trop
peu pour décider). **On fige les 48 features actuelles.** Le gain restant se joue sur les lignes (cible des Champs,
jours anormaux) et sur la fenêtre elle-même, pas sur de nouvelles colonnes.

Déjà refusées avant (rappel, avec leurs chiffres dans les docs cités) : emprises de chantier (une semaine gagnée sur
deux), cible relative au niveau 28 j (pire sur Saint-Pères : 109 contre 96 véh/h), correction du biais de la semaine
précédente (le biais ne persiste pas), normalisation par le nombre de voies (la capacité vient des feux : 380 véh/h
par voie sur les Champs contre 730 sur Saint-Pères).

### Et les corrélations entre axes ?

| Ce qu'on a mesuré (anomalie journalière, hors calendrier) | Valeur | Conséquence pour le dataset |
|---|---|---|
| Entre rues différentes (ex. Saint-Pères / Convention) | 0,07-0,28 | rien à partager : pas de feature croisée entre les 3 arcs |
| Entre arcs de la même rue | 0,35-0,74 | testé (famille « Rue ») : pas de gain stable |
| Même arc, une semaine plus tard | 0,18-0,49 | déjà capté par S-1 et le niveau 28 j |

Le partage entre les 3 arcs se fait déjà par le modèle unique (indicatrices d'arc) : il apprend le calendrier une
fois pour les trois. Sauf le profil des Champs, propre à l'avenue (pointe de l'après-midi), les effets calendaires
sont communs (corrélation des profils de débit 0,96-0,98).

## 2. Ce qu'on traite avant de livrer le dataset

| # | Élément | Pourquoi (chiffre) | Effet sur le dataset | Quand | Statut |
|---|---|---|---|---|---|
| 1 | **Mesurer en 24 h** | le brief note les 24 heures ; nos chiffres sont en 6 h-20 h, jusqu'à 23 % plus hauts | aucun ; `evaluate.py` rend les deux RMSE | mer. soir | à faire |
| 2 | **Cible occupation des Champs** (proxy) | débit reconstruit fiable (ratio 4264 / 4274 = 0,98-1,07 chaque mois de 2025) ; occupation non (ratio mensuel 4264 / 4274 entre 1,7 et 2,8 ; 4274 / 4281 entre 0,8 et 1,5) | garder D1 ; ajouter dans `dataset.json` l'intervalle du ratio pour le documenter. Au rendu, l'occupation prédite des Champs est la valeur la moins sûre des 6 | mer. soir | à faire |
| 3 | **Saint-Pères : pas d'indicateur « quai Voltaire »** | le chantier finit le 30/10 ; un indicateur 10/2024-10/2026 serait confondu avec la ZTL et le trou capteur, et l'invérifiable fin de régime ferait prédire le niveau de 2024 | rien ; on regarde les 31/10 → 05/11 vendredi (contrôle 7) | — | décidé |
| 4 | **Jours anormaux connus** (déviation Saint-Pères 28/08 → 20/09/2026, etc.) | 10 % des jours portent 47 à 71 % de l'erreur ; la déviation de septembre gonfle encore le niveau 28 j de Saint-Pères | fichier séparé `episodes_anormaux.csv` (arc, début, fin, cause) pour **découper les backtests** (erreur avec / sans épisodes). Pas une feature : inconnu dans le futur | jeu. matin | à faire |
| 5 | **Pas de colonne business dans le dataset** | classe de trafic et indice de temps de parcours viennent de la cible à l'heure t : dans le dataset, le prétraitement les prendrait comme features → fuite | calculées **après** la prédiction, dans le CSV d'analyse, pas dans `dataset.parquet` | après le rendu | décidé |
| 6 | **Fenêtre 07-11/11 à la main** | événements et déviations = plus gros écarts ; aucune source ouverte | vérifier manifestations, grèves, cérémonie du 11/11 (déjà saisie), arrêtés de circulation ; compléter `evenements_manuels.csv` si besoin | jeu. | à faire |
| 7 | **Régénération et contrôles du vendredi** | les lags S-1 de la fenêtre ne sont complets qu'avec les données du 05/11 | voir §3 | ven. matin | à faire |

Optionnel, seulement si l'équipe a du temps jeudi : **entraîner sur les arcs de la rue des Champs** (lignes en plus,
pas colonnes). C'est le seul gain stable mesuré (occupation des Champs -6 à -9 %, 3 cutoffs gagnés sur 4), mais
avec un modèle simplifié, pas celui du dépôt. Il faut ajouter des lignes au dataset et adapter le prétraitement aux
nouveaux arcs. Go / no-go jeudi midi avec la même règle de décision ; sinon on n'y touche pas.

## 3. Calendrier

| Quand | Qui | Quoi | Sortie |
|---|---|---|---|
| **Mer. 04/11 soir** | data | éléments 1 et 2 ; dataset v2 = colonnes actuelles, figées | `dataset.parquet` + note de version à l'équipe |
| **Jeu. 05/11 matin** | data | élément 4 (épisodes) ; élément 6 (veille de la fenêtre) | `episodes_anormaux.csv`, événements à jour |
| Jeu. 05/11 journée | modèle | backtests du XGBoost à réglage fixe sur le dataset v2, en 24 h, découpés avec / sans épisodes | tableau de résultats pour la présentation |
| **Jeu. 05/11 soir, tutorat** | tous | les 3 questions ci-dessous | décisions éventuelles (agrégation, Champs) |
| **Ven. 06/11 matin** | data | `update` → `build_dataset` → contrôles → `predict` | `output_<groupe>.csv` |
| Ven. 06/11 avant 18 h | tous | relecture, envoi CSV + code + présentation | rendu |

Questions du tutorat :
1. Comment les 6 RMSE (2 variables × 3 arcs) sont-elles agrégées : somme brute, moyenne, normalisée ?
2. Le capteur des Champs (4264) est muet depuis le 19/11/2025 : sur quelle vérité les Champs seront-ils notés ?
3. Toutes les heures comptent-elles, nuit comprise ?

Contrôles du vendredi (avant `predict`) :
- dernière heure valide = 05/11 23:00 sur les 3 arcs ;
- lags S-1 à S-3 et niveau 28 j présents sur 100 % de la fenêtre (61 % avec les données du 02/11) ;
- météo de la fenêtre = prévision courante, sans trou ;
- Champs : les proxys 4274 et 4281 ont mesuré la semaine passée, et leur rapport reste dans sa plage habituelle
  (débit 1,0-1,15) ;
- Saint-Pères : comparer le 31/10 → 05/11 (après le chantier du quai) au niveau 28 j. Un écart > 15 % est à
  signaler dans la présentation comme risque connu. On ne corrige pas à la main ;
- `check_submission` passe ; profil prédit cohérent avec les mêmes jours des 3 dernières semaines (contrôle visuel).

## 4. Ce que ça donne côté business

Chaque choix se dit en une phrase à un CEO :
- « On prédit à partir du calendrier, de l'historique récent des rues et de la météo prévue ; ajouter les rues
  voisines n'apporte rien de stable, on ne l'a donc pas fait. »
- « Nos erreurs viennent surtout de quelques jours de travaux ou de déviations, que personne ne publie : c'est la
  donnée à acheter ou à collecter en phase 2. »
- « L'occupation des Champs est la prédiction la moins sûre : le capteur est en panne depuis un an et on la
  reconstruit à partir des capteurs voisins. »

La traduction en créneaux (classe de trafic, indice de temps de parcours, coût jour / nuit) vient ensuite, à partir
des prédictions : [`reformulation_probleme.md`](reformulation_probleme.md) §2.
