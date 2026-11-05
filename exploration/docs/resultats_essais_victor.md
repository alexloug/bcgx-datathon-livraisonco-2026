# Essais d'amélioration de XGBoost : ce qui marche, ce qui ne marche pas

Victor, du 3 au 5/11/2026. Code des essais : `exploration/essais_victor/`. Ce qui est retenu : `src/models/ensemble.py`
(`--model ensemble`) et l'option `--manif` de `src/predict.py`.

## En bref

- Le XGBoost de départ est déjà proche de son maximum : un seul modèle pour les 3 rues, toutes les colonnes, toutes
  les données depuis 2021. Sur une trentaine d'idées testées, **3 sont retenues**. Ensemble, elles apportent peu
  (de l'ordre de 1 à 2 %).
- Ce qui pèse le plus sur le score du rendu, c'est la semaine elle-même : la manifestation du samedi 7/11 et le
  11 novembre férié (voir la dernière partie).

## Comment on a jugé une idée

- **Backtest** : on coupe les données un jeudi à 23h, on saute le vendredi et on prédit du samedi au mercredi,
  exactement comme pour le rendu. On calcule le RMSE par rue et par cible (débit, occupation).
- **Bruit** : changer seulement la graine aléatoire fait varier le RMSE de 1,7 à 2,6 % (moyenne sur 16 jeudis). Un gain
  plus petit que ça ne prouve rien. On moyenne donc 3 graines par variante.
- **Règle d'adoption** : gain moyen d'au moins 1 %, au moins 65 % des (jeudi, rue, cible) gagnés, aucun groupe de
  jeudis en dessous de −2 %, et aucun couple (rue, cible) en dessous de −5 %.
- **Jeudis testés** : 16 jeudis pour les premiers essais (automnes 2024 et 2025, octobre 2026). 31 jeudis pour les
  tests de la nuit : un tous les 2 mois en 2022-2024, les 8 jeudis de septembre-octobre 2026, et les 3 semaines du
  11 novembre (2023, 2024, 2025).

Dans les tableaux, un gain positif veut dire que l'erreur baisse.

## Le grand tableau

### Retenu

| Stratégie | Résultat | Pourquoi on la garde |
|---|---|---|
| **Retirer de l'entraînement les jours où le capteur d'occupation dérive** (l'occupation saute de +60 % ou plus alors que le débit est normal), pour l'occupation seulement | +2,0 % au total, confirmé avec 600 arbres. Occupation : Champs +2,1 %, Convention +5,8 %, Saint-Pères +6,8 %. Débit inchangé. 65 % gagnés | Corrige des erreurs de mesure, ne dégrade rien. Attention : le gain vient de 2022-2024 (+5,9 %), il est nul sur 2026 et sur les semaines du 11/11. Il faut s'attendre à presque rien au rendu |
| **Moyenne XGBoost + LightGBM** | +0,3 à +1,5 % sur les 6 couples (rue, cible), 68 % gagnés | Le seul petit gain régulier partout. Deux modèles voisins font des erreurs un peu différentes : la moyenne les compense |
| **Moyenne de plusieurs graines** | 0 à +1,4 %, 57 % gagnés | Le score moyen bouge peu, mais la prédiction dépend moins du hasard d'une seule graine. Ne coûte que du temps de calcul |

### Testé, pas retenu : structure du modèle

| Stratégie | Résultat | Pourquoi on ne la garde pas |
|---|---|---|
| Un modèle par rue | −4,0 à +2,3 % selon la rue, 47 % gagnés | Le modèle commun apprend des 3 rues à la fois : plus de données pour chaque règle |
| Un modèle par tranche horaire | −3,6 à −0,1 %, 38 % gagnés | Moins de données par modèle, et l'heure est déjà une colonne |
| Un modèle par heure | −6,2 à −0,6 %, 35 % gagnés | Idem, en pire |
| Un modèle par jour de semaine | jusqu'à −77 % (débit Convention le 11/11) | Trop peu de données par modèle |
| Un modèle spécial fériés | 11/11 : +15 % sur le débit des Champs, mais −20 % sur Convention et Saint-Pères | Incohérent d'une rue à l'autre, sur 3 jours seulement |
| Plus de poids aux jours fériés | Mélangé, et autres jours dégradés (jusqu'à −6,7 %) | Ce qu'on gagne sur les fériés, on le perd ailleurs |

### Testé, pas retenu : données d'entraînement

| Stratégie | Résultat | Pourquoi on ne la garde pas |
|---|---|---|
| Plus de poids aux données récentes (demi-vie 6 mois, 1 an, 2 ans) | Débit : −4,0 à +0,8 %. Gagne 36 à 53 % des jeudis | Les années anciennes restent utiles (saisons, fériés, vacances) |
| Commencer l'entraînement plus tard (2023, 24, 18 ou 12 derniers mois) | −0,6 à −3,2 % | Idem : moins d'historique, moins bon |
| Moins de poids aux heures reconstruites des Champs-Elysées (capteur mort depuis le 19/11/2025) | Entre −3,7 et +3,2 %, sans tendance | Neutre, dans le bruit |
| Retirer les jours anormaux (travaux, déviations), détectés automatiquement ou connus | −0,6 à −1,1 % | Le modèle a besoin de voir ces jours. Et au rendu, on ne peut pas retirer une déviation de la semaine à prédire |
| Ajouter les lignes des autres capteurs des Champs (option de Robin) | −1,3 %, 50 % gagnés (+2,2 % sur les semaines du 11/11, mais −1,6 % sur 2026) | Pas de gain d'ensemble |

### Testé, pas retenu : colonnes (features)

| Stratégie | Résultat | Pourquoi on ne la garde pas |
|---|---|---|
| Analyse d'importance (on mélange une famille de colonnes et on mesure l'erreur) | Historique du trafic : +180 à +200 % d'erreur s'il disparaît. Calendrier : +15 à +44 %. Rue : +9 %. Événements : +1 %. Météo : 0 % sur le débit, +3 % sur l'occupation | Aucune colonne inutile à retirer franchement |
| Retirer la météo | −0,5 %, 34 % gagnés | On la garde |
| Retirer les grèves 2023 des événements | −0,5 %, 30 % gagnés | On les garde |
| Données les plus fraîches selon le jour prédit (pour le samedi, on connaît déjà le jeudi) | +1,0 %, 47 % gagnés | Dans le bruit. Et il faudrait modifier le dataset et le rendu |
| Profil médian rue × jour × heure × vacances en colonne | +0,5 %, 60 % gagnés | Dans le bruit |
| Nouveaux événements (fichier v2) | Pas testé : le fichier n'est pas dans le dépôt | À refaire si le fichier est fourni |

### Testé, pas retenu : cible et fonction de perte

| Stratégie | Résultat | Pourquoi on ne la garde pas |
|---|---|---|
| Perte de Huber, petit seuil | −18 % | La note est un RMSE (erreur au carré) : minimiser autre chose s'en éloigne |
| Perte de Huber, grand seuil | −3,8 % | Idem |
| Erreur absolue | +1,9 %, 60 % gagnés, −0,4 % sur les semaines du 11/11 | Presque. Pas assez régulier |
| Prédire log(1 + y) | +1,9 %, 59 % gagnés, −1,1 % sur les semaines du 11/11 | Presque. Moins bon justement sur les semaines qui ressemblent au rendu |
| Prédire l'écart à la moyenne des semaines passées | −1,0 % | Moins bon |
| Prédire l'occupation à partir du débit prédit | +1,0 %, 50 % gagnés | Dans le bruit |

### Testé, pas retenu : réglage de XGBoost

| Stratégie | Résultat | Pourquoi on ne la garde pas |
|---|---|---|
| Recherche d'hyperparamètres (`RandomizedSearchCV`), version 10 min | +1 à +4 % sur 5 couples, −3,4 % sur l'occupation de Saint-Pères | Dans le bruit |
| Recherche d'hyperparamètres, version nuit (44 combinaisons) | Interrompue faute de temps. Les 10 meilleures combinaisons sont à moins de 1,7 % les unes des autres | Rien à gagner de net |
| Arrêt précoce (4 dernières semaines en validation) | −6,3 % | Moins d'arbres, moins bon |
| Contraintes de sens (la prédiction monte quand les lags montent) | −3,8 % | Trop rigide |

### Testé, pas retenu : combiner des modèles

| Stratégie | Résultat | Pourquoi on ne la garde pas |
|---|---|---|
| XGBoost plus aléatoire (60 % des lignes, 50 % des colonnes) | −1,1 à +1,7 %, 42 % gagnés | Mélangé |
| Random Forest | Débit : −4 à −7 % | Moins bon que le boosting sur le débit |
| XGBoost + LightGBM + Random Forest | 55 % gagnés | La Random Forest tire le débit vers le bas |
| Mélange avec la baseline (poids réglé par rue) | −0,7 %, 36 % gagnés | La baseline n'apporte rien de plus |

## Résultat du modèle final

<!-- complété par le backtest du modèle final -->

## La semaine du rendu (7 au 11/11/2026)

Méthode : on rejoue des semaines passées comme pour le rendu, puis on calcule, jour par jour, le rapport
**débit mesuré / débit prédit**. 1 veut dire que le modèle tombe juste, 0,65 qu'il surestime le trafic de 35 %.

| Jour | Que faire | Pourquoi |
|---|---|---|
| **Samedi 7/11, grosse manifestation** | **Corriger après le modèle** : débit et occupation × 0,65 aux Champs-Elysées, × 0,8 à Convention, × 0,65 à Saint-Pères, de 12h à 23h (`python -m src.predict ... --manif 2026-11-07`) | Les samedis de manif des 17 et 24/10/2026, le modèle surestime l'après-midi et la soirée : le rapport descend jusqu'à 0,42 (Champs, 18h-20h le 17/10). Le matin est normal. Ce n'est pas l'effet des vacances : les premiers samedis des vacances de la Toussaint 2023-2025, le rapport reste entre 0,87 et 1,16. Testé en réglant les facteurs sur un samedi et en vérifiant sur l'autre, l'erreur baisse de 22 à 48 % sur le débit et de 29 à 48 % sur l'occupation. Le modèle ne peut pas l'apprendre seul : la colonne événements ne pèse que 1 % |
| Mardi 10/11, petite manifestation | Rien | Les mardis d'octobre 2026, le rapport moyen est de 1,04 (Champs), 1,03 (Convention) et 0,95 (Saint-Pères). C'est le même que les lundis et les mercredis : aucun effet visible |
| Mercredi 11/11, férié | Rien de plus | Déjà dans le dataset : jour férié, fermeture du haut des Champs le matin pour la cérémonie, et mardi marqué « veille de férié ». Les 11/11 de 2023 à 2025, le modèle n'a pas de biais constant : rapport de 0,77 à 1,24 aux Champs, de 0,92 à 1,04 à Saint-Pères. L'erreur reste plus forte qu'un jour normal, mais rien ne permet de la corriger dans un sens précis |

**Risque de la correction du samedi** : sur les samedis normaux de septembre-octobre, appliquer ces facteurs à tort
multiplierait l'erreur du débit par 2 à 2,4. On ne l'applique donc que si la manifestation est sûre. Si son parcours
évite clairement une des trois rues, mieux vaut ne pas corriger cette rue-là.
