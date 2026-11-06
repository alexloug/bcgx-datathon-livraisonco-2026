# L'erreur du modèle, et ce qu'elle vaut pour un livreur

06/11/2026. Chiffres : `python -m src.analyse_business` (backtest de `xgb`, 23 cutoffs de 2023 à 2026 : 8 derniers
jeudis, 3 semaines du 11/11, 12 jeudis étalés ; 5 jours prédits à J+2…J+6, comme le rendu).

## 1. L'erreur qu'on fait

| Arc | Débit moyen 8-20 h | Erreur moyenne 8-20 h | RMSE 24 h (noté) | vs baseline | Occupation moyenne 8-20 h | Erreur moyenne | RMSE 24 h | vs baseline |
|---|---|---|---|---|---|---|---|---|
| Champs-Elysées | 900 véh/h | 133 véh/h (**15 %**) | 170 | −13 % | 21,3 % | 6,5 pts (31 %) | 9,7 | −10 % |
| Convention | 534 véh/h | 61 véh/h (**11 %**) | 71 | −22 % | 5,6 % | 2,0 pts (35 %) | 2,7 | −13 % |
| Saint-Pères | 563 véh/h | 69 véh/h (**12 %**) | 88 | −31 % | 8,8 % | 2,1 pts (25 %) | 2,6 | −18 % |

Baseline = ce que ferait un planificateur avec un tableur : la moyenne des 4 dernières semaines, même jour, même heure.

- **Débit : 11 à 15 % d'erreur de jour.** Bon niveau pour une prévision à 2-6 jours sans donnée temps réel.
- **Occupation : 25 à 35 % d'erreur relative**, mais sur de petites valeurs (2 points sur Convention et
  Saint-Pères). Sur les Champs, la cible elle-même est reconstruite depuis le 19/11/2025 (capteur muet), avec
  environ 27 % d'erreur de reconstruction : on ne peut pas descendre en dessous.

## 2. Pourquoi elle est là, et pourquoi elle est acceptable

1. **Une grosse part est imprévisible avec des données ouvertes.** Les 10 % pires jours portent 28 à 44 % de
   l'erreur² de débit et 47 à 67 % de celle d'occupation. Ce sont des travaux, déviations, manifestations et
   fermetures qu'aucune source ouverte n'annonce à J-6 (les emprises de chantier testées n'aident pas).
2. **L'horizon impose un recul de 6 jours.** Données publiées en J-1, prédiction de J+2 à J+6 : aucune information
   de moins de 144 h. L'anomalie d'un jour est corrélée à 0,45-0,79 avec la veille, à 0,2-0,5 avec la semaine
   précédente. Un flux temps réel ferait plus que n'importe quel modèle.
3. **Le modèle est déjà au plafond de ce que les features apportent.** Ajouter des arcs similaires (17 arcs,
   710 000 heures) ou retirer des features ne change rien au-delà de ±2 %
   ([`arcs_similaires_backtest.md`](arcs_similaires_backtest.md), [`selection_features.md`](selection_features.md)).
4. **Surtout : l'erreur ne change presque pas la décision.** Le client ne consomme pas un débit, il choisit un
   créneau. Une erreur de niveau qui touche toute la journée ne change pas l'ordre des créneaux (section 3).

**Limite à dire** : le modèle prédit bien la journée type, mal l'exception. Les rares heures de congestion de
Convention et Saint-Pères ne sont presque jamais prévues. D'où les deux compléments : une fourchette P10-P90 par
créneau, et une veille manuelle des événements de la semaine.

## 3. Ce que ça vaut pour choisir un créneau

On traduit débit q et occupation o en **indice de temps de parcours** : o / q rapporté à sa valeur de nuit
(1 = fluide, 2 = deux fois plus long ; [`reformulation_probleme.md`](reformulation_probleme.md) §2). Chaque jour, le
livreur choisit un créneau de 4 h entre 8 h et 20 h. Indice moyen du créneau choisi, sur 71 à 85 jours par arc :

| Arc | Meilleur possible (oracle) | **Modèle** | Habitude fixe | Baseline | Créneau au hasard | Après-midi 14-18 h | Pire créneau |
|---|---|---|---|---|---|---|---|
| Champs-Elysées | 1,24 | **1,29** | 1,29 | 1,29 | 2,06 | 2,47 | 2,86 |
| Convention | 1,30 | **1,52** | 1,58 | 1,52 | 1,69 | 1,62 | 2,24 |
| Saint-Pères | 1,06 | **1,16** | 1,21 | 1,18 | 1,24 | 1,27 | 1,44 |

« Habitude fixe » = le créneau qui était le meilleur en moyenne en 2021-2022, par arc et type de jour (Champs : 8-12 h).

Trois messages :

1. **Le bon créneau vaut beaucoup.** Entre le meilleur et le pire créneau de la journée, le temps de parcours
   varie de +38 % (Saint-Pères) à +118 % (Champs). Sur les Champs, livrer l'après-midi prend deux fois plus de temps
   que le matin.
2. **Le modèle choisit un créneau à moins de 20 % du meilleur 83 à 96 % des jours**, et le meilleur exact 87 % des
   jours sur les Champs.
3. **Face à une habitude bien choisie, le gain moyen est faible (0 à 4 %)… sauf les jours où l'habitude se
   trompe**, soit 20 à 24 % des jours sur Convention et Saint-Pères :

| Jours où l'habitude est à plus de 20 % du meilleur créneau | Part des jours | Habitude | Modèle | Gain |
|---|---|---|---|---|
| Convention | 20 % | 2,54 | 2,25 | **−11 %** |
| Saint-Pères | 24 % | 1,38 | 1,08 | **−22 %** |
| Champs-Elysées | 4 % | 2,39 | 2,39 | 0 % |

Sur les semaines du 11/11, le modèle fait aussi mieux que l'habitude sur Saint-Pères (1,14 contre 1,23).

## 4. Traduction en productivité

Hypothèse à afficher : la conduite représente **40 % du temps d'une tournée urbaine** (le reste : stationnement,
manutention, remise). Seule cette part dépend du trafic. Gain de temps de tournée = 40 % × baisse de l'indice ;
livraisons par heure = 1 / (1 − gain de temps).

| Décision | Baisse du temps de conduite | Temps de tournée | Livraisons par heure | Pour 1 000 h de livreur (10 €/h) |
|---|---|---|---|---|
| **Champs : matin au lieu de l'après-midi** (grille de créneaux) | −48 % | −19 % | **+24 %** | ~190 h, ~1 900 € |
| Champs : créneau du modèle au lieu d'un créneau au hasard | −37 % | −15 % | +18 % | ~150 h |
| Convention : créneau du modèle au lieu de l'habitude, jours atypiques | −11 % | −4,5 % | +5 % | ~45 h |
| Saint-Pères : idem | −22 % | −9 % | +10 % | ~90 h |
| Moyenne, modèle vs habitude, tous les jours | −0 à −4 % | −0 à −1,5 % | +0 à +1,5 % | ~0-15 h |

Avec une part de conduite de 30 % ou 50 %, ces gains se multiplient par 0,75 ou 1,25 ; l'ordre de grandeur ne
change pas.

**Jour ou nuit (14 €/h la nuit contre 10 €/h le jour)** : coût d'une heure de conduite utile = salaire × indice. Le
meilleur créneau de jour (10 € × 1,05 à 1,3 = 10,5 à 13 €) bat toujours la nuit (14 € × 1). **Exception : l'après-midi
des Champs** (10 € × 2,5 = 25 €) coûte plus cher que la nuit. Si une livraison sur les Champs ne peut pas se faire
le matin, mieux vaut la nuit que l'après-midi.

## 5. Comment le présenter

Une phrase : *« Le modèle se trompe de 11 à 15 % sur le trafic, mais presque jamais sur le choix du créneau :
bien choisir son créneau fait gagner jusqu'à 24 % de livraisons par heure sur les axes chargés. »*

1. **Le problème du client** : quand envoyer ses livreurs. Le coût, c'est le temps de conduite dans la congestion.
2. **Le livrable** : une grille jour × créneau × axe, en 4 couleurs (fluide / pré-saturé / saturé / bloqué, seuils
   de la Ville) avec l'indice de temps de parcours, une fourchette P10-P90, et la recommandation de créneau.
3. **La précision**, honnêtement : 11-15 % sur le débit, mieux que la pratique actuelle de 13 à 31 %, et 83 à 96 %
   des jours dans les 20 % du meilleur créneau.
4. **La valeur** : la grille de créneaux (gros gain, structurel : +18 à +24 % de livraisons par heure sur les
   Champs) et le modèle (gain ciblé : +5 à +10 % les jours où la routine se trompe, un jour sur cinq).
5. **Les limites et la feuille de route** : les jours anormaux (abonnement aux arrêtés de circulation, flux temps
   réel J-1), la vraie mesure des Champs, puis passer de 3 arcs à la tournée entière (données de tournée du client,
   temps de parcours FCD).

## Précautions

- Un arc mesure 150 à 250 m : l'indice décrit **l'état du quartier**, pas le temps d'une tournée complète.
- o / q suppose une longueur moyenne de véhicule stable ; à présenter comme un indice relatif.
- 40 % de conduite est une hypothèse, à remplacer par les données de tournée du client dès qu'on les a.
- Les créneaux ont été évalués entre 8 h et 20 h, par jour et par axe, indépendamment : une vraie tournée combine
  plusieurs axes et des contraintes clients (horaires de réception).
