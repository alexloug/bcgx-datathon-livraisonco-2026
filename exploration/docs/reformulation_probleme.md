# Reposer le problème : du débit horaire au créneau de livraison

04/11/2026. Exercice « hors du cadre » côté données : on ne règle pas XGBoost, on se demande si on prédit la bonne
chose, mesurée de la bonne façon, avec les bonnes données. Scripts d'analyse hors dépôt (scratchpad de session),
mêmes données locales que `data/`.

## 1. Consignes vérifiées

Source : deck `2026 - Brief - LivraisonCo.pptx` (29/10/2026, hors dépôt, dans le dossier parent du repo).

| Point | Ce que dit le brief | Slide | Écart avec le dépôt |
|---|---|---|---|
| Cibles | débit horaire et taux d'occupation de la voie | 6, 11 | aucun |
| Arcs | Champs-Elysées (Washington → Berri), Convention (Lecourbe → Blomet), Saint-Pères (Voltaire → Université), filtrés par nœuds | 8 | aucun (`src/config.py`) |
| Fenêtre | « du 07 au 11 novembre », 360 lignes = 3 arcs × 120 heures | 11, 13 | aucun : **07/11 00:00 → 11/11 23:00, samedi → mercredi** |
| Date limite | envoi avant le **vendredi 06/11 18 h** | 11, 16 | « 5 jours à partir de vendredi » confond la date limite et la fenêtre |
| Données | « dernières données disponibles », à retélécharger avant le rendu | 7, 11 | l'API publie en J-1 : téléchargées le 06/11, dernière heure le 05/11 23:00 (cohérent avec `recent.parquet` du 03/11, qui s'arrête au 03/11 00:00) |
| Métrique | RMSE des 2 variables, sur les 3 axes, du 07 au 11/11 | 12 | **le dépôt mesure de 6 h à 20 h ; le brief compte les 24 heures** |
| Note | Data Science /5 (RMSE + présentation du livrable 1), Business /5 (solution d'optimisation + présentation) ; égalité départagée par la RMSE | 12 | — |
| Format | `output_<groupe>.csv`, `sep=";"`, colonnes `arc`, `datetime` (`%Y-%m-%d %H:%M`), `debit_horaire`, `taux_occupation` | 13 | aucun (`check_submission` reprend le test) |
| Contexte business | livreurs à **10 €/h de jour (8 h-20 h), 14 €/h de nuit**, créneaux de 4 à 8 h ; le client veut définir ses créneaux de la semaine suivante | 5, 10 | la nuit est une option, pas une heure à ignorer |
| Données externes suggérées | météo, **caractéristiques des routes**, vacances, fériés, confinements | 7 | caractéristiques des routes non utilisées |

**À demander au tutorat du jeudi 05/11** : (1) comment les 6 RMSE sont agrégées (somme brute ? moyenne ? normalisée
par arc ou par variable ?) ; (2) toutes les heures comptent-elles bien ; (3) quelles heures le client livre-t-il.

Pourquoi (1) compte : en RMSE brute, le débit (60-170 véh/h) écrase l'occupation (2-10 points), et les Champs
pèsent 2 à 3 fois plus que les deux autres arcs. Une somme brute revient à optimiser le débit des Champs ;
une RMSE normalisée donnerait du poids à l'occupation de Convention et Saint-Pères.

Effet de la fenêtre horaire (backtest XGBoost du dépôt, 52 jeudis consécutifs de 2023 et 18 de 07-10/2026) : la RMSE
sur 24 h est jusqu'à 19 % plus basse qu'en 6 h-20 h (la nuit est facile ; écart nul pour le débit des Champs en 2023),
le classement des arcs ne change pas.
Passer `src/evaluate.py` en 24 h ne remettra pas en cause les choix passés mais donne le bon chiffre à annoncer.

## 2. Lecture business

Le client ne consomme pas un débit : il choisit **quand** envoyer ses livreurs. Ce qui compte pour lui :

1. **Le temps de parcours par créneau**, qui dépend de la congestion, donc de l'occupation, pas du débit.
2. **Les journées anormales**, qui font rater une tournée entière.
3. **L'incertitude** : un créneau « probablement fluide » ne vaut pas un créneau « sûrement fluide ».

### Relier débit et occupation à un temps de parcours

Une boucle mesure le débit q et l'occupation o. La densité est proportionnelle à o, la vitesse à q / o, donc le
**temps de parcours d'un arc est proportionnel à o / q**. En rapportant o / q à sa valeur de nuit (1 h-4 h, trafic
libre), on obtient un **indice de temps de parcours** (1 = fluide, 2 = deux fois plus long) sans connaître le nombre
de voies. Limite : suppose une longueur moyenne de véhicule stable (bus, camions la font varier) ; bruité sous
30 véh/h.

Indice médian par heure, jours ouvrés, 2021-2025 :

| Heure | 7 h | 8 h | 9 h | 11 h | 13 h | 14 h | 17 h | 18 h | 19 h | 21 h |
|---|---|---|---|---|---|---|---|---|---|---|
| Champs-Elysées | 0,85 | **0,89** | 0,96 | 1,31 | 1,78 | **1,84** | 1,77 | 1,78 | 1,72 | 1,56 |
| Convention | 1,26 | 1,34 | 1,48 | 1,39 | 1,29 | 1,28 | 1,62 | **2,20** | 1,77 | 1,20 |
| Saint-Pères | 0,99 | 1,09 | 1,08 | 1,21 | 1,23 | 1,21 | 1,20 | 1,12 | 1,03 | 1,02 |

Aux heures de pointe (8-9 h, 17-18 h), l'indice P90 vaut 2,3 sur les Champs et **4,3 sur Convention** : c'est la queue
de distribution, pas la moyenne, qui fait rater un créneau.

Lecture pour la feuille de route : les Champs sont fluides le matin (8-10 h) et chargés de 12 h à 21 h ; Convention a
une pointe du soir marquée ; Saint-Pères varie peu. En coût horaire × indice, la nuit (14 €) ne bat le jour (10 €) sur
aucun des trois arcs, sauf à comparer avec l'après-midi des Champs (14 € × 0,9 la nuit contre 10 € × 1,8). L'arc
mesure 150 à 250 m : c'est un **indicateur de l'état du quartier**, pas le temps d'une tournée.

### Ce que le modèle actuel rate, en termes business

Backtest XGBoost du dépôt, 6 h-20 h (2023 : 52 semaines ; 2026 : 18 semaines) :

| | Champs-Elysées | Convention | Saint-Pères |
|---|---|---|---|
| Heures en congestion (occupation > 15 %, seuil « pré-saturé » de la Ville) | 48-69 % | 2-9 % | 3-4 % |
| Part de ces heures prédites congestionnées | 92-97 % | 0-59 % | 0-3 % |
| Classe de trafic exacte (fluide / pré-saturé / saturé / bloqué) | 71 % | 91-98 % | 95-97 % |
| Erreur médiane sur l'indice de temps de parcours | 17-20 % | 12-22 % | 10-12 % |
| Part de l'erreur² d'occupation portée par les 10 % pires jours | 55-65 % | 47-71 % | 48-60 % |
| Largeur P10-P90 de l'erreur d'occupation (points) | 15-16 | 2-5 | 5 |

Trois constats :

- Les seuils `etat_trafic` publiés par la Ville sont des seuils d'occupation (fluide < 15 %, pré-saturé 15-30 %,
  saturé 30-50 %, bloqué > 50 %, vérifié sur les données). On peut donc livrer au client des **classes de trafic**
  par créneau, plus lisibles qu'un pourcentage.
- Sur Convention et Saint-Pères, **les rares heures congestionnées ne sont presque jamais prévues** : le modèle prédit
  bien la moyenne, mal l'exception.
- **La moitié à deux tiers de l'erreur vient de 10 % des jours** (travaux, déviations, événements,
  `docs/determinants_trafic.md`). Gagner sur ces jours compte plus que gagner 2 % partout.

## 3. Typologie des 3 axes

### Ce que disent les données

Diagramme fondamental (débit médian par tranche d'occupation, heures mesurées depuis 06/2021) :

| | Champs-Elysées | Convention | Saint-Pères |
|---|---|---|---|
| Débit max (plateau) | 1 150 véh/h | 930 véh/h | 730 véh/h |
| Occupation au plateau | ~31 % | ~35 % (plateau dès 8 %) | ~23 % |
| Occupation médiane de jour | 20 % | 4,5 % | 7,4 % |
| Heures de jour au-delà du plateau (saturé) | 13,5 % | 0,9 % | 0,7 % |
| Heures de jour pré-saturées ou pire (`etat_trafic`) | **67 %** | 7 % | 4 % |
| Corrélation débit-occupation de jour | 0,55 | 0,58 | 0,81 |

- **Les arcs ne diffèrent pas surtout par leur taille mais par leur régime.** Les Champs vivent sur le plateau du
  diagramme : le débit y bouge peu quand l'occupation bouge beaucoup, d'où une occupation mal expliquée par le
  calendrier. Saint-Pères est presque toujours fluide : l'occupation y suit le débit (corrélation 0,81).
- **Normaliser par le nombre de voies ne marche pas** : avec 3 voies, les Champs passeraient ~380 véh/h/voie, avec
  1 voie Saint-Pères ~730. La capacité vient des feux, pas du profil en travers. Mieux vaut normaliser par la
  **capacité empirique** (plateau du diagramme) ou par le niveau propre de l'arc.
- **À vérifier sur le terrain** : les données ne montrent qu'un sens pour Saint-Pères (arcs 191 → 194 tous dans le
  même sens), cohérent avec une rue à sens unique, pas avec « 2 × 1 voie ». Convention et les Champs ont bien les deux
  sens (5671/5672, 4264/4262).
- Les formes de profil journalier se ressemblent pour le débit (corrélation 0,96-0,98 entre les 3 arcs), beaucoup
  moins pour l'occupation (0,77-0,97) : partager l'information entre arcs a plus de sens pour le débit… sauf sur
  les Champs (voir A3).

### Effets de réseau

Anomalie journalière (écart au profil heure × jour × férié × vacances, 7 h-20 h, 2021-2025) :

| | Champs | Convention | Saint-Pères |
|---|---|---|---|
| Corrélation avec la moyenne de la ville (autres rues) | 0,07-0,19 | 0,19-0,28 | 0,21-0,23 |
| Corrélation avec les autres arcs de la même rue | 0,35-0,43 | 0,62-0,74 | 0,41-0,45 |
| Persistance à J-7 (ce qu'on peut encore voir au cutoff) | 0,18-0,38 | 0,34-0,47 | 0,45-0,49 |

Les anomalies sont **locales à la rue**, pas à la ville. Elles persistent une semaine (0,2-0,5), mais les lags et le
niveau 28 j le captent déjà (A6). Les arcs amont / aval servent surtout à fiabiliser un arc qui décroche (le proxy
des Champs le fait déjà), pas à anticiper : avec 144 h d'écart, la propagation d'une congestion (minutes) est
invisible.

## 4. Approches classées (impact × effort)

| # | Approche | Impact attendu | Effort | Statut |
|---|---|---|---|---|
| A1 | Aligner la mesure sur la note : RMSE 24 h et agrégation demandée au tutorat | élevé (on optimise la bonne chose) | très faible | à faire |
| A2 | Traduire les prédictions en indice de temps de parcours et classes de trafic par créneau | élevé sur la note Business | faible | chiffré (§2) |
| A3 | Modèle partagé entre arcs de la même rue pour l'occupation des Champs | moyen (-6 à -9 % sur l'arc qui pèse le plus) | moyen | testé, gain robuste |
| A4 | Traiter les jours anormaux à part : veille des déviations sur la fenêtre et intervalles de confiance | élevé si un épisode tombe dans la fenêtre | moyen, manuel | analysé |
| A5 | Normaliser par la capacité empirique et utiliser le régime (diagramme fondamental) | faible à moyen | moyen | analysé, non testé en modèle |
| A6 | Cible relative au niveau 28 j, correction du biais de la semaine précédente | nul | — | testé, rejeté |

### A1. Mesurer ce qui est noté

- **Chiffres** : RMSE 24 h = 0,81 à 1,0 × RMSE 6 h-20 h selon l'arc ; débit 60-170 véh/h contre occupation 2-10 points.
- **Données** : aucune. Changer la fenêtre horaire de `rmse_table` (ou ajouter une colonne 24 h).
- **Risque** : aucun. Si l'agrégation est une somme brute, tout effort sur l'occupation est invisible dans la note :
  il faut le savoir avant d'y passer du temps.

### A2. Parler en temps de parcours et en créneaux

- **Chiffres** : §2. Erreur médiane du modèle sur l'indice de temps de parcours 10-22 %, classes de trafic exactes
  71 % (Champs) à 98 % (Convention).
- **Données** : déjà là (débit et occupation prédits). Pour aller plus loin (feuille de route) : longueur des arcs
  (`geo_shape`), vitesses commerciales (données FCD type TomTom / Waze, payantes), historique de tournées du client.
- **Usage** : une grille jour × heure × arc en 4 couleurs + indice ; un créneau est « sûr » si sa borne P90 reste
  fluide. C'est le pont entre le livrable 1 et la feuille de route (slide 10 : créneaux de 4-8 h, coût jour/nuit).
- **Risque** : l'indice o / q est une approximation (longueur des véhicules) ; à présenter comme relatif.

### A3. Partager l'information entre arcs de la même rue

Test : XGBoost à réglage fixe, features minimales (calendrier, lags 168/336/504 h, niveau 28 j, pluie, neige,
capacité et moyennes de l'arc), 19 jeudis de 06/2022 à 12/2024 (un toutes les 6 semaines) + 8 derniers jeudis 2026,
6 h-20 h. Un modèle par arc contre un modèle entraîné sur tous les arcs de la rue (3 à 26 arcs selon la rue) ou des 3 rues
(51 arcs), cible rapportée au niveau 28 j pour les modèles partagés.

| RMSE (2022-2024) | Par arc | Même rue | 3 rues |
|---|---|---|---|
| Champs, occupation | 10,43 | **9,45** (gagne 74 % des cutoffs) | 9,84 (74 %) |
| Champs, débit | 164,6 | **156,9** (48 %) | 159,6 (65 %) |
| Convention, occupation | 4,84 | **4,71** (62 %) | 5,01 |
| Convention, débit | **89,7** | 96,9 | 92,6 |
| Saint-Pères, occupation | 3,78 | 3,77 | 3,79 |
| Saint-Pères, débit | 79,3 | 81,4 | 81,2 |

Sur les 8 derniers jeudis (Convention et Saint-Pères seulement, Champs muet) : Convention gagne un peu (débit 71 → 68),
Saint-Pères perd nettement (débit 96 → 109-116), car son niveau 28 j est gonflé par la déviation de septembre.

- **Lecture** : le partage aide **l'occupation des Champs**, l'arc et la variable les plus difficiles, de façon
  stable (3 cutoffs sur 4). Ailleurs, c'est du bruit ou une perte.
- **Données** : déjà téléchargées (`data/trafic/`, 51 arcs). Il faut étendre `build_dataset` aux arcs voisins.
- **Risque** : depuis 11/2025, la cible des Champs est elle-même reconstruite à partir de 4274 et 4281, deux des arcs
  partagés : le gain mesuré en 2022-2024 n'est pas garanti sur une cible proxy. Ce test n'utilise pas les features
  du dépôt (événements, calendrier scolaire) : à reconfirmer avec le XGBoost du dépôt avant adoption.

### A4. Les jours anormaux d'abord

- **Chiffres** : 10 % des jours portent 47-71 % de l'erreur² d'occupation ; les heures congestionnées de Convention
  et Saint-Pères sont prévues dans 0-59 % des cas.
- **Ce qui est déjà acquis** : les emprises de chantier n'aident pas (`docs/determinants_trafic.md` §5) ; les
  anomalies sont locales à la rue et ne se voient pas dans les autres rues.
- **Données** : arrêtés de circulation et déviations (Préfecture de police, pas en open data), agenda des
  manifestations et grèves du 07-11/11, cérémonie du 11/11 à l'Étoile. Pour la feuille de route : un flux temps réel
  (J-1 au lieu de J-6) changerait plus la précision que n'importe quel modèle : l'anomalie d'un jour est corrélée
  à 0,45-0,79 avec celle de la veille, contre 0,2-0,5 avec celle de la semaine précédente.
- **Livrable** : intervalles P10-P90 par créneau (largeur actuelle 2-16 points d'occupation selon l'arc), et une
  vérification manuelle de la fenêtre avant le rendu.
- **Risque** : manuel, non reproductible ; gain nul si la semaine est calme.

### A5. Capacité et régime de congestion

- **Chiffres** : §3. Les Champs sont pré-saturés 67 % des heures de jour, les deux autres arcs 4-7 %.
- **Pistes** : features « capacité empirique » et « occupation critique » pour un modèle partagé (déjà dans A3) ;
  pour Saint-Pères, dériver l'occupation du débit prédit via le diagramme (corrélation 0,81) pour imposer la
  cohérence des deux cibles ; pour les Champs, prédire la probabilité d'être au-delà du plateau.
- **Données** : déjà là ; nombre de voies et carrefours à feux via le référentiel voirie de l'open data Paris si on
  veut une version physique.
- **Risque** : gain RMSE incertain (non testé en modèle) ; surtout utile pour expliquer, moins pour la note.

### A6. Rejetés après test

- **Cible relative au niveau 28 j, un modèle par arc** : pas mieux que la cible absolue (Champs débit 169 contre 165,
  Convention 93 contre 90) et nettement pire quand le niveau est faussé (Saint-Pères, 8 derniers jeudis : 109 contre 96).
- **Corriger chaque semaine du biais observé la semaine précédente** (connu au cutoff) : le biais hebdomadaire ne
  persiste pas (corrélation -0,45 à +0,54 selon l'arc, de signe instable). Avec une demi-correction, RMSE égale ou pire
  (Champs débit 171 → 184 en 2026).

## 5. Recommandation : ce qu'on teste en premier

1. **Avant le tutorat du 05/11** : passer le backtest en 24 h (A1) et préparer les 3 questions du §1. Sans la règle
   d'agrégation, on ne sait pas s'il faut travailler le débit des Champs ou l'occupation de tous les arcs.
2. **Ensuite, en parallèle** :
   - côté modèle : **A3 restreint à l'occupation des Champs**, avec le XGBoost et les features du dépôt (arcs de la rue
     des Champs ajoutés au dataset, cible rapportée au niveau). C'est le seul gain mesuré qui tient sur 3 cutoffs sur 4 ;
   - côté présentation : **A2**, la grille de créneaux (classe de trafic + indice de temps de parcours + P90), qui
     transforme la prédiction en décision pour le CEO et nourrit la feuille de route.
3. **Juste avant le rendu** : la veille manuelle de la fenêtre (A4) : manifestations, cérémonie du 11/11, chantier
   rue de l'Université, fin du chantier du quai Voltaire.
