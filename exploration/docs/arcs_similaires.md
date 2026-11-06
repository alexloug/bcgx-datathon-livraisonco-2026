# Arcs similaires aux 3 arcs du brief

Script : `exploration/data/arcs_similaires.py`. Résultat : `data/arcs/arcs_similaires.csv` (15 arcs par arc du brief).

```bash
python -m exploration.data.arcs_similaires            # télécharge (~10 min) puis classe
python -m exploration.data.arcs_similaires --local    # reclasse à partir des fichiers déjà téléchargés (quelques secondes)
```

## Pourquoi

Nos 3 arcs ont peu d'exemples des jours qui comptent pour le rendu : 1 à 5 « 11 novembre » valides selon l'arc
(`docs/dataset_entrainement.md`, H1). Paris a ~3 000 capteurs. Des capteurs qui se comportent comme les nôtres
pourraient, plus tard, être ajoutés à l'entraînement pour apprendre l'effet des fériés sur plus d'exemples.

Cette étape ne fait que **choisir** ces capteurs. Rien n'est ajouté au dataset ni au modèle.

## Démarche et choix

### 1. Décrire chaque arc : profil de trafic d'abord, voirie en complément

| Source | Usage |
|---|---|
| Profil de trafic mesuré (débit et occupation moyens par heure × jour) | **cœur du score** (3 écarts sur 4) : c'est ce qu'on veut prédire, et il intègre déjà largeur, vitesse, rôle de l'axe |
| Voirie OpenStreetMap (type de route, voies, vitesse, couloir de bus) | **complément** : nombre de voies dans le score, bretelles et périphérique exclus, le reste affiché pour contrôle |
| Tracé (`referentiel-comptages-routiers`) | position, longueur, sens de circulation, distance ; **pas dans le score** : deux axes proches peuvent se comporter très différemment |

### 1 bis. OpenStreetMap

- Téléchargement : API Overpass, routes carrossables dans un rectangle autour de Paris (31 452 routes). Les serveurs
  publics sont souvent saturés (réponse « at capacity » au premier essai le 05/11/2026) ; un miroir renvoyait 0 route
  avec le contour administratif de Paris, d'où le rectangle. Le script essaie 3 serveurs, 3 fois, avec attente
  croissante.
- Remplissage sur les grands axes (primary, secondary, tertiary, trunk) : voies 79 %, vitesse 90 %, sens unique 72 %.
- Association capteur → route : segment OSM le plus proche du milieu de l'arc (≤ 30 m), à peu près parallèle
  (angle < 60°) et, pour un sens unique, orienté dans le sens de circulation du capteur. Le sens compte : les grands
  axes ont souvent une route OSM par chaussée. Le tracé du référentiel va bien de l'amont vers l'aval (4264
  Washington → Berri finit exactement où commence 4274 Berri → La Boétie).
- Résultat : 96 % des capteurs associés, à 4,5 m en médiane. Contrôle par le nom : 88 % des noms concordent. Les
  écarts sont surtout des renommages (Quai Anatole-France → Quai Valéry-Giscard-d'Estaing), des fautes du libellé
  (`Bd_de_Stasbourg`) ou des carrefours. Quand le nom ne concorde pas, le nombre de voies n'est pas utilisé.
- `voies_sens` = voies dans le sens du capteur : `lanes` pour un sens unique, `lanes:forward/backward` sinon, à
  défaut `lanes / 2`. Champs-Elysées : 4 voies par sens, 50 km/h ; Saint-Pères : 2 voies, sens unique, 30 km/h ;
  Convention : nombre de voies absent d'OSM (écart de voies neutre pour cet arc).

### 2. Profil sur 12 mois glissants

- 12 mois = toutes les saisons, tous les jours de la semaine, ~50 mesures par case heure × jour.
- Période récente : postérieure aux JO 2024 (rupture de niveau sur les Champs-Elysées) et identique pour tous les arcs.
- Calculé par l'API (agrégation côté serveur) : 1 requête au lieu de 28 millions de lignes. Contrôle : sur
  Convention, moyennes par jour identiques à nos données locales à 3 véh/h près (sur ~400), heure de Paris correcte.
- Heures « Barré » et « Invalide » exclues, comme pour l'entraînement (H2).
- **4264 (Champs-Elysées)** est muet depuis le 19/11/2025 : profil des 12 mois précédents (19/11/2024 → 18/11/2025),
  depuis les données locales. Couverture 89 %.

### 3. Filtres sur les candidats

| Filtre | Effet | Pourquoi |
|---|---|---|
| arcs ayant au moins une mesure sur 12 mois | 3 348 arcs au référentiel → 1 856 | les autres sont démontés ou muets |
| couverture ≥ 80 % des heures | 1 856 → 1 028 | un arc souvent en panne n'apportera pas d'historique utilisable. Médiane de couverture : 87 % ; à 90 % il n'en resterait que 805 |
| débit moyen ≥ 100 véh/h | 1 028 → 985 | nos arcs font 400 à 750 véh/h ; en dessous, ce sont des rues d'une autre nature |
| bretelles et périphérique (`motorway`, `*_link` dans OSM, ou « bretelle » dans le libellé) | ex. bretelle de la Porte des Sablons, 4e pour Convention avant ce filtre | comportement de voie rapide, pas comparable à nos avenues |
| copies du débit de l'arc du brief | Convention 5662, 5664, 5674 ; Saint-Pères 192 | voir ci-dessous |
| doublons entre candidats (débit moyen identique) | 128 arcs concernés, un seul gardé par groupe | même raison |
| au plus 2 tronçons par rue | sans ce filtre, 5 des 15 arcs des Champs-Elysées étaient sur le Bd de Sébastopol | des tronçons qui se suivent apportent peu d'information neuve |

**Un compteur de débit partagé.** Des tronçons qui se suivent sur une même rue publient souvent **le même débit**
(même boucle de comptage), avec chacun sa propre occupation. Sur les 12 derniers mois :

| Arc | Débit identique à l'arc du brief | Corrélation de l'occupation |
|---|---|---|
| Convention 5662 | 94,5 % des heures | 0,74 |
| Convention 5664 | 94,5 % des heures | 0,79 |
| Saint-Pères 192 | 99,7 % des heures | 0,75 |

Pour le débit, ces arcs ne sont donc pas de nouveaux exemples : ce sont des copies de notre cible. Règle : un arc
de la même rue dont le débit est identique sur au moins 90 % des heures (`SAME_COUNTER`) est écarté.
Ils ne servent pas non plus de secours pour 5671 ou 191 : s'ils sont identiques, ils tombent en panne en même temps
(non vérifié sur l'historique).

### 4. Score de ressemblance

Pour chaque arc du brief, quatre écarts avec chaque candidat :

| Écart | Calcul | Ce qu'il mesure |
|---|---|---|
| forme | écart quadratique moyen entre profils divisés par leur moyenne (24 h semaine + 24 h week-end) | pointes aux mêmes heures, même poids du week-end |
| niveau de débit | \|log(débit moyen candidat / débit moyen arc)\| | même volume de voitures |
| niveau d'occupation | idem sur l'occupation | même gabarit : à débit égal, plus d'occupation = voie plus étroite ou plus lente |
| voies | \|log(voies_sens candidat / voies_sens arc)\| (OSM) ; inconnu → écart moyen, donc neutre | même largeur de chaussée |

`score` = somme des quatre, chacun divisé par sa moyenne sur tous les candidats, pour qu'aucun ne domine à cause de
son unité. (La médiane ne marche pas pour les voies : souvent égales, leur écart médian vaut 0.) Plus petit = plus proche. On garde les 15 meilleurs.

**Pourquoi pas la forme seule ?** (chiffres calculés avant l'ajout d'OSM) Classés sur la forme seule, les 15 plus proches des Champs-Elysées incluent
Ledru-Rollin (212 véh/h) et Turbigo (332 véh/h), contre 742 véh/h pour les Champs : même rythme, mais pas le même
type d'axe. Les deux classements n'ont que 3 arcs sur 15 en commun pour les Champs, 5 pour Convention et 1 pour
Saint-Pères. Le niveau change donc nettement le résultat.

**Poids égaux** : aucun backtest ne permet encore de les régler. On le fera si les arcs similaires améliorent le
modèle (étape 4).

## Résultat

| Arc du brief | Les plus proches |
|---|---|
| Champs-Elysées (742 véh/h, occ. 15,6 %, 4 voies) | Bd de la Madeleine, Bd de Sébastopol, Av. Marceau, Champs-Elysées 4281 (aval, déjà proxy), Av. du Général-Leclerc, Quai des Tuileries, Av. de la Grande-Armée |
| Convention (403 véh/h, occ. 4,0 %) | Bd Brune, Quai d'Orsay, Bd des Invalides, Bd Auguste-Blanqui, Av. de Versailles, Bd Richard-Lenoir |
| Saint-Pères (451 véh/h, occ. 6,3 %, 2 voies) | Av. des Gobelins, Bd Raspail, Bd Brune, Av. Georges-Mandel, Quai André-Citroën, Bd des Capucines |

Colonnes du CSV : `score` et ses quatre écarts, `ecart_feries` et `percentile_feries` (vérification ci-dessous), infos OSM (`osm_nom`, `osm_highway`, `voies_sens`, `osm_vitesse`,
`osm_bus`), niveaux moyens, `ratio_weekend`, `couverture`, `distance_km`,
`meme_rue`, `date_debut` (mise en service de l'arc au référentiel).

## Vérification : réagissent-ils aux jours fériés comme nos arcs ?

C'est la raison pour laquelle on veut ces arcs : il faut donc vérifier qu'ils réagissent aux fériés comme les nôtres.
Le score ne l'utilise pas : c'est un contrôle indépendant.

- **Effet férié** d'un arc = débit moyen 7h-21h le jour férié / médiane du même jour de semaine 1 et 2 semaines
  avant et après (jours de référence non fériés). Jours comptés s'ils ont au moins 12 heures sur 14.
- Fériés de semaine des 12 derniers mois : 11/11/2025, 25/12, 01/01, lundi de Pâques, 01/05, 08/05, Ascension,
  lundi de Pentecôte, 14/07 (9 jours).
- `ecart_feries` = écart moyen entre l'effet férié du candidat et celui de l'arc du brief.
  `percentile_feries` = part des ~950 à 1 500 arcs comparables (mêmes filtres de couverture et de débit) qui font
  mieux : au hasard on attendrait **50 %** en moyenne.
- Champs-Elysées : 4264 est muet sur la période, on compare à la moyenne de ses proxys 4274 et 4281. Il ne reste que
  **4 fériés** : les autres sont des jours de fermeture (11/11, 08/05, 14/07 : heures « Barré ») ou tombent dans des
  trous des proxys.

| Arc du brief | Fériés comparés | Écart moyen des 15 similaires | Écart médian de tous les arcs | Percentile moyen |
|---|---|---|---|---|
| Convention | 9 | 0,093 | 0,112 | **34 %** |
| Saint-Pères | 9 | 0,105 | 0,122 | **35 %** |
| Champs-Elysées | 4 | 0,111 | 0,102 | 47 % |

Lecture :
- **Convention et Saint-Pères : oui, mais modestement.** Les arcs similaires réagissent aux fériés plus comme nos
  arcs que la moyenne : écart plus faible de 15 à 17 % et percentile moyen de 34-35 % contre 50 % au hasard.
  Sur 15 arcs, un écart de 15 points est environ deux fois ce qu'on attendrait par hasard : probablement réel.
  Mais ce n'est qu'un indice, sur 9 jours, et certains arcs similaires réagissent mal (Av. de Villars,
  Bd Malesherbes pour Convention ; Av. de Flandre pour Saint-Pères).
- **Champs-Elysées : non concluant.** 4 jours, surtout pendant les fêtes de fin d'année, et une référence
  reconstruite. Deux arcs réagissent très différemment (Av. du Général-Leclerc 712, Quai de Gesvres). Une vraie
  vérification demande l'historique 2021-2025 (étape 3), où 4264 a été mesuré.
- **Piste** : utiliser `percentile_feries` comme filtre (par exemple < 60 %) avant l'étape 4. Pas fait : avec 9
  jours, on risquerait de garder des arcs proches par hasard.

## Limites et suite

- **Poids non validés** : les quatre écarts pèsent autant faute de mieux. Les fériés confirment modestement le
  classement pour Convention et Saint-Pères ; le vrai test reste le backtest (étape 4).
- **Périodes différentes pour 4264** (2024-2025) et pour les autres arcs (2025-2026).
- **Historique non vérifié** : la couverture porte sur les 12 derniers mois. Les trous et les ruptures depuis 2021 se
  verront en téléchargeant les archives de ces arcs (étape 3).
- **Le 11/11 des Champs-Elysées est unique** (cérémonie, avenue fermée) : les arcs similaires peuvent aider pour
  l'effet férié, pas pour la fermeture.
- **Étape 4** : les ajouter à l'entraînement (modèle commun avec les caractéristiques des arcs en colonnes, ou lags des
  voisins), et ne garder que si le backtest s'améliore.
