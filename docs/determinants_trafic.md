# De quoi dépend le trafic sur nos 3 arcs ?

Objectif : vérifier qu'on n'a pas oublié de source externe essentielle, en partant de l'usage final (prédire le
trafic pour optimiser des tournées de livraison) plutôt que de la liste des données disponibles.

Méthode (analyse du 04/11/2026, script hors dépôt) : sur les heures **mesurées** depuis 06/2021, on modélise chaque
cible par un profil heure × jour × férié × vacances, multiplié par le niveau des 28 derniers jours. Puis on regarde
les **jours où l'écart (7 h-21 h) est le plus grand** et on cherche leur cause. Champs-Elysées : seulement jusqu'au
19/11/2025 (après, la cible est reconstruite).

## 1. Ce qui compte pour le business

- Une tournée dépend des **temps de parcours**, donc de la congestion : c'est le **taux d'occupation** qui
  la mesure le mieux, pas le débit (à débit égal, une rue peut être fluide ou saturée).
- Les heures qui comptent sont les **heures de livraison** (à confirmer avec le brief : en gros 6 h-20 h, jours
  ouvrés et samedi). La nuit pèse peu.
- Ce qui coûte cher, ce n'est pas le bruit d'une heure à l'autre, ce sont les **journées anormales** : une rue
  barrée ou déviée qu'on n'a pas prévue fait rater une tournée entière.

Or c'est exactement là qu'on est faible :

| Arc | Débit : R² profil + niveau | Occupation : R² profil + niveau | Occupation : R² naïf S-1 |
|---|---|---|---|
| Champs-Elysées | 0,76 | **0,52** | 0,27 |
| Convention | 0,90 | **0,63** | 0,27 |
| Saint-Pères | 0,85 | **0,67** | 0,50 |

Le calendrier explique l'essentiel du débit. **Un tiers à la moitié de la variance de l'occupation n'est expliquée
par rien de ce qu'on a** : c'est là qu'il faut chercher des sources.

## 2. Ce que fait chaque axe

| Arc | Rôle dans le réseau | Ce qui le perturbe |
|---|---|---|
| Champs-Elysées (4264, Washington → Berri, sens Étoile → Concorde) | axe de prestige, touristique et commercial, à 400 m de l'Étoile | cérémonies à l'Arc de Triomphe, défilés, fermetures piétonnes, illuminations, manifestations et célébrations, visites d'État, **réaménagement de l'avenue** |
| Convention (5671, Lecourbe → Blomet) | axe de quartier du 15e, trafic pendulaire et local, marché Convention (mar., jeu., dim. matin) | travaux et déviations, courses (marathon), neige |
| Saint-Pères (191, quai Voltaire → Université) | rue en sens unique qui alimente la rive gauche depuis le quai et le pont du Carrousel | **ce qui se passe sur le quai Voltaire et les quais**, déviations, ZTL Paris Centre, courses, neige |

## 3. Ce que montrent les jours anormaux

Écart moyen au modèle (7 h-21 h) : débit / occupation.

| Cause | Exemples | Effet mesuré | Dans nos données ? |
|---|---|---|---|
| **Travaux et déviations** (jours ou semaines) | Saint-Pères 28/08 → 20/09/2026 ; Saint-Pères 04-12/05/2023 ; Convention 05-12/09/2022 ; Champs 19-30/09/2024 | **+45 à +85 %** / **+100 à +245 %** | features `chantier_*` dans le dataset, sans gain (§5) |
| **Rentrée et fins de vacances** | fin août → mi-septembre chaque année, 3 arcs | +40 à +60 % / +60 à +150 % | **mal** : `vacances` traite l'été comme un bloc, le niveau 28 j est en retard |
| Événements Champs | JO 2024 (-55 à -62 %), illuminations (-49 % / +36 %), journée sans voiture (-39 %), 1er dimanche piéton (-11 % en moyenne, jusqu'à -92 %) | fort, quelques jours par an | partiel (4 dates fixes) |
| Neige, verglas | 05-08/01/2026 (Convention, Saint-Pères), 17-18/01/2024 | -13 à -30 % / -22 à -38 % | à moitié (prévision de neige, pas d'indicateur) |
| Courses | marathon (Convention -16 %, Saint-Pères -27 %), arrivée du Tour (Saint-Pères -22 %) | moyen, 1-3 jours par an | non |
| Grèves, journées de mobilisation | retraites 2023 (Champs -6 %, Saint-Pères -11 %) ; septembre 2025 (-8 %) | faible sur le débit, -12 à -23 % sur l'occupation | non |
| Pluie (> 0,5 mm/h) | — | débit -1 à -2 % ; occupation **+14 % Convention**, +6 % Saint-Pères | oui |

Lecture :
1. **Le premier facteur oublié, ce sont les travaux et déviations.** Ce sont les plus gros écarts, ils durent des
   semaines, et ils touchent surtout l'occupation.
2. Le deuxième n'est pas une source externe mais une feature : **la position dans le calendrier scolaire**
   (semaine de reprise, première semaine de septembre), mal captée aujourd'hui.
3. La météo n'a d'effet que dans les cas extrêmes (neige) et sur l'occupation.
4. Les autres causes (événements, courses, grèves) sont rares et datées : **une saisie manuelle suffit**.

## 4. Changements de régime à connaître (sources réglementaires)

| Date | Arc | Changement | Conséquence |
|---|---|---|---|
| 10/2024 → **30/10/2026** | Saint-Pères | quai Voltaire réduit à une voie (chantiers Orsay / Conseil d'État, `chantiers_perturbants`) | **fin du chantier 8 jours avant la fenêtre** : l'amont de l'arc change |
| 04/11/2024 | Saint-Pères | ZTL Paris Centre (zone à trafic limité) ; l'arc est entre la Seine et le bd Saint-Germain, a priori dans le périmètre (**à vérifier**) | coïncide avec le trou du capteur 11/2024 → 10/2025 : le niveau d'avant n'est plus comparable |
| 07/2026 → 18/12/2026 | Champs-Elysées | « aménagement de voirie » Ville de Paris, 1 086 m² sur chaussée, à ~110 m de l'arc 4264 (côté nord, rue adjacente plutôt que l'avenue) | à surveiller : effet possible sur le ratio 4264 / 4274 du proxy |
| 19/10 → 18/12/2026 | Saint-Pères | réseau de froid rue de l'Université (nœud aval), circulation restreinte | possible remontée de file sur l'arc |

Depuis le 24/10/2026, Saint-Pères est 15 à 30 % sous le modèle : probablement l'effet d'un niveau 28 j gonflé par
la déviation de septembre, plus la fin du chantier du quai. À regarder de près avant le rendu.

## 5. Sources à ajouter, par priorité

| # | Source | Pourquoi | Comment | Utilisable |
|---|---|---|---|---|
| 1 | **Chantiers sur et autour des arcs** | plus gros écarts | **fait le 04/11, sans gain** : voir l'encadré ci-dessous | dataset oui, modèle non |
| 2 | **Calendrier des événements Champs** | effets forts, dates connues à l'avance | compléter `evenements_manuels.csv` : dimanches piétons, journée sans voiture, illuminations, Tour de France, JO, parades, visites d'État | oui |
| 3 | **Courses et grands événements sportifs** | marathon, semi, 20 km, Tour | saisie manuelle (10 dates par an) | oui |
| 4 | **Neige / vigilance météo** | -30 % sur le débit | indicateur neige ou vigilance Météo-France (open data) | oui |
| 5 | **Grèves et manifestations** | effet modéré mais réel sur l'occupation | saisie manuelle des grandes journées ; **vérifier l'actualité pour 07-11/11** | oui, si annoncé |
| 6 | Régimes réglementaires (ZTL, quai Voltaire, réaménagement Champs) | ruptures de niveau | indicateurs de régime, comme `post_jo_2024` | oui |

**Chantiers : ce qu'on a fait et ce que ça donne (04/11).**
- Source : emprises de chantier de la Ville, archives annuelles 2019-2024 (identifiants `chantiers-a-paris-copie*`)
  et jeu courant. Trou de 01/2025 à la veille du téléchargement (pas d'archive 2025) : features à NaN.
- Features (`src/data/build_dataset.py`) : surface d'emprise sur chaussée à ≤ 30 m et ≤ 150 m de l'arc, la même
  sans les chantiers de bâtiment, et « chantier commencé il y a moins de 7 jours ».
- Les épisodes coïncident bien avec des emprises : chauffage urbain sur l'arc Convention le 05/09/2022,
  aménagement de voirie à 12-19 m de l'arc Champs en mai-juin 2022 et en septembre 2023.
- **Mais aucun gain en backtest XGBoost** (39 jeudis de 2022 à 2024, 6 h-20 h). RMSE avec / sans : débit 193 / 193
  (Champs), 91 / 89 (Convention), 79 / 78 (Saint-Pères) ; occupation 9,4 / 9,6, 4,7 / 4,6, 2,5 / 2,4. Les
  chantiers gagnent un fold sur deux. Sur les épisodes ciblés : gain ponctuel (Champs, mai 2022 : occupation 23 → 21 ;
  septembre 2023 : débit 276 → 252), pertes ailleurs.
- Pourquoi : la moitié des jours ont une emprise sur chaussée à moins de 150 m ; une emprise ne dit pas si une voie
  est fermée ni où part la déviation ; la déviation de Saint-Pères en septembre 2026 ne correspond à aucune emprise
  proche.
- Décision : les colonnes restent dans le dataset (analyse, contrôle de la fenêtre), **le modèle ne les utilise
  pas** (`EXCLUDED_PREFIXES`). Sur la fenêtre du 07-11/11, aucune emprise de voirie ou de réseau sur chaussée à
  moins de 150 m des 3 arcs : rien à corriger à la main.
- Ce qui aiderait vraiment : les fermetures et déviations elles-mêmes (arrêtés de circulation), qu'on n'a pas en
  open data.

Côté features (pas de source nouvelle) :
- **semaine de reprise** après vacances et position dans les grandes vacances (début / milieu / fin d'août).

## 6. La fenêtre du 07 au 11/11/2026

| Jour | Ce qui le distingue |
|---|---|
| sam. 07/11 | samedi ordinaire, deuxième semaine après la reprise de la Toussaint (02/11) |
| dim. 08/11 | 2e dimanche : pas de Champs piétons |
| lun. 09/11, mar. 10/11 | jours ouvrés ; le 10 est une veille de férié (départs en soirée) |
| mer. 11/11 | **férié** + cérémonie à l'Arc de Triomphe : haut des Champs fermé le matin, à 400 m de l'arc |

Trois risques à traiter avant le rendu du 06/11 :
1. **Champs-Elysées** : un aménagement de voirie à ~110 m de l'arc depuis juillet peut rendre le proxy incertain.
   Comparer 4274 et 4281 avant / après juillet 2026 pour voir si l'écart entre les deux proxys a bougé.
2. **Saint-Pères** : fin du chantier du quai Voltaire le 30/10 et chantier rue de l'Université : le niveau des 28
   derniers jours ne décrit sans doute pas la semaine prédite.
3. **Le niveau 28 j** de la fenêtre couvre ~04/10 → 01/11, dont les vacances de la Toussaint : il est biaisé vers
   le bas.
