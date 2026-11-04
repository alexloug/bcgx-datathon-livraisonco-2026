# 1. Le problème : ce qu'on prédit, comment on est noté, de quoi dépend le trafic

## 1. Ce que dit le brief

Source : deck `2026 - Brief - LivraisonCo.pptx` (29/10/2026).

| Point | Ce que dit le brief | Conséquence dans le dépôt |
|---|---|---|
| Cibles | débit horaire et taux d'occupation de la voie | 2 cibles × 3 arcs |
| Arcs | Champs-Elysées (Washington → Berri), Convention (Lecourbe → Blomet), Saint-Pères (Voltaire → Université), filtrés par nœuds | `src/config.py` |
| Fenêtre | 360 lignes = 3 arcs × 120 heures, **samedi 07/11 00:00 → mercredi 11/11 23:00** | « 5 jours à partir de vendredi » confond la date limite et la fenêtre |
| Date limite | envoi avant le **vendredi 06/11, 18 h** | |
| Données | « dernières données disponibles » ; l'API publie en J-1 | téléchargées le 06/11, dernière heure connue le 05/11 23:00 |
| Métrique | RMSE des 2 variables sur les 3 axes, **toutes les heures** | le backtest compte les 24 h (il comptait 6 h-20 h au début) |
| Note | Data Science /5 (RMSE + présentation), Business /5 (solution d'optimisation + présentation) | égalité départagée par la RMSE |
| Format | `output_<groupe>.csv`, `sep=";"`, colonnes `arc`, `datetime` (`%Y-%m-%d %H:%M`), `debit_horaire`, `taux_occupation` | `check_submission` reprend le test du brief |
| Business | livreurs à 10 €/h de jour (8 h-20 h), 14 €/h de nuit, créneaux de 4 à 8 h | la nuit est une option, pas une heure à ignorer |

**Décalage imposé.** Dernière donnée au 05/11 23:00, première heure prédite le 07/11 00:00, dernière le 11/11 23:00 :
144 h d'écart maximum. Toute feature de l'heure t doit donc n'utiliser que des valeurs connues 144 h avant ou plus
(`MIN_LAG_HOURS`). Les backtests reproduisent ce découpage : cutoff le jeudi 23:00, 1 jour sauté, 5 jours prédits.

**Questions posées au tutorat du 05/11** : comment les 6 RMSE sont agrégées (en somme brute, le débit de 60 à 170 véh/h
écrase l'occupation de 2 à 10 points, et les Champs pèsent 2 à 3 fois plus que les autres arcs) ; toutes les heures
comptent-elles ; sur quelle vérité les Champs sont-ils notés (capteur muet).

Effet de la fenêtre horaire (backtest XGBoost, 52 jeudis de 2023 et 18 de 2026) : la RMSE sur 24 h est jusqu'à 19 %
plus basse qu'en 6 h-20 h (la nuit est facile). Le classement des arcs ne change pas.

## 2. Lecture business

Le client choisit **quand** envoyer ses livreurs. Ce qui compte : le temps de parcours par créneau (donc la
congestion, l'occupation plus que le débit), les journées anormales qui font rater une tournée, l'incertitude.

**Indice de temps de parcours.** Densité ∝ occupation o, vitesse ∝ q / o : le temps de parcours d'un arc est
proportionnel à o / q. Rapporté à sa valeur de nuit (1 h-4 h, trafic libre) : 1 = fluide, 2 = deux fois plus long.
Limite : suppose une longueur moyenne de véhicule stable ; bruité sous 30 véh/h. Indice médian, jours ouvrés 2021-2025 :

| Heure | 7 h | 8 h | 9 h | 11 h | 13 h | 14 h | 17 h | 18 h | 19 h | 21 h |
|---|---|---|---|---|---|---|---|---|---|---|
| Champs-Elysées | 0,85 | 0,89 | 0,96 | 1,31 | 1,78 | 1,84 | 1,77 | 1,78 | 1,72 | 1,56 |
| Convention | 1,26 | 1,34 | 1,48 | 1,39 | 1,29 | 1,28 | 1,62 | 2,20 | 1,77 | 1,20 |
| Saint-Pères | 0,99 | 1,09 | 1,08 | 1,21 | 1,23 | 1,21 | 1,20 | 1,12 | 1,03 | 1,02 |

Aux heures de pointe, l'indice P90 vaut 2,3 sur les Champs et 4,3 sur Convention : c'est la queue de distribution qui
fait rater un créneau. Les seuils `etat_trafic` de la Ville sont des seuils d'occupation (fluide < 15 %, pré-saturé
15-30 %, saturé 30-50 %, bloqué > 50 %) : on peut livrer des **classes de trafic** par créneau. Ces indicateurs se
calculent après la prédiction ; dans le dataset, ils fuiraient la cible.

**Ce que le modèle rate** (backtest XGBoost, 6 h-20 h) :

| | Champs-Elysées | Convention | Saint-Pères |
|---|---|---|---|
| Heures en congestion (occupation > 15 %) | 48-69 % | 2-9 % | 3-4 % |
| Part de ces heures prédites congestionnées | 92-97 % | 0-59 % | 0-3 % |
| Classe de trafic exacte | 71 % | 91-98 % | 95-97 % |
| Part de l'erreur² d'occupation portée par les 10 % pires jours | 55-65 % | 47-71 % | 48-60 % |

Le modèle prédit bien la moyenne, mal l'exception : sur Convention et Saint-Pères, les rares heures congestionnées ne
sont presque jamais prévues, et **la moitié à deux tiers de l'erreur vient de 10 % des jours** (travaux, déviations,
événements).

## 3. Les trois arcs

| | Champs-Elysées | Convention | Saint-Pères |
|---|---|---|---|
| Rôle | axe touristique et commercial, à 400 m de l'Étoile | axe de quartier du 15e, pendulaire | rue à sens unique alimentant la rive gauche depuis le quai |
| Débit max (plateau) | 1 150 véh/h | 930 véh/h | 730 véh/h |
| Occupation médiane de jour | 20 % | 4,5 % | 7,4 % |
| Heures de jour pré-saturées ou pire | **67 %** | 7 % | 4 % |
| Corrélation débit-occupation de jour | 0,55 | 0,58 | 0,81 |
| Perturbations | cérémonies, défilés, Champs piétons, réaménagement | travaux, déviations, courses, neige | quai Voltaire, déviations, ZTL, neige |

Les arcs diffèrent par leur régime plus que par leur taille : les Champs vivent sur le plateau du diagramme
fondamental (débit stable, occupation très variable), Saint-Pères est presque toujours fluide. Normaliser par le nombre
de voies ne marche pas (380 véh/h/voie sur les Champs contre 730 sur Saint-Pères : la capacité vient des feux).

Les anomalies journalières sont **locales à la rue** : corrélation avec la moyenne de la ville 0,07-0,28, avec les
autres arcs de la même rue 0,35-0,74, avec la même heure une semaine plus tôt 0,18-0,49 (déjà captée par les lags).

## 4. De quoi dépend le trafic

Un profil heure × jour × férié × vacances × niveau des 28 derniers jours explique le débit (R² 0,76-0,90) mais
seulement 0,52-0,67 de l'occupation : **un tiers à la moitié de l'occupation n'est expliquée par rien de ce qu'on a**.
Causes des plus gros écarts (7 h-21 h), débit / occupation :

| Cause | Effet mesuré | Dans le modèle ? |
|---|---|---|
| Travaux et déviations (Saint-Pères 28/08 → 20/09/2026, Convention 09/2022, Champs 09/2024) | +45 à +85 % / +100 à +245 % | **non** : emprises de chantier testées, sans gain (voir 3_modele.md) |
| Rentrée et fins de vacances | +40 à +60 % / +60 à +150 % | oui : calendrier scolaire |
| Événements Champs (JO 2024 -55 à -62 %, illuminations, journée sans voiture, dimanches piétons) | fort, quelques jours par an | oui : événements saisis à la main |
| Neige, verglas | -13 à -30 % / -22 à -38 % | oui : neige prévue |
| Courses (marathon : Convention -16 %, Saint-Pères -27 %) | moyen, 1-3 jours par an | partiel |
| Grèves, journées de mobilisation | faible sur le débit, -12 à -23 % sur l'occupation | partiel |
| Pluie > 0,5 mm/h | débit -1 à -2 % ; occupation +14 % Convention, +6 % Saint-Pères | oui : intensité horaire, prévue |

**Changements de régime** à connaître : quai Voltaire réduit à une voie (10/2024 → 30/10/2026, Saint-Pères) ; ZTL Paris
Centre (04/11/2024, Saint-Pères, à vérifier) ; aménagement de voirie à ~110 m de l'arc des Champs (07 → 12/2026) ;
réseau de froid rue de l'Université (19/10 → 18/12/2026, Saint-Pères).

## 5. La fenêtre du 07 au 11/11/2026

| Jour | Ce qui le distingue |
|---|---|
| sam. 07/11 | samedi ordinaire, 2e semaine après la reprise de la Toussaint |
| dim. 08/11 | 2e dimanche : pas de Champs piétons |
| lun. 09/11, mar. 10/11 | jours ouvrés ; le 10 est une veille de férié |
| mer. 11/11 | **férié** + cérémonie à l'Arc de Triomphe (haut des Champs fermé le matin) |

Risques connus : (1) Champs : l'occupation est reconstruite par proxy, c'est la prédiction la moins sûre des six ;
(2) Saint-Pères : le niveau des 28 derniers jours est gonflé par la déviation de septembre et la fin du chantier du
quai Voltaire (30/10) change l'amont de l'arc ; depuis le 24/10, l'arc est 15 à 30 % sous le modèle ; (3) le niveau
28 j couvre les vacances de la Toussaint (corrigé par `vacances_niveau28j`, ≈ 0,55 sur la fenêtre).

## 6. Pistes classées (impact × effort)

| Piste | Résultat |
|---|---|
| Mesurer en 24 h comme la note | fait : `evaluate` donne la RMSE sur les 24 h |
| Traduire en classes de trafic et indice de temps de parcours par créneau | chiffré ci-dessus, pour la présentation Business |
| Modèle partagé entre arcs de la même rue | seul gain stable : occupation des Champs -6 à -9 % (3 cutoffs sur 4), mais avec un modèle simplifié ; non adopté (voir 3_modele.md) |
| Traiter les jours anormaux (arrêtés de circulation, intervalles de confiance) | gain potentiel élevé, mais donnée absente de l'open data ; veille manuelle de la fenêtre |
| Cible relative au niveau 28 j | rejeté : Champs débit 169 contre 165, Saint-Pères 109 contre 96 |
| Corriger du biais de la semaine précédente | rejeté : le biais ne persiste pas (corrélation -0,45 à +0,54) |

Le gain le plus grand viendrait d'un flux temps réel (J-1 au lieu de J-6) : l'anomalie d'un jour est corrélée à
0,45-0,79 avec celle de la veille, contre 0,2-0,5 avec celle de la semaine précédente.
