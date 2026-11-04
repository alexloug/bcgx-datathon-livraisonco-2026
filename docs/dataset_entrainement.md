# Dataset d'entraînement

Construction : `src/data/build_dataset.py`. Justifications chiffrées : `notebooks/02_qualite_donnees.ipynb`
(qualité, ruptures) et `notebooks/03_construction_dataset.ipynb` (proxy, couverture, features).

```bash
python -m src.data.update --history 2021 2025     # sources (archives trafic utiles à partir de 2020)
python -m src.data.build_dataset                  # -> data/processed/dataset.parquet + dataset.json
```

```python
from src.data.build_dataset import build_dataset
df, meta = build_dataset()                                        # défauts de src/config.py
df, meta = build_dataset(start="2022-01-01 00:00", end="2026-11-11 23:00")
```

## Ce que contient le dataset

- **Une ligne par arc du brief et par heure**, du 21/06/2021 00:00 à la fin de la fenêtre à prédire
  (11/11/2026 23:00, heure de Paris). `datetime` en UTC.
- Les deux cibles `debit_horaire` (véh/h) et `taux_occupation` (%), nettoyées, NaN quand inconnues.
- `is_valid` : la ligne a ses deux cibles (mesurées ou reconstruites) et peut servir à l'entraînement.
- `source_cible` : `mesure`, `proxy` (Champs-Elysées reconstruit), `absent` (trou), `futur` (après la dernière
  heure observée).
- `partie` : `historique` (≤ dernière heure observée), `a_predire` (07/11 00:00 → 11/11 23:00), `non_predit`
  (le jour sauté entre les deux).
- `dataset.json` : période, dernière heure observée, nombre de lignes par arc et par origine, ratios du proxy.

Avec les données du 03/11/2026 : 141 843 lignes × 48 colonnes ; lignes valides : Champs-Elysées 42 807
(dont 7 554 reconstruites), Convention 30 374, Saint-Pères 35 240.

## Hypothèses et choix

### H1. Début au 21/06/2021

| Contrainte | Date |
|---|---|
| Prévisions météo archivées (nécessaires pour backtester avec l'information réellement disponible) | à partir du 01/01/2021 |
| Fin du couvre-feu Covid (dernière restriction affectant le trafic) | 20/06/2021 |
| Rupture de capacité des Champs-Elysées (débit ÷ 2, occupation × 2) | fin 2018 → mi-2019 |
| Recalibrage probable du capteur Convention (occupation ÷ 2 à débit égal) | 03/2020 |

Le 21/06/2021 est la première date qui satisfait toutes ces contraintes. Avant, le trafic n'est pas comparable à
aujourd'hui (notebook 02, parties 4-5). Les données de 2020-2021 sont quand même chargées pour que les lags (jusqu'à
52 semaines) existent dès la première ligne. Paramètre : `TRAIN_START` dans `src/config.py`.

Conséquence : le 11 novembre est rare dans les données valides (Champs-Elysées 4, Convention 1, Saint-Pères 5).
L'effet férié s'apprend donc surtout sur l'ensemble des jours fériés, d'où les features `ferie`, `veille_ferie`,
`lendemain_ferie`, `pont` et `lag52s`.

### H2. Nettoyage des cibles (D3, D4, D5)

Une heure est valide si **toutes** ces conditions sont réunies :
1. débit **et** occupation présents (on ne garde pas une cible seule : le couple doit rester cohérent) ;
2. état de l'arc différent de « Invalide » et « Barré » ;
3. pas de séquence figée : valeur identique non nulle pendant 6 h ou plus (`STUCK_HOURS`) ;
4. pas de pic : valeur ≤ 2 × quantile 99,9 % de l'arc (`SPIKE_FACTOR`).

Pas d'interpolation : les trous sont surtout des blocs de plusieurs semaines (voir les ruptures plus bas).
Les règles 3 et 4 n'enlèvent aucune heure depuis 2021, mais restent actives pour les mises à jour futures.

### H3. Champs-Elysées : proxy après l'arrêt du capteur (D1)

L'arc 4264 (Washington → Berri) est muet depuis le 19/11/2025. On reconstruit ses heures manquantes à partir des arcs
du même sens qui émettent encore :

| Candidat | Tronçon | Verdict |
|---|---|---|
| 4267 | Balzac → Washington (amont) | **écarté** : aucune mesure de 2021 à fin 2025, donc pas de recalage possible, et muet depuis 08/2026 |
| 4274 | Berri → La Boétie (aval immédiat) | **retenu** |
| 4281 | Colisée → Rond-Point (2 tronçons en aval) | **retenu** en complément |

Recalage : ratio médian 4264 / proxy par (heure locale × semaine/samedi/dimanche), appris sur toute la période commune
depuis 06/2021. Validation : apprentissage sur 06/2021 → 12/2024, test sur 01/2025 → 19/11/2025.

| Cible | 4274 | 4281 | Moyenne des deux | Retenu |
|---|---|---|---|---|
| Débit (MAE, véh/h) | **46** (6 %) | 72 | 51 | 4274, puis 4281 si 4274 manque |
| Occupation (MAE, points) | 5,5 | 4,5 | **4,1** (27 %) | moyenne, puis le proxy disponible |

- Une heure n'est reconstruite que si les deux cibles le sont.
- Apprendre le recalage seulement sur 2023+ ou 2024+ dégrade le résultat (moins d'heures, anomalie de 4281 en 2024).
- Réglage : `CHAMPS_PROXIES` dans `src/config.py`.
- L'occupation reconstruite est nettement moins fiable que le débit (corrélation 0,68 contre 0,97).

### H4. Features disponibles au moment de prédire

Règle : une feature de l'heure t n'utilise que des valeurs connues à t − 144 h ou avant (`MIN_LAG_HOURS`), vérifié
par `assert_no_leakage`. 144 h = 1 jour sauté + 5 jours prédits : **toutes les heures de la fenêtre ont leurs
features avec les seules données disponibles au cutoff**.

| Colonnes | Définition | Source |
|---|---|---|
| `hour`, `dow`, `month`, `is_weekend` | calendrier, heure de Paris | — |
| `ferie`, `veille_ferie`, `lendemain_ferie`, `pont` | pont = lundi avant un mardi férié, ou vendredi après un jeudi férié | calendrier.api.gouv.fr |
| `vacances` | vacances scolaires zone C (Paris) | data.education.gouv.fr |
| `vacances_type` | 0 hors vacances, 1 Toussaint, 2 Noël, 3 Hiver, 4 Printemps, 5 Été (ponts de l'Ascension exclus) | idem |
| `jours_avant_reprise` | en vacances : jours jusqu'à la reprise (creux de la mi-août, retour fin août) ; sinon -1 | idem |
| `jours_depuis_reprise`, `jours_avant_vacances` | hors vacances : jours depuis la dernière reprise / jusqu'aux prochaines vacances ; sinon -1. Plafonnés à 42 | idem |
| `vacances_niveau28j` | part d'heures de vacances dans la fenêtre de `<cible>_niveau28j` : corrige un niveau tiré vers le bas après des vacances (≈ 0,55 sur la fenêtre du 07-11/11) | idem |
| `heure_ete` | heure d'été en vigueur | — |
| `<cible>_lag168/336/504`, `<cible>_lag_mean` | même heure il y a 1, 2, 3 semaines, et leur moyenne, **pour les deux cibles** | trafic |
| `<cible>_niveau28j` | moyenne des 28 jours qui s'arrêtent 144 h avant t : suit les changements de niveau | trafic |
| `<cible>_lag52s` | même heure, même jour de semaine, 52 semaines avant | trafic |
| `meteo_*` | **prévision** : prévisions archivées Open-Meteo, puis prévision courante sur la fenêtre | Open-Meteo |
| `meteo_pluie` | `meteo_precipitation` > 0,1 mm/h | Open-Meteo |
| `meteo_neige_jour`, `meteo_neige_veille`, `meteo_tmin_jour` | neige prévue sur la journée et la veille (cm), température minimale prévue du jour | Open-Meteo (agrégats journaliers de la prévision) |
| `meteo_obs_*` | météo **observée** : pour l'analyse uniquement, **ne pas utiliser comme feature** (vide sur la fenêtre) | Open-Meteo |
| `evenement_fermeture`, `evenement_perturbation`, `evenement` | événement manuel en cours sur la rue de l'arc (journée entière, ou heures si renseignées) : fermeture (cérémonies, défilés, Champs piétons, marathon…), perturbation (grèves, salons), l'un ou l'autre | `data/evenements/evenements_manuels.csv` |
| `chantier_surface_arc` | log(1 + m²) des emprises sur chaussée à 30 m ou moins du segment de l'arc | chantiers à Paris (archives + courant) |
| `chantier_surface_150m` | idem à 150 m ou moins (carrefours, rues adjacentes) | idem |
| `chantier_voirie_150m` | idem, sans les chantiers de bâtiment : voirie et réseaux (eau, gaz, chauffage…) | idem |
| `chantier_nouveau_7j` | une emprise sur chaussée à 150 m ou moins a commencé il y a moins de 7 jours | idem |
| `jo_2024`, `post_jo_2024`, `post_30kmh` | indicateurs de régime | — |

Chantiers : les dates des emprises sont connues à l'avance (déclarées pour obtenir l'autorisation), donc
utilisables sur la fenêtre. Deux réserves. (1) Les archives donnent les dates finales, éventuellement prolongées,
alors qu'au moment de prédire on n'a que les dates prévues : petite fuite en backtest. (2) Du 01/01/2025 à la
veille du téléchargement, les emprises sont incomplètes (pas d'archive 2025) : les colonnes `chantier_*` y valent
NaN. **Les colonnes `chantier_*` ne sont pas utilisées par le modèle** (`EXCLUDED_PREFIXES` dans
`src/models/preprocessing.py`) : aucun gain en backtest (voir `docs/determinants_trafic.md`, §5).

Les valeurs reconstruites des Champs-Elysées alimentent aussi les lags : sans elles, l'arc n'aurait aucun lag pour la
fenêtre à prédire.

### H5. Backtest

Un seul dataset sert pour tous les backtests. Pour un cutoff c (jeudi 23:00) : entraîner sur les lignes
`is_valid` avec `datetime ≤ c`, prédire c + 25 h → c + 144 h, comparer aux cibles du dataset. Comme toutes les
features ont un décalage d'au moins 144 h, celles des heures prédites n'utilisent que des valeurs ≤ c.

Une seule fuite résiduelle, assumée : les ratios du proxy des Champs-Elysées sont appris sur toute la période
(jusqu'au 19/11/2025). Elle ne touche que les heures reconstruites (`source_cible == "proxy"`), qu'on peut exclure de
l'évaluation.

## Ruptures et anomalies dans la période retenue

| Date | Arc | Constat | Traitement |
|---|---|---|---|
| 30/08/2021 | tous | 30 km/h généralisé à Paris : **aucune rupture visible** sur débit ou occupation | `post_30kmh` |
| 05-06/2022 | Champs-Elysées | pic d'occupation en journée (42-48 % contre ~20 % habituellement), débit inchangé | conservé ; à surveiller |
| 11/2022 → 06/2023 | Convention | capteur absent pendant 7 mois | lignes non valides |
| 2022 → 06/2023 | proxy 4274 | aucune mesure (pas de recalage sur cette période) | sans effet : 4264 mesuré |
| fin 2022 | proxy 4281 | changement de niveau (ratio débit 4264/4281 de ~0,95 à ~1,25) | 4281 seulement en secours pour le débit |
| T2 2024 | proxy 4281 | débits anormalement bas (ratio ~16) | ratios médians robustes ; 4281 en secours |
| 2024 | Convention | trous répétés (mars-juin, juillet-août, août-décembre) | lignes non valides |
| 18/07 → 08/09/2024 | tous | JO et Jeux paralympiques (périmètres de sécurité) | `jo_2024` |
| après 09/2024 | Champs-Elysées | **baisse durable du débit** de journée (~1 200 → ~950 véh/h) | `post_jo_2024`, `niveau28j` |
| 2023 → 2026 | Convention | baisse progressive du débit (~800 → ~600 véh/h) | `niveau28j` |
| 11/2024 → 10/2025 | Saint-Pères | capteur absent pendant 11 mois | lignes non valides |
| 19/11/2025 | Champs-Elysées | arrêt du capteur 4264 | reconstruction par proxy (H3) |
| 08-09/2026 | proxy 4274 | trou de 6 semaines | 4281 prend le relais |

Ruptures antérieures (hors période, raison du choix H1) : Convention 03/2016 (saut de débit) et 03/2020 (recalibrage
probable), Champs-Elysées 2018-2019 (perte de capacité), Covid 03/2020 → 06/2021.

## Limites connues

- **Occupation des Champs-Elysées après le 19/11/2025** : reconstruite avec ~27 % d'erreur relative.
- **Peu de 11 novembre** dans la période (1 à 5 selon l'arc).
- **Prévisions archivées** : Open-Meteo les reconstitue à partir des premières heures de chaque prévision, pas des
  prévisions à J+2…J+6 qu'on aura pour le rendu. L'écart réel sera plus grand en production qu'en backtest.
- **Événements** : les événements manuels ne listent que les dates fixes ; les exceptions (défilé annulé…)
  ne sont pas saisies.
