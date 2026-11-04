# 2. Les données : sources, nettoyage, features, dataset livré

Construction : `src/data/build_dataset.py` (dataset complet) puis `src/data/export_dataset.py` (dataset livré au modèle,
contrôlé). Tout se lance avec `python -m src.predict --group <nom> --refresh`.

```bash
python -m src.data.update                 # sources open data (--history 2020 2025 pour les archives trafic)
python -m src.data.build_dataset          # -> data/processed/dataset.parquet (+ .json)
python -m src.data.export_dataset         # -> data/processed/dataset_modele.parquet (+ .json), contrôles bloquants
```

## 1. Le dataset livré

`dataset_modele.parquet` : une ligne par arc et par heure, du 21/06/2021 au 11/11/2026 23:00 (heure de Paris),
2 cibles et **44 features figées**. 141 843 lignes × 51 colonnes avec les données au 03/11/2026. `datetime` est en UTC ;
les features calendaires sont en heure de Paris.

| Groupe | Colonnes | Rôle |
|---|---|---|
| Clés | `arc`, `datetime` | `arc` = `Champs-Elysées`, `Convention` ou `Saint-Pères` |
| Cibles | `debit_horaire` (véh/h), `taux_occupation` (%) | NaN quand inconnues |
| Suivi | `is_valid`, `source_cible`, `partie` | quelles lignes entraîner, évaluer, prédire |
| Features | 44 | 6 familles, voir §4 |

| `partie` | `source_cible` | Lignes | Sens |
|---|---|---|---|
| `historique` | `mesure` | 100 915 | cible mesurée |
| `historique` | `proxy` | 7 571 | Champs-Elysées après l'arrêt du capteur (19/11/2025), reconstruit |
| `historique` | `absent` | 32 784 | trou de capteur ou heure invalide : cible NaN, features présentes |
| `non_predit` / `a_predire` | `futur` | 573 | après la dernière heure observée ; `a_predire` = les 360 lignes du rendu |

`export_dataset` refuse d'écrire si un contrôle échoue : doublons, grille horaire trouée, arcs manquants, 360 lignes à
prédire, cibles présentes sur la fenêtre, cible hors bornes, feature exclue présente.

**Règles d'utilisation.**
1. Entraîner sur `is_valid` et `datetime <= cutoff`. `is_valid` = les deux cibles présentes (mesurées ou reconstruites),
   hors heures « Invalide », « Barré », figées ou aberrantes.
2. Prédire `partie == "a_predire"`.
3. **Pas de fuite, sans rien recalculer** : toute feature de l'heure t n'utilise que des valeurs connues 144 h avant ou
   plus. Un seul fichier sert à tous les backtests. Ne jamais ajouter de lag plus court que 144 h (`assert_no_leakage`).
4. Les NaN des features historiques sont normaux (trous de capteur) ; le prétraitement les remplit sans fuite.
5. Champs-Elysées après le 19/11/2025 : `source_cible == "proxy"`. Débit fiable (6 % d'erreur), **occupation beaucoup
   moins (≈ 27 %)** : c'est la prédiction la moins sûre des six.

Volontairement absents : les chantiers et la météo observée (aucun gain / inconnue à la prédiction), et toute colonne
calculée à partir de la cible de l'heure t (classe de trafic, indice de temps de parcours : fuite).

## 2. Choix de construction

### Début au 21/06/2021 (pas avant)

| Contrainte | Date |
|---|---|
| Prévisions météo archivées (pour backtester avec l'information réellement disponible) | à partir du 01/01/2021 |
| Fin du couvre-feu Covid | 20/06/2021 |
| Rupture de capacité des Champs-Elysées (débit ÷ 2, occupation × 2) | fin 2018 → mi-2019 |
| Recalibrage probable du capteur Convention (occupation ÷ 2 à débit égal) | 03/2020 |

Le 21/06/2021 est la première date qui les satisfait toutes. Les données antérieures sont chargées pour que les lags
(jusqu'à 52 semaines) existent dès la première ligne. Conséquence : peu de 11 novembre valides (Champs 4,
Convention 1, Saint-Pères 5) ; l'effet férié s'apprend sur tous les fériés (`ferie`, `veille_ferie`, `lendemain_ferie`,
`pont`, `lag52s`).

### Nettoyage, sans interpolation

Une heure est valide si : débit **et** occupation présents (le couple doit rester cohérent) ; état de l'arc différent de
« Invalide » et « Barré » ; pas de séquence figée (valeur identique non nulle ≥ 6 h) ; pas de pic (> 2 × quantile
99,9 % de l'arc). **Pas d'interpolation** : les trous sont des blocs de plusieurs semaines (Convention 11/2022 → 06/2023,
Saint-Pères 11/2024 → 10/2025), qu'aucune interpolation ne rendrait crédibles.

### Champs-Elysées : le capteur 4264 est muet depuis le 19/11/2025

Reconstruction par les arcs du même sens qui émettent encore, recalés par le ratio médian 4264 / proxy par (heure ×
semaine/samedi/dimanche). Validation : apprentissage 06/2021 → 12/2024, test 01/2025 → 19/11/2025.

| Candidat | Verdict | Débit (MAE) | Occupation (MAE) |
|---|---|---|---|
| 4267 (Balzac → Washington, amont) | écarté : aucune mesure 2021-2025 | | |
| 4274 (Berri → La Boétie) | retenu pour le débit | **46 véh/h (6 %)** | 5,5 points |
| 4281 (Colisée → Rond-Point) | secours | 72 | 4,5 |
| Moyenne 4274 + 4281 | retenu pour l'occupation | 51 | **4,1 points (27 %)** |

Une heure n'est reconstruite que si les deux cibles le sont ; ces valeurs alimentent aussi les lags (sans elles, l'arc
n'aurait aucun lag sur la fenêtre). Apprendre le recalage sur 2023+ ou 2024+ dégrade le résultat. Réglage :
`CHAMPS_PROXIES` dans `src/config.py`. Sur l'occupation, la corrélation avec le capteur n'est que de 0,68 (0,97 pour
le débit), et le ratio mensuel 4264 / 4274 varie de 1,7 à 2,8.

### Lags en heure de Paris

Un lag « n jours avant » vise la **même heure locale**, pas un décalage de n × 24 lignes sur la grille UTC. Avec l'ancien
décalage, après le passage à l'heure d'hiver du 25/10/2026, le 07/11 08:00 visait le 17/10 à 09:00 (valeur 322 au lieu
de 237). `local_lag` (`src/features/build.py`) cherche l'heure locale ; heure en double (automne) = moyenne, heure
inexistante (printemps) = NaN. `level_features` reste en décalage réel de 144 h. `assert_no_leakage` exige des jours
entiers et `lag - 1 >= 144`. Vérifié sur la fenêtre du rendu : 0 écart entre `lag168`, `lag336`, `lag504` et la valeur
à l'heure locale attendue. Effet sur la RMSE : dans le bruit (voir 3_modele.md) ; la correction est retenue parce
qu'elle est exacte.

### Une fuite résiduelle, assumée

Les ratios du proxy des Champs sont appris sur toute la période (jusqu'au 19/11/2025). Elle ne touche que les heures
`source_cible == "proxy"`, qu'on peut exclure de l'évaluation.

## 3. Ruptures et anomalies dans la période

| Date | Arc | Constat | Traitement |
|---|---|---|---|
| 30/08/2021 | tous | 30 km/h généralisé : aucune rupture visible | `post_30kmh` |
| 11/2022 → 06/2023 | Convention | capteur absent 7 mois | lignes non valides |
| 2024 | Convention | trous répétés | lignes non valides |
| 18/07 → 08/09/2024 | tous | JO et Jeux paralympiques | `jo_2024` |
| après 09/2024 | Champs-Elysées | baisse durable du débit de journée (~1 200 → ~950 véh/h) | `post_jo_2024`, `niveau28j` |
| 2023 → 2026 | Convention | baisse progressive du débit (~800 → ~600 véh/h) | `niveau28j` |
| 11/2024 → 10/2025 | Saint-Pères | capteur absent 11 mois | lignes non valides |
| 19/11/2025 | Champs-Elysées | arrêt du capteur 4264 | proxy |
| 08-09/2026 | proxy 4274 | trou de 6 semaines | 4281 prend le relais |

Avant 2021 (hors période, raison du début au 21/06/2021) : Convention 03/2016 et 03/2020, Champs 2018-2019, Covid.

## 4. Les 44 features

Trois règles : rien d'inconnu au moment de prédire ; exogène = connu à l'avance ou **prévu** (jamais la météo observée) ;
heure de Paris pour tout le calendaire.

| Famille | Cols | Définition | Où |
|---|---|---|---|
| Calendrier | 9 | `hour`, `dow`, `month`, `is_weekend`, `ferie`, `veille_ferie`, `lendemain_ferie`, `pont` (lundi avant un mardi férié, vendredi après un jeudi férié), `heure_ete` | `features/build.py`, `calendar_extra` |
| Calendrier scolaire | 6 | `vacances` (zone C), `vacances_type` (0-5), `jours_avant_reprise`, `jours_depuis_reprise`, `jours_avant_vacances` (plafonnés à 42, sinon -1), `vacances_niveau28j` | `school_calendar` |
| Historique | 12 | par cible : `lag168/336/504` (S-1 à S-3, même heure locale), `lag_mean`, `niveau28j` (28 jours s'arrêtant 144 h avant t, ≥ 7 jours), `lag52s` | `features/build.py` |
| Météo prévue | 11 | `meteo_temperature_2m`, `_precipitation`, `_rain`, `_snowfall`, `_cloud_cover`, `_wind_speed_10m`, `_weather_code`, `meteo_pluie` (> 0,1 mm/h), `meteo_neige_jour`, `meteo_neige_veille`, `meteo_tmin_jour` | `weather` |
| Événements | 3 | `evenement_fermeture`, `evenement_perturbation`, `evenement` | `event_flags` |
| Régimes | 3 | `jo_2024`, `post_jo_2024` (≥ 09/09/2024), `post_30kmh` (≥ 30/08/2021) | `regime_flags` |

Le préprocesseur du modèle ajoute 4 colonnes (3 indicatrices d'arc, `nb_lags_manquants`), soit 48 features reçues.
Colonnes exclues du modèle : identifiants, cibles, `meteo_obs_*` (observée, vide sur la fenêtre), `chantier_*`.

**Effets mesurés** qui justifient les familles :
- *Calendrier* : le profil heure × type de jour explique 66 à 81 % de la variance du débit.
- *Calendrier scolaire* : mi-août -12 à -30 % de débit, dernière semaine avant la rentrée +17 à +43 % ; 1re semaine après
  la Toussaint : Convention -20 % d'occupation ; après Noël jusqu'à -36 %. `vacances_niveau28j` corrige un niveau de
  référence tiré vers le bas après des vacances. Backtest, RMSE débit avant → après : fin d'été Convention 69 → 55,
  Saint-Pères 75 → 69, Champs 181 → 175 ; neutre sur les semaines ordinaires (1 à 3 %).
- *Historique* : les lags apportent le niveau récent et les ruptures de chaque arc. Seules les heures valides les
  alimentent.
- *Météo* : l'effet passe par l'**intensité horaire** de pluie (au-delà de 1 mm/h : occupation Convention +19 à +23 %,
  Saint-Pères +8 à +10 %) ; le cumul du jour n'explique rien. La neige horaire est presque toujours nulle même un jour
  de neige, d'où des agrégats journaliers : ≥ 0,5 cm prévu (9 jours depuis 2021) = débit -13 à -16 %, occupation -11 à
  -40 %, effet encore net le lendemain. Backtest neige : Convention 131 → 125, occupation Champs 7,5 → 6,5. Limite :
  les prévisions archivées sont à courte échéance, celles du rendu sont à J+2…J+6 : le backtest est un peu optimiste.
- *Événements* : saisis à la main, `data/evenements/evenements_manuels.csv` (71 lignes, 67 vérifiées par Pétronille ;
  139 occurrences). Fermetures (Champs : 1er dimanche piéton, 11/11, 08/05, 14/07, Nouvel An, Tour, PSG, VivaTech) et
  perturbations (grèves, manifestations, Foire de Paris, Mondial de l'Auto). `heure_debut`/`heure_fin` ne sont pas
  renseignées : marquer seulement les heures de fermeture dégradait les Champs sur les 11/11 2023-2025 (RMSE 254 → 272),
  car ces heures « Barré » sont absentes de l'entraînement. Sur la fenêtre : la cérémonie du 11/11.
- *Régimes* : marquent les ruptures durables (baisse des Champs après les JO) pour qu'elles ne passent pas pour du bruit.

**Chantiers (générés, non utilisés).** 4 colonnes `chantier_*` (surface d'emprise sur chaussée à ≤ 30 m et ≤ 150 m,
hors bâtiment, « commencé il y a moins de 7 jours ») sont générées si `data/chantiers/` existe
(`python -m src.data.update --sources chantiers`) ; `export_dataset --avec-chantiers` les ajoute au fichier livré. Elles
sont NaN du 01/01/2025 au jour du téléchargement (pas d'archive 2025) : inconnu, pas « aucun chantier ». Le modèle les
ignore (`EXCLUDED_PREFIXES`), faute de gain (voir 3_modele.md).

## 5. Prétraitement du modèle

`src/models/preprocessing.py`, appris sur les seules lignes ≤ cutoff. NaN remplis du plus précis au plus grossier :
(1) un lag S-1/S-2/S-3 manquant prend la moyenne des lags disponibles ; (2) un lag ou niveau encore manquant prend le
profil médian de la cible par arc × jour × heure ; (3) toute autre colonne prend sa médiane. Booléens en 0/1, arc en
indicatrices.

## 6. Limites connues

- Occupation des Champs-Elysées reconstruite depuis le 19/11/2025 (≈ 27 % d'erreur).
- Peu de 11 novembre dans les données valides (1 à 5 selon l'arc).
- Prévisions météo archivées plus précises que celles du rendu.
- Chantiers inconnus en 2025-2026 ; fermetures et déviations absentes de l'open data. 10 % des jours portent la moitié
  à deux tiers de l'erreur.

## 7. Sources (`data/<source>/`, non versionnées, sauf `evenements_manuels.csv`)

Tout est téléchargé par `python -m src.data.update` (connecteurs dans `src/data/sources/`). `datetime` en UTC,
dates calendaires en heure de Paris. Le téléchargement des archives annuelles demande `unzip` dans le PATH.

| Source | Fichier | Origine | Couverture |
|---|---|---|---|
| Trafic | `trafic/recent.parquet` | comptages routiers permanents (API Opendatasoft, opendata.paris.fr) | 13 mois glissants, J-1 |
| Trafic | `trafic/historique/<année>.parquet` | archives annuelles (`--history 2010 2025`) | 2010 → 2025 |
| Météo | `meteo/historique.parquet` | Open-Meteo, réanalyse ERA5 (observé, analyse seulement) | 2010 → J-2 |
| Météo | `meteo/prevision.parquet` | Open-Meteo Forecast | J-7 → J+16 |
| Météo | `meteo/prevision_historique.parquet` | Open-Meteo Historical Forecast | 2021 → J-1 |
| Calendrier | `calendrier/jours_feries.parquet` | calendrier.api.gouv.fr | 2006 → 2031 |
| Calendrier | `calendrier/vacances_scolaires.parquet` | data.education.gouv.fr, académie de Paris | 2017 → 2028 |
| Événements | `evenements/evenements_manuels.csv` → `evenements.parquet` | saisie manuelle | 2010 → N+1 |
| Optionnel | `evenements/que_faire_a_paris.parquet` | agenda « Que faire à Paris » (`update --all`) | non utilisé |
| Optionnel | `chantiers/*.parquet` | chantiers perturbants, chantiers à Paris, archives 2019-2024 (`update --all`) | non utilisé par le modèle |

Trafic : on garde toutes les rues des 3 arcs (~50 arcs) pour trouver des proxys ; filtrer sur `arc_id` ou sur les nœuds,
pas sur `libelle` (il change selon les années). Les archives sont en heure de Paris, converties en UTC (l'heure 02:00
dupliquée du passage à l'heure d'hiver est retirée) ; sur les heures communes, l'API l'emporte. Codes vérifiés contre
l'API (100 % des débits identiques) : `etat_trafic` 0-4 = Inconnu, Fluide, Pré-saturé, Saturé, Bloqué. Météo : un point au
centre de Paris ; le modèle n'utilise que les prévisions.
