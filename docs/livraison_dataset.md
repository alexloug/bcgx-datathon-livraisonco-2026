# Dataset livré à l'équipe modèle (v2, 04/11/2026)

`data/processed/dataset_modele.parquet` : une ligne par arc et par heure, du 21/06/2021 au 11/11/2026 23:00
(heure de Paris), avec les 2 cibles et les **44 features figées** du modèle. Les données sont arrêtées au
**03/11/2026 23:00**. Le fichier sera régénéré vendredi 06/11 au matin avec les données du 05/11.

## Récupérer le fichier

Le fichier (9 Mo) est transmis à part : `data/processed/` n'est pas versionné. Pour le régénérer à l'identique depuis
la branche `robin/data-plan` :

```bash
pip install -r requirements.txt                      # macOS : brew install libomp
python -m src.data.update --history 2020 2025        # sources open data (~10 min la première fois)
python -m src.data.build_dataset                     # -> data/processed/dataset.parquet (toutes les colonnes)
python -m src.data.export_dataset                    # -> data/processed/dataset_modele.parquet + .json, contrôlé
```

`export_dataset` refuse d'écrire si un contrôle échoue : doublons, grille horaire trouée, 360 lignes à prédire,
cibles sur la fenêtre, cible hors bornes, feature exclue présente.
`dataset_modele.json` donne la version, la dernière heure observée, les comptes de lignes et la liste des features
par famille.

## Contenu

141 843 lignes × 51 colonnes. `datetime` est en UTC (tz-aware) ; les features calendaires sont en heure de Paris.

| Groupe | Colonnes | Rôle |
| --- | --- | --- |
| Clés | `arc`, `datetime` | `arc` vaut `Champs-Elysées`, `Convention` ou `Saint-Pères`, comme dans le CSV de rendu |
| Cibles | `debit_horaire` (véh/h), `taux_occupation` (%) | NaN quand inconnues |
| Suivi | `is_valid`, `source_cible`, `partie` | quelles lignes entraîner, évaluer, prédire (règles ci-dessous) |
| Calendrier | 9 | `hour`, `dow`, `month`, `is_weekend`, `ferie`, `veille_ferie`, `lendemain_ferie`, `pont`, `heure_ete` |
| Calendrier scolaire | 6 | `vacances`, `vacances_type`, `jours_avant_reprise`, `jours_depuis_reprise`, `jours_avant_vacances`, `vacances_niveau28j` |
| Historique du trafic | 12 | `<cible>_lag168/336/504`, `_lag_mean`, `_niveau28j`, `_lag52s`, pour les 2 cibles |
| Météo prévue | 11 | `meteo_temperature_2m`, `_precipitation`, `_rain`, `_snowfall`, `_cloud_cover`, `_wind_speed_10m`, `_weather_code`, `meteo_pluie`, `meteo_neige_jour`, `meteo_neige_veille`, `meteo_tmin_jour` |
| Événements | 3 | `evenement`, `evenement_fermeture`, `evenement_perturbation` |
| Régimes | 3 | `jo_2024`, `post_jo_2024`, `post_30kmh` |

Définition, source et effet mesuré de chaque feature : [`features.md`](features.md).

Ce qui n'est **pas** dans le fichier, volontairement :
- **chantiers** et **météo observée** : aucun gain en backtest pour les premiers, inconnue au moment de prédire pour
  la seconde. Ces colonnes restent générées dans `dataset.parquet` ; les chantiers peuvent être ajoutés au fichier
  livré (section suivante) ;
- toute colonne calculée à partir de la cible de l'heure t (classe de trafic, indice de temps de parcours) : ce serait
  une fuite. Elles se calculent après la prédiction ([`reformulation_probleme.md`](reformulation_probleme.md) §2).

| `partie` | `source_cible` | Lignes | Sens |
| --- | --- | --- | --- |
| `historique` | `mesure` | 100 915 | cible mesurée par le capteur |
| `historique` | `proxy` | 7 571 | Champs-Elysées après l'arrêt du capteur (19/11/2025), reconstruit à partir des arcs voisins 4274 et 4281 |
| `historique` | `absent` | 32 784 | trou de capteur ou heure invalide : cible NaN, features présentes |
| `non_predit` / `a_predire` | `futur` | 573 | après la dernière heure observée ; `a_predire` = les 360 lignes du rendu (07/11 00:00 → 11/11 23:00) |

## Générer les données de chantier

Les chantiers ne sont pas dans le modèle pour l'instant, mais leur code est conservé et tourne à chaque
`build_dataset` : on peut les générer et les tester à tout moment.

```bash
python -m src.data.update --sources chantiers          # emprises 2019-2024 (archives) + chantiers en cours
python -m src.data.build_dataset                       # calcule les 4 colonnes chantier_* dans dataset.parquet
python -m src.data.export_dataset --avec-chantiers     # les ajoute au fichier livré (55 colonnes)
```

| Colonne | Définition |
| --- | --- |
| `chantier_surface_arc` | log(1 + m²) des emprises sur chaussée à 30 m ou moins de l'arc |
| `chantier_surface_150m` | idem à 150 m ou moins |
| `chantier_voirie_150m` | idem, voirie et réseaux seulement (sans chantiers de bâtiment) |
| `chantier_nouveau_7j` | une emprise sur chaussée à 150 m ou moins a commencé il y a moins de 7 jours |

- Code : `chantier_features` dans `src/data/build_dataset.py`, connecteur `src/data/sources/chantiers.py`.
- Couverture : complète de 2019 à 2024 et à partir du téléchargement ; **NaN du 01/01/2025 à la veille du
  téléchargement** (pas d'archive 2025 publiée), ce qui veut dire « inconnu », pas « aucun chantier ».
- Même avec `--avec-chantiers`, le modèle ne les lit pas : `chantier_` est dans `EXCLUDED_PREFIXES`
  (`src/models/preprocessing.py`). Pour les tester dans le modèle, retirer ce préfixe et laisser leurs NaN au lieu
  de les imputer (XGBoost les gère).
- Pourquoi ils sont sortis : en backtest (39 jeudis de 2022 à 2024), ils gagnent une semaine sur deux, soit du bruit
  ([`determinants_trafic.md`](determinants_trafic.md) §5). Une emprise ne dit ni si une voie est fermée ni par où
  passe la déviation : la donnée qui aiderait vraiment, ce sont les arrêtés de circulation.

## Règles d'utilisation

1. **Entraîner sur `is_valid` et `datetime <= cutoff`.** `is_valid` = les deux cibles présentes (mesurées ou
   reconstruites), hors heures « Invalide », « Barré », figées ou aberrantes.
2. **Prédire `partie == "a_predire"`** : exactement les 360 lignes du CSV de rendu.
3. **Pas de fuite, sans rien recalculer.** Toute feature de l'heure t n'utilise que des valeurs connues 144 h avant ou
   plus (1 jour sauté + 5 jours prédits). Pour un cutoff c, les lignes de c + 25 h à c + 144 h n'utilisent que des
   données ≤ c : un seul fichier sert à tous les backtests. Ne pas ajouter de lag plus court que 144 h.
4. **Les NaN sont normaux** dans les features historiques : trous de capteur (Convention 11/2022 → 06/2023,
   Saint-Pères 11/2024 → 10/2025). Le prétraitement du dépôt (`src/models/preprocessing.py`) les remplit sans fuite ;
   XGBoost sait aussi les gérer seul (écart < 2 % en backtest). Sur la fenêtre, les lags S-1 et S-2 sont renseignés
   à 80 % aujourd'hui, à 100 % vendredi.
5. **Champs-Elysées après le 19/11/2025 : `source_cible == "proxy"`.** Le débit reconstruit est fiable (6 % d'erreur
   en validation, ratio mensuel 4264 / 4274 entre 0,98 et 1,07 en 2025). **L'occupation l'est beaucoup moins**
   (environ 27 % d'erreur ; ratio mensuel 4264 / 4274 entre 1,7 et 2,8). C'est la prédiction la moins sûre des 6.
   `XGBModel(proxy_weight=...)` permet de pondérer ces heures ; à 0,5, rien ne change en backtest.

## Évaluer

Même découpage que le rendu : cutoff le jeudi 23:00, 1 jour sauté, 5 jours prédits. **RMSE sur les 24 heures**, comme
la note du brief (slide 12).

```bash
python -m src.evaluate --model xgb --n-folds 8 --dataset data/processed/dataset_modele.parquet
python -m src.evaluate --model xgb --cutoffs 2023-11-09 2024-11-07 2025-11-06 --dataset data/processed/dataset_modele.parquet
```

Références sur ce fichier (RMSE 24 h, débit en véh/h / occupation en points) :

| Arc | Baseline, 8 derniers jeudis | XGBoost, 8 derniers jeudis | Baseline, semaines du 11/11 | XGBoost, semaines du 11/11 |
| --- | --- | --- | --- | --- |
| Champs-Elysées | 192 / 9,6 | **171 / 8,3** | 216 / 15,6 | **190 / 13,7** |
| Convention | 82 / 2,9 | **66 / 2,5** | 92 / 2,1 | **62 / 1,8** |
| Saint-Pères | 160 / 3,7 | **103 / 2,9** | 108 / 2,2 | **90 / 1,9** |

8 derniers jeudis : du 03/09 au 22/10/2026 ; la cible des Champs y est entièrement reconstruite. Baseline : moyenne
des 4 dernières semaines au même jour et à la même heure. XGBoost : `src/models/xgb.py`, sans réglage.
Comparaison avec LightGBM et un réseau de neurones : [`comparaison_modeles.md`](comparaison_modeles.md).

## Limites connues

- **Occupation des Champs-Elysées** reconstruite (règle 5) ; on ne sait pas encore sur quelle vérité les Champs seront
  notés : question posée au tutorat du 05/11.
- **Peu de 11 novembre** dans les données valides : 1 à 5 selon l'arc. L'effet férié s'apprend sur tous les fériés.
- **Saint-Pères** : déviation du 28/08 au 20/09/2026 (+50 à +85 % de débit), qui gonfle encore le niveau 28 j ; fin
  du chantier du quai Voltaire le 30/10, juste avant la fenêtre. Écart à surveiller vendredi.
- **Météo** : en backtest, les prévisions archivées sont à courte échéance ; au rendu, on aura une prévision à
  J+2…J+6, moins précise.
- **Jours anormaux** (travaux, déviations, événements) : 10 % des jours portent la moitié à deux tiers de l'erreur, et
  aucune source ouverte ne les annonce.

## Vendredi 06/11 au matin

```bash
python -m src.data.update            # météo incrémentale : quelques appels, plus de refus Open-Meteo
python -m src.data.build_dataset
python -m src.data.export_dataset    # v2 régénérée : mêmes colonnes, données jusqu'au 05/11 23:00
```

À vérifier avant de prédire : dernière heure observée = 05/11 23:00 ; plus aucune feature incomplète sur la fenêtre
dans la sortie d'`export_dataset` ; proxys des Champs présents la semaine passée. Liste complète :
[`feuille_de_route_dataset.md`](feuille_de_route_dataset.md) §3.

## Changements depuis la v1 (03/11)

- Ajout du calendrier scolaire et de la neige ([`determinants_trafic.md`](determinants_trafic.md) §5).
- Événements séparés en fermetures et perturbations, 67 événements vérifiés (Pétronille).
- Chantiers retirés du modèle.
- 6 heures reconstruites des Champs avec une occupation > 100 % écartées (`source_cible` passe de `proxy` à `absent`).
- Mise à jour météo incrémentale, pour ne plus être bloqué par Open-Meteo.
