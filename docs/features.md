# Features du dataset de prédiction du trafic

04/11/2026 · Robin. Version partagée : https://claude.ai/artifact/VRJtEtVTdKJWyxz6dps83k

## Vue d'ensemble

Le dataset compte une ligne par arc et par heure, du 21/06/2021 au 11/11/2026 : 2 cibles (`debit_horaire`, `taux_occupation`) et 55 features en 8 familles. Le modèle XGBoost en reçoit 48 : 44 colonnes du dataset (tout sauf les chantiers et la météo observée) et 4 ajoutées au prétraitement.

Trois règles valent pour toutes les features :

1. **Rien d'inconnu au moment de prédire.** Une feature de l'heure t n'utilise que des valeurs connues au cutoff, soit 144 h avant t ou plus (1 jour sauté + 5 jours prédits). `assert_no_leakage` le vérifie sur les lags.
2. **Exogène = connu à l'avance ou prévu.** Calendrier et événements sont connus à l'avance ; la météo est la **prévision**, jamais l'observé.
3. **Heure de Paris** pour tout ce qui est calendaire (jours, heures, vacances) ; `datetime` est stocké en UTC.

| Famille | Colonnes | Utilisée par le modèle | Où c'est codé |
| --- | --- | --- | --- |
| Calendrier | 9 | oui | `src/features/build.py`, `calendar_extra` |
| Calendrier scolaire | 6 | oui | `school_calendar`, `level_vacation_share` |
| Historique du trafic | 12 | oui | `src/features/build.py` |
| Météo prévue | 11 | oui | `weather` |
| Météo observée | 7 | non (analyse) | `weather` |
| Événements | 3 | oui | `event_flags` |
| Chantiers | 4 | non | `chantier_features` |
| Régimes | 3 | oui | `regime_flags` |
| Prétraitement | 4 | oui | `src/models/preprocessing.py` |

Sauf mention contraire, les fonctions sont dans `src/data/build_dataset.py`. Commande : `python -m src.data.build_dataset` (après `python -m src.data.update`).

## Calendrier

Le calendrier porte l'essentiel du débit : le profil heure × type de jour explique à lui seul 66 à 81 % de sa variance selon l'arc. Le 11/11 étant férié et dans la fenêtre, fériés, veilles et ponts sont prioritaires.

| Colonne | Type | Définition | Source |
| --- | --- | --- | --- |
| `hour` | 0-23 | heure de Paris | `datetime` |
| `dow` | 0-6 | jour de la semaine, 0 = lundi | `datetime` |
| `month` | 1-12 | mois | `datetime` |
| `is_weekend` | bool | samedi ou dimanche | `datetime` |
| `ferie` | bool | jour férié | calendrier.api.gouv.fr |
| `veille_ferie` | bool | la veille d'un férié (départs le soir) | idem |
| `lendemain_ferie` | bool | le lendemain d'un férié | idem |
| `pont` | bool | lundi avant un mardi férié, ou vendredi après un jeudi férié | idem |
| `heure_ete` | bool | heure d'été en vigueur | `datetime` |

Sur la fenêtre du 07 au 11/11/2026 : le 10 est `veille_ferie`, le 11 est `ferie`. Le 11 étant un mercredi, il n'y a pas de pont.

## Calendrier scolaire

Le trafic bouge à l'intérieur des vacances et autour des reprises, ce qu'un simple booléen `vacances` ne voit pas. Exemple : le débit de la mi-août est 12 à 30 % sous le profil habituel, celui de la dernière semaine avant la rentrée 17 à 43 % au-dessus.

| Colonne | Valeurs | Définition |
| --- | --- | --- |
| `vacances` | bool | vacances scolaires zone C (Paris), ponts de l'Ascension compris |
| `vacances_type` | 0-5 | 0 hors vacances, 1 Toussaint, 2 Noël, 3 Hiver, 4 Printemps, 5 Été |
| `jours_avant_reprise` | 0-42, sinon -1 | en vacances : jours jusqu'à la reprise |
| `jours_depuis_reprise` | 0-42, sinon -1 | hors vacances : jours depuis la dernière reprise |
| `jours_avant_vacances` | 0-42, sinon -1 | hors vacances : jours jusqu'aux prochaines vacances |
| `vacances_niveau28j` | 0-1 | part d'heures de vacances dans la fenêtre du niveau 28 j (mêmes bornes que `<cible>_niveau28j`) |

- Source : [calendrier scolaire](https://data.education.gouv.fr/explore/dataset/fr-en-calendrier-scolaire/), académie de Paris, 2017 → 2028.
- Les compteurs sont plafonnés à 42 jours : au-delà de 6 semaines, plus d'effet visible.
- Les ponts de l'Ascension ne comptent pas comme une période pour les compteurs.
- `vacances_niveau28j` corrige un niveau de référence tiré vers le bas après des vacances. Sur la fenêtre du 07 au 11/11, il vaut environ 0,55 : plus de la moitié du niveau tombe pendant la Toussaint.

Effets mesurés avant d'ajouter ces features (écart au profil habituel, 6 h-20 h) :

| Période | Débit | Occupation |
| --- | --- | --- |
| Mi-août (2-3 semaines avant la reprise) | -12 à -30 % | -8 à -32 % |
| Dernière semaine avant la rentrée | +17 à +43 % | +22 à +76 % |
| 1re semaine après la Toussaint | -3 à -10 % | Convention -20 %, Saint-Pères -11 % |
| 1re semaine après Noël | -5 à -9 % | -11 à -36 % |

## Historique du trafic

Les valeurs passées des deux cibles apportent ce que le calendrier ne voit pas : le niveau récent de chaque arc et ses ruptures (baisse des Champs-Elysées après les JO, baisse progressive de Convention). Chaque colonne existe pour `debit_horaire` et pour `taux_occupation`, soit 12 colonnes.

| Colonne | Décalage | Définition |
| --- | --- | --- |
| `<cible>_lag168` | 1 semaine | même heure, S-1 |
| `<cible>_lag336` | 2 semaines | même heure, S-2 |
| `<cible>_lag504` | 3 semaines | même heure, S-3 |
| `<cible>_lag_mean` | 1-3 semaines | moyenne des trois lags disponibles |
| `<cible>_niveau28j` | 144 h → 28 j | moyenne des 28 jours qui s'arrêtent 144 h avant t ; au moins 7 jours de données |
| `<cible>_lag52s` | 52 semaines | même heure, même jour de semaine, un an avant (utile pour le 11/11) |

- Seules les heures valides alimentent les lags : heures « Invalide », « Barré », figées ou aberrantes masquées (règles D3-D5).
- Champs-Elysées : le capteur 4264 est muet depuis le 19/11/2025. Ses heures sont reconstruites à partir des arcs 4274 et 4281 (ratios médians par heure × type de jour) et alimentent aussi les lags ; sans elles, l'arc n'aurait aucun lag sur la fenêtre. Erreur de la reconstruction : 6 % sur le débit, environ 27 % sur l'occupation.
- Un an d'historique est chargé avant le début du dataset pour que `lag52s` existe dès la première ligne.

## Météo

La météo pèse peu sur le débit (-1 à -2 % par pluie), mais nettement sur l'occupation et par neige. Toutes les colonnes `meteo_*` sont des **prévisions** : prévisions archivées Open-Meteo pour l'historique (depuis 2021), prévision courante sur la fenêtre. Un seul point, au centre de Paris.

| Colonne | Pas | Définition |
| --- | --- | --- |
| `meteo_temperature_2m` | heure | température (°C) |
| `meteo_precipitation`, `meteo_rain`, `meteo_snowfall` | heure | précipitations, pluie (mm), neige (cm) |
| `meteo_cloud_cover` | heure | nébulosité (%) |
| `meteo_wind_speed_10m` | heure | vent à 10 m (km/h) |
| `meteo_weather_code` | heure | code temps WMO |
| `meteo_pluie` | heure | précipitations > 0,1 mm/h |
| `meteo_neige_jour` | jour | neige prévue sur la journée (cm) |
| `meteo_neige_veille` | jour | neige prévue la veille (cm) |
| `meteo_tmin_jour` | jour | température minimale prévue du jour (°C) |

- **Pluie** : l'effet passe par l'intensité horaire. Au-delà de 1 mm/h, l'occupation de Convention monte de 19 à 23 %, celle de Saint-Pères de 8 à 10 %. Le cumul du jour n'explique rien : pas de feature journalière de pluie.
- **Neige** : la chute horaire est presque toujours nulle, même un jour de neige ; d'où les agrégats journaliers. Avec 0,5 cm ou plus prévus sur la journée (9 jours depuis 2021) : débit -13 à -16 %, occupation -11 à -40 %. L'effet reste net le lendemain.
- **Météo observée** (`meteo_obs_*`, 7 colonnes) : gardée pour l'analyse, jamais donnée au modèle, car vide sur la fenêtre.
- Limite : les prévisions archivées sont à courte échéance. Au rendu, on aura une prévision à J+2…J+6, moins précise : le backtest est un peu optimiste.

## Événements

Les événements qui ferment ou chargent nos rues ne sont publiés par aucune source ouverte : ils sont saisis à la main dans `data/evenements/evenements_manuels.csv` (71 lignes, dont 67 vérifiées par Pétronille). Le fichier distingue les fermetures, qui vident la rue, des perturbations, qui reportent ou chargent le trafic.

| Colonne | Définition |
| --- | --- |
| `evenement_fermeture` | rue de l'arc fermée ou vidée : cérémonies, défilés, Champs piétons, départ du marathon… |
| `evenement_perturbation` | trafic reporté ou chargé : grèves, manifestations, salons |
| `evenement` | l'un ou l'autre |

| Type | Rues | Lignes | Exemples |
| --- | --- | --- | --- |
| Fermeture | Champs-Elysées | 51 | 1er dimanche piéton (38), 11/11, 8/05, 14/07, Nouvel An, Tour de France, célébrations du PSG, VivaTech |
| Perturbation | les 3 rues | 15 | grèves et manifestations (retraites 2023, intersyndicale) |
| Perturbation | Convention | 5 | Foire de Paris, Mondial de l'Auto (Porte de Versailles) |

- `recurrence` : `annuelle` (même date chaque année, de 2010 à N+1) ou `ponctuelle`. 4 lignes annuelles donnent 72 occurrences ; avec les 67 ponctuelles, 139 au total.
- `heure_debut` / `heure_fin` (facultatives, heures de Paris) permettent de ne marquer que quelques heures. Elles ne sont pas renseignées : au backtest des 11/11 2023-2025, marquer seulement les heures de fermeture dégradait le débit des Champs-Elysées (RMSE 254 → 272), car ces heures « Barré » sont absentes de l'entraînement.
- Les 1ers dimanches piétons ne sont saisis que quand la fermeture se voit dans les données, pas tous les mois.
- Sur la fenêtre : la cérémonie du 11/11 à l'Arc de Triomphe (fermeture, Champs-Elysées).

## Chantiers

Les 4 colonnes chantiers sont dans le dataset mais **pas dans le modèle** : elles n'ont rien apporté en backtest. Elles servent à l'analyse et au contrôle de la fenêtre.

| Colonne | Définition |
| --- | --- |
| `chantier_surface_arc` | log(1 + m²) des emprises sur chaussée à 30 m ou moins du segment de l'arc |
| `chantier_surface_150m` | idem à 150 m ou moins (carrefours, rues adjacentes) |
| `chantier_voirie_150m` | idem, sans les chantiers de bâtiment : voirie et réseaux (eau, gaz, chauffage…) |
| `chantier_nouveau_7j` | une emprise sur chaussée à 150 m ou moins a commencé il y a moins de 7 jours |

- Source : emprises de chantier de la Ville de Paris, archives annuelles 2019-2024 (environ 160 000 emprises) et jeu courant.
- **Trou de couverture** du 01/01/2025 à la veille du téléchargement : pas d'archive 2025, et le jeu courant ne garde que les chantiers en cours. Les colonnes y valent NaN (inconnu), pas 0.
- Backtest XGBoost (39 jeudis de 2022 à 2024, 6 h-20 h) : les chantiers gagnent une semaine sur deux, soit du bruit. Sur les épisodes de chantier eux-mêmes, quelques gains et autant de pertes.
- Pourquoi ça ne marche pas : la moitié des jours ont une emprise sur chaussée à moins de 150 m, et une emprise ne dit ni si une voie est fermée ni par où passe la déviation.
- Pour les réactiver : retirer `chantier_` de `EXCLUDED_PREFIXES` (`src/models/preprocessing.py`) et laisser leurs NaN au lieu de les imputer.

## Régimes

Trois indicateurs marquent les ruptures durables de la période 2021-2026, pour que le modèle ne les confonde pas avec du bruit.

| Colonne | Période | Pourquoi |
| --- | --- | --- |
| `jo_2024` | 18/07 → 11/08/2024 et 28/08 → 08/09/2024 | JO et Jeux paralympiques, périmètres de sécurité |
| `post_jo_2024` | à partir du 09/09/2024 | baisse durable du débit de journée des Champs-Elysées (~1 200 → ~950 véh/h) |
| `post_30kmh` | à partir du 30/08/2021 | 30 km/h généralisé ; aucune rupture visible, gardé comme simple indicateur |

Le Covid (03/2020 → 06/2021) n'a pas d'indicateur : il est hors de la période, qui commence au 21/06/2021.

## Prétraitement et modèle

Le prétraitement (`src/models/preprocessing.py`) choisit les colonnes, ajoute 4 features et remplit les NaN. Il est appris sur les seules lignes antérieures au cutoff : aucune statistique ne vient du futur.

| Feature ajoutée | Définition |
| --- | --- |
| `arc_Champs-Elysées`, `arc_Convention`, `arc_Saint-Pères` | indicatrices d'arc : un seul modèle pour les 3 arcs |
| `nb_lags_manquants` | nombre de lags et niveaux manquants avant imputation |

Colonnes exclues du modèle (identifiants, cibles, puis par `EXCLUDED_PREFIXES`) : `arc`, `datetime`, `is_valid`, `source_cible`, `partie` ; `meteo_obs_*`, `chantier_*`.

Remplissage des NaN, du plus précis au plus grossier :

1. un lag S-1, S-2 ou S-3 manquant prend la moyenne des lags disponibles ;
2. un lag ou un niveau encore manquant prend le profil médian de la cible par arc × jour × heure ;
3. toute autre colonne prend sa médiane.

Un XGBoost par cible ; prédictions bornées à 0, et à 100 % pour l'occupation.

## Bilan et limites

Le calendrier et l'historique font l'essentiel ; les autres familles corrigent des périodes rares mais coûteuses pour les tournées. Les RMSE ci-dessous viennent des backtests XGBoost (cutoff le jeudi 23:00, 1 jour sauté, 5 jours prédits, 6 h-20 h sauf mention).

| Famille | Apport mesuré | Statut |
| --- | --- | --- |
| Calendrier + historique (modèle de base) | RMSE débit sur les 8 derniers jeudis, baseline → XGBoost (toutes heures) : Convention 82 → 65, Saint-Pères 160 → 102 | utilisée |
| Calendrier scolaire + neige | fin d'été : débit Convention 69 → 55, Saint-Pères 75 → 69 ; neige : occupation Champs 7,5 → 6,5 ; neutre sur les semaines ordinaires | utilisée |
| Événements | marquer seulement les heures du 11/11 dégradait les Champs (254 → 272) : journée entière gardée | utilisée |
| Chantiers | aucun gain (une semaine gagnée sur deux) | exclue |

Limites connues :

- Occupation des Champs-Elysées reconstruite depuis le 19/11/2025, avec environ 27 % d'erreur.
- Peu de 11 novembre dans la période : 1 à 5 selon l'arc.
- Prévisions météo archivées plus précises que celles du rendu.
- Chantiers inconnus en 2025-2026 ; fermetures et déviations absentes de l'open data.

Pistes : optimiser les hyperparamètres puis réévaluer chaque famille ; renseigner les heures des événements quand elles sont sûres ; vérifier si la ZTL Paris Centre (11/2024) couvre Saint-Pères et, si oui, ajouter un indicateur de régime.

Détail des analyses : `docs/determinants_trafic.md` et `docs/dataset_entrainement.md` dans le dépôt.
