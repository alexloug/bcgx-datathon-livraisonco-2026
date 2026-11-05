# data/

Un dossier par source. Tout est téléchargé depuis l'open data par `python -m src.data.update`,
sauf `evenements/evenements_manuels.csv`, saisi à la main et versionné. Les connecteurs sont dans
`src/data/sources/<source>.py`. Les fichiers `.parquet` ne sont pas versionnés : chacun les régénère.

```bash
python -m src.data.update                        # toutes les sources (trafic : 13 mois glissants), ~2 min
python -m src.data.update --history 2010 2025    # + archives trafic manquantes, ~2 min et 200 Mo par an
python -m src.data.update --sources meteo        # une seule source
```

Convention : `datetime` en UTC (tz-aware), dates calendaires (`date`, `debut`, `fin`) en heure de Paris.

## Inventaire

| Dossier | Fichier | Source | Couverture | Mise à jour |
|---|---|---|---|---|
| `trafic/` | `recent.parquet` | [Comptages routiers permanents](https://opendata.paris.fr/explore/dataset/comptages-routiers-permanents/), API Opendatasoft | 13 mois glissants | quotidienne, J-1 |
| `trafic/` | `historique/<année>.parquet` | [Archives annuelles](https://opendata.paris.fr/explore/dataset/comptages-routiers-permanents-historique/) (zips hebdomadaires) | 2010 → 2025 | annuelle |
| `meteo/` | `historique.parquet` | [Open-Meteo Historical](https://open-meteo.com/en/docs/historical-weather-api) (réanalyse ERA5) | 2010 → J-2 | quotidienne |
| `meteo/` | `prevision.parquet` | [Open-Meteo Forecast](https://open-meteo.com/en/docs) | J-7 → J+16 | horaire |
| `meteo/` | `prevision_historique.parquet` | [Open-Meteo Historical Forecast](https://open-meteo.com/en/docs/historical-forecast-api) | 2021 → J-1 | quotidienne |
| `calendrier/` | `jours_feries.parquet` | [calendrier.api.gouv.fr](https://calendrier.api.gouv.fr/jours-feries/) | 2006 → 2031 | rare |
| `calendrier/` | `vacances_scolaires.parquet` | [Calendrier scolaire](https://data.education.gouv.fr/explore/dataset/fr-en-calendrier-scolaire/), académie de Paris | 2017-2018 → 2027-2028 | annuelle |
| `evenements/` | `evenements_manuels.csv` → `evenements.parquet` | saisie manuelle | 2010 → N+1 | à la main |
| `evenements/` | `que_faire_a_paris.parquet` | [Que faire à Paris](https://opendata.paris.fr/explore/dataset/que-faire-a-paris-/) | événements encore en ligne | quotidienne |
| `chantiers/` | `chantiers_perturbants.parquet` | [Chantiers perturbants](https://opendata.paris.fr/explore/dataset/chantiers-perturbants/) | en cours / à venir | quotidienne |
| `chantiers/` | `chantiers_a_paris.parquet` | [Chantiers à Paris](https://opendata.paris.fr/explore/dataset/chantiers-a-paris/) | en cours / à venir | quotidienne |
<<<<<<< Updated upstream
| `chantiers/` | `historique/<année>.parquet` | Historique des chantiers à Paris, une archive par an ([2024](https://opendata.paris.fr/explore/dataset/historique-chantiers-a-paris-en-2024/)) | 2019 → 2024 | figée |
=======
| `chantiers/` | `chantiers_2024.parquet` | [Historique des chantiers 2024](https://opendata.paris.fr/explore/dataset/historique-chantiers-a-paris-en-2024/) | 2024 | figée |
| `arcs/` | `referentiel.parquet`, `profils.parquet`, `osm.parquet` → `caracteristiques.parquet`, `arcs_similaires.csv` | [Référentiel des arcs](https://opendata.paris.fr/explore/dataset/referentiel-comptages-routiers/) + profil horaire agrégé par l'API trafic + [OpenStreetMap](https://overpass-api.de/) | tous les arcs, 12 mois glissants | `python -m src.data.arcs_similaires` |
>>>>>>> Stashed changes
| `processed/` | `dataset.parquet` (+ `dataset.json`) | `src/data/build_dataset.py`, voir `docs/dataset_entrainement.md` | 21/06/2021 → fin de la fenêtre | dérivé |
| `processed/` | `traffic_hourly.parquet`, `external.parquet` | `src/data/build_traffic.py`, `build_external.py` | — | dérivés |

## Détail par source

### trafic/

Schéma commun API et archives : `arc_id, libelle, nd_amont, nd_aval, datetime, debit_horaire,
taux_occupation, etat_trafic, etat_arc, source` (`api` ou `archive`).

- On garde toutes les rues de `config.STREETS` (Champs-Elysées, Saints-Pères, Convention), environ 50 arcs,
  pas seulement les 3 du brief. Cela permet de chercher des proxys (Champs-Elysées) et des arcs voisins.
- Archives en heure de Paris, converties en UTC. Au passage à l'heure d'hiver, l'heure 02:00 n'apparaît
  qu'une fois pour deux heures réelles : elle est retirée.
- Codes des archives vérifiés contre l'API sur oct.-nov. 2025 (100 % des débits identiques heure par heure) :
  `etat_trafic` 0-4 = Inconnu, Fluide, Pré-saturé, Saturé, Bloqué ; `etat_barre` 2 = Barré, 3 = Ouvert.
  Les autres codes `etat_barre` (0, 1) sont gardés tels quels (`code_1`…) : leur sens reste à établir.
  L'API publie aussi « Invalide », que les archives n'ont pas.
- Le libellé d'un même arc change selon les années (`AV_Champs_Elysees` / `Av_des_Champs_Elysees`) :
  filtrer sur `arc_id` ou sur les nœuds, pas sur `libelle`.
- `trafic.load()` assemble archives et API ; sur les heures communes, l'API l'emporte.

### meteo/

Un point au centre de Paris (48.8566, 2.3522). Variables : température (réelle et ressentie), précipitations,
pluie, neige, nébulosité, vent et rafales, humidité, code temps, jour/nuit.
On entraîne sur l'observé, mais au moment du rendu on n'aura que la prévision du 07 au 11/11 :
`prevision_historique` sert à mesurer l'effet de cette différence en backtest.

### calendrier/

Vacances : `debut` = premier jour sans cours, `fin` = jour de la reprise (exclu). Rien avant 2017 en open data.
Les ponts et les veilles de fériés se dérivent de `jours_feries` dans la phase features.

### evenements/

`evenements_manuels.csv` liste les événements qui ferment ou chargent nos rues et qu'aucune source ouverte
ne publie. `recurrence` vaut `annuelle` (même date chaque année) ou `ponctuelle`. Mettre `a_verifier = oui`
en cas de doute. Les exceptions ne sont pas encore saisies (défilé du 14 juillet 2020 annulé, etc.).
Colonnes facultatives : `type` (`fermeture`, par défaut, ou `perturbation`) et `heure_debut` / `heure_fin`
(heures de Paris, fin incluse, par défaut 0 / 23 : toute la journée). Les heures ne sont pas renseignées pour
l'instant : au backtest des 11/11 2023-2025, marquer seulement les heures de fermeture dégradait les Champs-Elysées
(RMSE débit 254 -> 272), car ces heures sont « Barré » donc absentes de l'entraînement.
Les 1ers dimanches piétons ne sont saisis que lorsque la fermeture est visible dans les données (pas tous les mois).
« Que faire à Paris » contient surtout des petits événements culturels et pas d'historique fiable : à évaluer.

### chantiers/

Les archives annuelles 2019-2024 sont publiées sous des identifiants peu parlants (`chantiers-a-paris-copie`,
`-copie0`… `-copie3`, puis `historique-chantiers-a-paris-en-2024`), avec le même schéma que le jeu courant.
`chantiers.load_emprises()` assemble archives et jeu courant.

**Trou de couverture** (`chantiers.coverage_gap()`) : il n'y a pas d'archive 2025, et le jeu courant ne contient
que les chantiers pas encore terminés. Du 01/01/2025 à la veille du téléchargement, on ne connaît donc que les
chantiers encore en cours : les features chantiers y valent NaN (inconnu), pas 0.

Les coordonnées (`lat`, `lon`) situent chaque emprise par rapport au segment de l'arc (`segment` dans
`config.ARCS`). `chantiers_perturbants` n'a pas d'historique : il ne sert qu'à lire la situation actuelle.

### arcs/

But : trouver les capteurs qui se comportent comme nos 3 arcs, pour élargir l'entraînement (plus de fériés, de
11 novembre). Script : `src/data/arcs_similaires.py` (~10 min, `--local` pour recalculer sans retélécharger).

- `caracteristiques.parquet` : une ligne par arc mesuré sur les 12 derniers mois (~1 850) : position, longueur,
  débit et occupation moyens, couverture, rapport week-end / semaine, profil horaire normalisé (`sem_00`…`we_23`).
  4264 (muet depuis le 19/11/2025) : profil des 12 mois précédents, depuis les données locales.
- `arcs_similaires.csv` (versionné, à lire) : 15 arcs par arc du brief, classés par `score` (plus petit = plus
  proche) = écart de forme du profil + écart de niveau de débit + écart de niveau d'occupation, chacun divisé par
  sa médiane. Candidats : couverture ≥ 80 %, débit moyen ≥ 100 véh/h, doublons (même capteur sous deux arcs) retirés.
- Premiers constats : les Champs ressemblent aux grands boulevards (Madeleine, Sébastopol, Clichy, Foch) ;
  Convention ressemble au Bd Brune, au Quai d'Orsay, au Bd des Invalides ; Saint-Pères à Raspail, Gobelins,
  Vincent Auriol. Des tronçons voisins partagent le même compteur de débit (Convention 5662/5664/5674 = 5671,
  Saint-Pères 192 = 191) : écartés, ce ne sont pas des exemples en plus.
- `jours_feries.parquet` : débit 7h-21h par arc les jours fériés et les jours de référence. Contrôle : les arcs
  similaires réagissent aux fériés plus comme nos arcs que la moyenne (Convention, Saint-Pères ; non concluant pour
  les Champs-Elysées, 4 jours seulement).
- Justification de chaque choix : `docs/arcs_similaires.md`.
- Pas encore vérifié : l'historique de ces arcs depuis 2021 (trous, ruptures), à faire en les téléchargeant.
- `osm.parquet` : routes OpenStreetMap de Paris (type, voies, vitesse, sens, couloir de bus), associées à 96 % des
  capteurs. Serveurs Overpass souvent saturés : le script réessaie sur plusieurs serveurs.

## Sources cherchées sans succès (ou laissées de côté)

| Besoin | Constat |
|---|---|
| Manifestations | pas d'open data de la Préfecture de police |
| Grèves de transport (RATP, SNCF) | pas de jeu structuré |
| Matchs, salons, concerts majeurs | pas de source unique ; à saisir dans `evenements_manuels.csv` si utile |
| Secteurs « Paris Respire » | `secteurs-paris-respire` : liste statique, sans historique |
| Vacances scolaires avant 2017 | absentes de data.education.gouv.fr |
