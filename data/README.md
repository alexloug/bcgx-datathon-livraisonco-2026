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
| `meteo/` | `prevision_historique.parquet` | [Open-Meteo Historical Forecast](https://open-meteo.com/en/docs/historical-forecast-api) | 2022 → J-1 | quotidienne |
| `calendrier/` | `jours_feries.parquet` | [calendrier.api.gouv.fr](https://calendrier.api.gouv.fr/jours-feries/) | 2006 → 2031 | rare |
| `calendrier/` | `vacances_scolaires.parquet` | [Calendrier scolaire](https://data.education.gouv.fr/explore/dataset/fr-en-calendrier-scolaire/), académie de Paris | 2017-2018 → 2027-2028 | annuelle |
| `evenements/` | `evenements_manuels.csv` → `evenements.parquet` | saisie manuelle | 2010 → N+1 | à la main |
| `evenements/` | `que_faire_a_paris.parquet` | [Que faire à Paris](https://opendata.paris.fr/explore/dataset/que-faire-a-paris-/) | événements encore en ligne | quotidienne |
| `chantiers/` | `chantiers_perturbants.parquet` | [Chantiers perturbants](https://opendata.paris.fr/explore/dataset/chantiers-perturbants/) | en cours / à venir | quotidienne |
| `chantiers/` | `chantiers_a_paris.parquet` | [Chantiers à Paris](https://opendata.paris.fr/explore/dataset/chantiers-a-paris/) | en cours / à venir | quotidienne |
| `chantiers/` | `chantiers_2024.parquet` | [Historique des chantiers 2024](https://opendata.paris.fr/explore/dataset/historique-chantiers-a-paris-en-2024/) | 2024 | figée |
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
« Que faire à Paris » contient surtout des petits événements culturels et pas d'historique fiable : à évaluer.

### chantiers/

Pas d'historique au-delà de 2024. Les coordonnées (`lat`, `lon`) permettent de garder les chantiers proches
de nos arcs.

## Sources cherchées sans succès (ou laissées de côté)

| Besoin | Constat |
|---|---|
| Manifestations | pas d'open data de la Préfecture de police |
| Grèves de transport (RATP, SNCF) | pas de jeu structuré |
| Matchs, salons, concerts majeurs | pas de source unique ; à saisir dans `evenements_manuels.csv` si utile |
| Secteurs « Paris Respire » | `secteurs-paris-respire` : liste statique, sans historique |
| Vacances scolaires avant 2017 | absentes de data.education.gouv.fr |
