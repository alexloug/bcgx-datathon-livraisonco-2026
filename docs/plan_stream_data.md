# Plan du stream data

Objectif : livrer un jeu d'entraînement propre, documenté et sans fuite pour prédire `debit_horaire` et
`taux_occupation` sur les 3 arcs du brief, du 07/11 00:00 au 11/11 23:00 (dernière heure connue : 05/11 23:00).

On avance en trois phases. On ne passe à la suivante qu'après avoir validé la précédente en équipe.

| Phase | Question | Livrable | Statut |
|---|---|---|---|
| 1. État des lieux | Qu'a-t-on, et que voudrait-on avoir ? | Inventaire des sources + liste des features cibles (§1) | à faire |
| 2. Qualité | Que valent ces données, arc par arc et feature par feature ? | `notebooks/02_qualite_donnees.ipynb` + tableau de synthèse (§2) | à faire |
| 3. Décisions et dataset | Que garde-t-on, comment le traite-t-on ? | Journal des décisions (§3) + `data/processed/train.parquet` (§4) | à faire |

---

## 1. État des lieux

### 1.1 Données à disposition

Inventaire détaillé, schémas et limites : [`data/README.md`](../data/README.md). Tout se met à jour avec
`python -m src.data.update` (connecteurs dans `src/data/sources/`).

| Source | Contenu | Période | Dossier |
|---|---|---|---|
| Comptages routiers (API + archives open data Paris) | Débit, occupation, états, ~50 arcs des 3 rues | 2010 → J-1 | `data/trafic/` |
| Open-Meteo | Météo horaire observée, prévue, et prévisions archivées | 2010 → J+16 | `data/meteo/` |
| calendrier.api.gouv.fr, data.education.gouv.fr | Jours fériés, vacances zone C | 2006 → 2031 / 2017 → 2028 | `data/calendrier/` |
| Saisie manuelle + Que faire à Paris | Cérémonies, défilés ; agenda culturel | 2010 → N+1 | `data/evenements/` |
| Open data Paris | Chantiers perturbants, emprises de chantier | 2024 → à venir | `data/chantiers/` |

Constat rapide sur `traffic_hourly.parquet` (01/10/2025 → 01/11/2026, 9 525 heures par arc) :

| Arc | Heures valides | Dernière heure valide | État de l'arc |
|---|---|---|---|
| Champs-Elysées | 12 % | 19/11/2025 | 8 087 h « Invalide », 156 h « Barré » |
| Convention | 93 % | 01/11/2026 | 512 h « Invalide » |
| Saint-Pères | 93 % | 01/11/2026 | 555 h « Invalide » |

Champs-Elysées est le problème n°1 : sans proxy ni historique long, on n'a pas de quoi l'entraîner.

### 1.2 Features que l'on veut avoir

Contrainte forte : une feature pour l'heure t n'utilise que des valeurs observées à t − 144 h ou avant
(`MIN_LAG_HOURS`). Les variables exogènes doivent être connues à l'avance (calendrier) ou prévues (météo).

| Famille | Feature | Source | Disponible ? | Priorité |
|---|---|---|---|---|
| Calendrier | heure, jour de semaine, mois, week-end | dérivé de `datetime` | oui | haute |
| Calendrier | jour férié, pont, veille de férié | `external.parquet` (ponts à ajouter) | partiel | haute (le 11/11 est dans la fenêtre) |
| Calendrier | vacances scolaires zone C | API data.education.gouv.fr | non | haute (fin des vacances de Toussaint) |
| Historique | lags S-1, S-2, S-3 même heure, moyenne | `features/build.py` | oui | haute |
| Historique | même jour l'an dernier (lag 52 semaines), profil moyen par heure × jour sur 4-8 semaines | archives + recent | non | moyenne |
| Historique | lags croisés débit ↔ occupation, arcs voisins | recent | non | moyenne |
| Météo | pluie, température, vent (historique + prévision 07-11/11) | Open-Meteo / Météo-France | non | moyenne |
| Événements | manifestations, matchs, salons, cérémonies du 11 novembre (Champs-Elysées) | à recenser | non | moyenne |
| Travaux | chantiers et fermetures sur ou près des arcs | open data Paris (chantiers perturbants) | non | basse |
| Régimes | confinement, JO 2024 | `external.parquet` | oui | basse (utile seulement avec les archives) |

> À compléter avec ce qu'on s'est dit en équipe : ajouter une ligne par idée, avec la source envisagée.

---

## 2. Évaluation de la qualité

Un notebook `notebooks/02_qualite_donnees.ipynb`, une section par source, puis un tableau de synthèse.

### 2.1 Comptages (par arc, débit et occupation séparément)

- **Complétude** : part d'heures présentes sur la grille continue, part de NaN, longueur et position des trous
  (heatmap jour × heure, histogramme des durées de trous).
- **Validité** : répartition de `etat_arc` (Ouvert / Barré / Invalide) et de `etat_trafic` ; valeurs présentes
  sur des heures « Invalide » ; cohérence entre `q` et `k` (débit nul avec occupation > 0, etc.).
- **Valeurs aberrantes** : `q < 0`, `k > 100`, valeurs constantes sur plusieurs heures (capteur figé),
  pics isolés (écart > n σ au profil horaire habituel).
- **Ruptures** : changement de niveau dans le temps (travaux, recalibrage du capteur, changement de sens),
  à comparer d'une année sur l'autre si on récupère les archives.
- **Horodatage** : doublons, passages heure d'été / heure d'hiver (fin octobre, juste avant la fenêtre),
  cohérence UTC ↔ Paris.
- **Cohérence entre sources** : instantané du brief vs API vs archives sur les périodes communes.
- **Proxys Champs-Elysées** : couverture de 4401 / 4402, corrélation avec 4264 sur la période où les trois
  émettent, ratio de niveau stable ou non.

### 2.2 Données externes

- Couverture temporelle (historique **et** fenêtre 07-11/11), granularité, fuseau horaire.
- Pour la météo : écart prévision / observé, puisqu'on entraîne sur l'observé et prédit sur la prévision.
- Pouvoir explicatif rapide : profil moyen du trafic avec / sans le flag (férié, vacances, pluie).

### 2.3 Synthèse attendue

| Source / arc | Couverture | Valides | Problèmes repérés | Gravité | Traitement proposé |
|---|---|---|---|---|---|
| … | … | … | … | bloquant / gênant / mineur | … |

---

## 3. Décisions de traitement

À remplir après la phase 2, une ligne par décision, avec la raison et l'impact mesuré quand on peut.

| # | Sujet | Options envisagées | Décision | Pourquoi | Date |
|---|---|---|---|---|---|
| D1 | Arc Champs-Elysées muet depuis le 19/11/2025 | proxy 4401/4402 recalé sur 4264 ; historique 4264 seul ; profil moyen | | à trancher au tutorat du 05/11 | |
| D2 | Profondeur d'historique | 13 mois API seuls ; + archives 2010-2025 ; + archives à partir de 2022 | | | |
| D3 | Heures « Invalide » / « Barré » | exclure de l'entraînement ; garder avec un flag | | | |
| D4 | Trous dans les séries | laisser NaN (le GBM gère) ; interpoler les trous courts ; remplir par profil | | | |
| D5 | Valeurs aberrantes | écrêtage ; suppression ; rien | | | |
| D6 | Périodes atypiques (confinements, JO) | exclure ; garder avec flag | | | |
| D7 | Météo dans la fenêtre de prédiction | prévision ; climatologie ; pas de météo | | | |

---

## 4. Construction du dataset d'entraînement

Une fois les décisions prises :

1. Étendre `src/data/build_traffic.py` (archives, proxys, règles de nettoyage D3-D5).
2. Étendre `src/data/build_external.py` (vacances, ponts, météo, événements retenus).
3. Étendre `src/features/build.py` avec les features validées, en gardant `assert_no_leakage`.
4. Écrire `data/processed/train.parquet` : une ligne par arc × heure, cibles, features, `is_valid`,
   colonne `split` cohérente avec le backtest de `src/evaluate.py` (cutoff le jeudi 23:00, 1 jour sauté, 5 jours prédits).
5. Documenter le schéma (colonnes, types, unités, source) dans `data/README.md`.
6. Ajouter des tests dans `tests/` : pas de fuite, grille horaire continue, pas de doublons, couverture de la fenêtre 07-11/11.
7. Passer le relais au stream modèle : relancer le backtest baseline sur le nouveau dataset pour vérifier qu'on ne régresse pas.

---

## Points d'attention

- On prédit autour du 11 novembre : férié, cérémonie aux Champs-Elysées, fin des vacances de Toussaint.
  Sur 13 mois on n'a qu'une seule occurrence du 11/11 : les archives comptent.
- Toute feature exogène doit exister sur la fenêtre de prédiction, sinon elle est inutilisable.
- Les données récentes sont refetchées le 06/11 : le pipeline doit tourner de bout en bout sans intervention manuelle.
