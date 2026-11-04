# Avancement du stream data — 03/11/2026

Branche `robin/data-plan`. Objectif : des données propres et à jour, un dataset d'entraînement justifié, et un
premier modèle backtesté pour prédire le couple **(débit horaire, taux d'occupation)** des 3 arcs du 07/11 00:00 au
11/11 23:00.

## En bref

| Étape | Statut | Où |
|---|---|---|
| Plan de méthode (état des lieux → qualité → décisions → dataset) | fait | `docs/plan_stream_data.md` |
| Données : un dossier par source, connecteurs open data, script de mise à jour | fait | `data/`, `src/data/sources/`, `data/README.md` |
| Qualité des données, horizons, ruptures | fait | `notebooks/02_qualite_donnees.ipynb` |
| Dataset d'entraînement (2021+), proxy Champs-Elysées | fait | `src/data/build_dataset.py`, `docs/dataset_entrainement.md`, `notebooks/03_construction_dataset.ipynb` |
| Premier modèle XGBoost + retraitement des NaN + backtest | fait | `src/models/`, `docs/modele_xgboost.md` |
| Validation des décisions D1-D7 en équipe / au tutorat du 05/11 | **à faire** | `docs/plan_stream_data.md` §3 |
| Optimisation du modèle, rendu du 06/11 | **à faire** | — |

## Démarrage rapide

```bash
pip install -r requirements.txt                    # macOS : brew install libomp (XGBoost, LightGBM)
python -m src.data.update --history 2020 2025      # sources open data (~10 min avec les archives)
python -m src.data.build_dataset                   # -> data/processed/dataset.parquet
python -m src.evaluate --model xgb --n-folds 8     # backtest
python -m src.predict --model xgb --group <nom>    # CSV de rendu
python -m pytest -q tests
```

Les données (`.parquet`) ne sont pas versionnées : chacun les régénère avec ces commandes.

## 1. Données

Les anciens exports CSV ont été supprimés. Tout vient maintenant de l'open data, une source par dossier :

| Source | Contenu | Horizon |
|---|---|---|
| `data/trafic/` | comptages routiers, API Paris (13 mois) + archives annuelles, **51 arcs** des 3 rues | 2010 → J-1 |
| `data/meteo/` | Open-Meteo : observé, prévision, prévisions archivées | 2010 → J+16 (prévisions archivées depuis 2021) |
| `data/calendrier/` | jours fériés, vacances scolaires zone C | 2006 → 2031 / 2017 → 2028 |
| `data/evenements/` | événements saisis à la main (11/11, 8/05, 14/07, Nouvel An) + Que faire à Paris | 2010 → N+1 |
| `data/chantiers/` | chantiers perturbants et emprises de chantier | emprises 2019 → 2024 (archives), puis en cours → à venir |

Points techniques : les archives sont en heure de Paris (l'API en UTC), ce qu'on a vérifié contre l'API heure par heure.
Les codes d'état des archives sont décodés. L'archive 2021, compressée en Deflate64, est lue via `unzip`.
Sans source ouverte : manifestations, grèves, vacances avant 2017.

## 2. Qualité des données (notebook 02)

- **Le 30 km/h (08/2021) ne crée aucune rupture visible** sur les 3 arcs.
- Les vraies ruptures :
  - **Champs-Elysées, fin 2018 → 2019** : perte de capacité (débit ÷ 2, occupation × 2) ;
  - **Convention, 03/2016** : saut de débit ;
  - **Convention, 03/2020** : occupation ÷ 2 à débit égal (recalibrage probable) ;
  - **Covid** : 03/2020 → 06/2021.
- Le diagramme débit/occupation change d'une période à l'autre : l'historique ancien n'est pas comparable à aujourd'hui.
- Longs trous de capteur :
  - Convention : 11/2022 → 06/2023, puis plusieurs en 2024 ;
  - Saint-Pères : 11/2024 → 10/2025 ;
  - **Champs-Elysées (4264) : muet depuis le 19/11/2025**.

## 3. Dataset d'entraînement (notebook 03, `docs/dataset_entrainement.md`)

| Décision | Choix | Justification |
|---|---|---|
| D2 Période | **21/06/2021 → aujourd'hui** | prévisions météo archivées (backtest) depuis 2021, fin du couvre-feu, après les ruptures Champs 2019 et Convention 2020 |
| D1 Champs-Elysées | débit : **4274** (Berri → La Boétie), secours 4281 ; occupation : **moyenne 4274/4281** | MAE 2025 : débit 46 véh/h (6 %), occupation 4,1 points ; 4267 (Balzac → Washington) inutilisable (aucune mesure de 2021 à 2025) |
| D3-D5 Nettoyage | heure valide = débit et occupation présents, arc ni Invalide ni Barré, pas figée ≥ 6 h, pas de pic | pas d'interpolation : les trous sont des blocs de plusieurs semaines |
| D7 Météo | prévision (archivée, puis courante sur la fenêtre) | c'est l'information disponible au moment de prédire |

Contenu : 141 843 lignes (arc × heure), les deux cibles et 34 features pour le modèle :
- calendrier : fériés, veilles et lendemains, ponts, vacances ;
- valeurs passées des deux cibles, toujours à 144 h ou plus : S-1/S-2/S-3, niveau 28 jours, 52 semaines ;
- météo prévue ;
- événements ;
- indicateurs de régime : JO 2024, après JO, après 30 km/h.

Toutes les features ont au moins 144 h de décalage. Un seul dataset sert donc pour tous les backtests, sans fuite.
La construction est une fonction réutilisable : `build_dataset(start, end)`.

## 4. Premier modèle : XGBoost (`docs/modele_xgboost.md`)

Le pipeline est appris uniquement sur les données disponibles au cutoff :
1. sélection des features ;
2. retraitement des NaN : moyenne des valeurs passées disponibles, puis profil arc × jour × heure, puis médiane ;
3. un XGBoost par cible.

RMSE en backtest (cutoff le jeudi 23:00, 1 jour sauté, 5 jours prédits), baseline → XGBoost :

| Arc | Débit, 8 derniers jeudis | Occupation, 8 derniers jeudis | Débit le 11/11 (2023-2025) |
|---|---|---|---|
| Champs-Elysées* | 193 → 193 | 10,9 → **9,4** | 344 → **259** |
| Convention | 82 → **65** | 2,9 → **2,5** | 154 → **67** |
| Saint-Pères | 160 → **102** | 3,7 → **2,9** | 128 → **62** |

\* Champs-Elysées est évalué sur une cible reconstruite par proxy depuis le 19/11/2025.

## 5. Points ouverts et prochaines étapes

1. **Tutorat du 05/11** : valider le proxy des Champs-Elysées (D1) et la période 2021+ (D2).
2. **Modèle** : optimiser les hyperparamètres sur les backtests et ajouter des cutoffs autour d'autres jours fériés.
   À tester aussi : prédire l'écart au niveau récent, et rendre cohérent le couple débit/occupation prédit.
3. **Rendu du 06/11** : relancer `update` → `build_dataset` → `predict` avec les données jusqu'au 05/11 23:00.
   Les valeurs à S-1 de la fenêtre seront alors complètes.
4. **Données** : enrichir `evenements_manuels.csv` (exceptions, gros événements), saisir les vacances 2010-2017 si on
   élargit la période.
5. **Limites connues** :
   - occupation des Champs-Elysées reconstruite avec ~27 % d'erreur relative ;
   - peu de 11 novembre dans la période (1 à 5 selon l'arc) ;
   - les prévisions météo archivées sont à courte échéance, donc plus optimistes que la prévision à J+2…J+6 du rendu.

## Historique de la branche

| Commit | Contenu |
|---|---|
| `bad34f2` | plan de méthode du stream data |
| `01f293b` | un dossier par source, connecteurs open data, `src/data/update.py` |
| `1785a56` | archives trafic en Deflate64, reprise si une année échoue |
| `16fa679` | notebook 02 de qualité des données, conclusions dans le plan |
| `ff68472` | construction du dataset (2021+, proxy Champs-Elysées) et documentation |
| `0b369d1` | XGBoost, retraitement des NaN, backtest et rendu branchés sur le dataset |
