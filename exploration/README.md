# Exploration : ce qu'on a testé et pourquoi on ne l'a pas retenu

Tout ce dossier est hors du chemin du rendu (`src/`). Il garde la trace des pistes écartées, avec leurs chiffres.
Le code déplacé ici n'est plus branché sur `src/models/registry.py` (qui connaît `baseline`, `xgb` et `ensemble`) :
il sert de référence, pas de commande à relancer telle quelle. Pour rejouer un modèle, l'ajouter au registre
(`"gbm": _factory("exploration.models.gbm", "GBMModel")`) et installer `torch` pour le MLP.
Ses tests : `python -m pytest exploration/tests`. Les essais de Victor se relancent un par un :
`python -m exploration.essais_victor.<script>`.

## Contenu

| Dossier | Quoi |
|---|---|
| `models/` | LightGBM (`gbm.py`) et réseau dense PyTorch (`mlp.py`) |
| `experiments/`, `configs/` | Protocole de réglage / validation du stream 2 et ses configurations |
| `notebooks/` | EDA, qualité des données, construction du dataset |
| `data/` | Anciens constructeurs `build_traffic.py` et `build_external.py`, remplacés par `src/data/build_dataset.py` |
| `docs/` | Les 13 documents d'origine, avant leur fusion en 4 dans `docs/` à la racine ; résultats détaillés des essais de Victor (`resultats_essais_victor.md`) |
| `essais_victor/` | Une trentaine d'essais pour améliorer XGBoost (3-5/11), un script par famille ; voir son README |

## Ce qu'on a écarté

### Essais d'amélioration de XGBoost (Victor, 3-5/11)
Backtest qui rejoue le rendu, RMSE sur la fenêtre de 5 jours, 3 graines moyennées. Changer seulement la graine fait
varier la RMSE de 1,7 à 2,6 % : un gain plus petit est du bruit. Règle : gain moyen ≥ 1 %, ≥ 65 % des
(jeudi, arc, cible) gagnés, aucun groupe de jeudis < −2 %, aucun couple (arc, cible) < −5 %. Gain positif = erreur
plus faible. Tableau complet, avec le pourquoi de chaque décision : [docs/resultats_essais_victor.md](docs/resultats_essais_victor.md).

| Famille | Pistes testées | Résultat |
|---|---|---|
| Découpage du modèle | un modèle par arc, par tranche horaire, par heure, par jour de semaine, spécial fériés | −0,5 à −77 % : chaque modèle a moins de données |
| Données d'entraînement | poids au récent (demi-vie 6 mois à 2 ans), début en 2023 ou sur 12-24 mois, poids du proxy, retrait des jours anormaux, arcs voisins des Champs | −4 à +1 %, rien de stable |
| Colonnes | retrait de la météo ou des grèves, données fraîches selon le jour prédit, profil médian | −0,5 à +1 %, dans le bruit |
| Cible et perte | Huber, erreur absolue, log(1 + y), écart à la moyenne des lags, occupation tirée du débit prédit | −18 à +1,9 % ; les meilleurs perdent sur les semaines du 11/11 |
| Réglage | `RandomizedSearchCV` (10 et 44 combinaisons), arrêt précoce, contraintes monotones | −6 à +4 % : hyperparamètres dans le bruit, le reste dégrade |
| Combinaisons | Random Forest, XGBoost plus aléatoire, mélange avec la baseline | −7 à +1,7 % |

Retenu et passé dans `src/` : la moyenne XGBoost + LightGBM sur 3 graines et le nettoyage des dérives du capteur
d'occupation (`src/models/ensemble.py`), plus la correction du samedi de manifestation (`src/predict.py --manif`).

### LightGBM et réseau de neurones : XGBoost suffit
Backtest du 04/11, mêmes données et mêmes features pour tous, RMSE de 6 h à 20 h, débit (véh/h) / occupation (points) :

| Jeu de cutoffs | Arc | Baseline | XGBoost | LightGBM | MLP |
|---|---|---|---|---|---|
| Jeudis réguliers 2022-2024 | Champs-Elysées | 213 / 11,1 | 186 / 9,5 | 184 / 9,3 | 199 / 9,6 |
| | Convention | 106 / 5,5 | 87 / 4,8 | 88 / 4,7 | 97 / 4,8 |
| 2e semaine après reprise | Champs-Elysées | 201 / 10,5 | 161 / 8,1 | 161 / 8,2 | 175 / 9,5 |
| | Convention | 125 / 4,9 | 70 / 4,3 | 67 / 4,4 | 77 / 4,7 |

- LightGBM ≈ XGBoost (écart de 0 à 3 %, il gagne 53 % des cutoffs × arcs sur le débit : du bruit) et 3 fois plus lent (9,5 s contre 3 s par cutoff).
- Le MLP perd 5 à 15 % de RMSE débit sur les jeudis réguliers et les reprises, et varie de ±5 % selon la graine.
- Moyenner les trois modèles gagne 1 à 2 % de RMSE (183 / 9,2 contre 186 / 9,5 sur les Champs-Elysées) : pas assez pour justifier trois modèles.
- Aucun modèle n'était réglé : on mesure l'effet du modèle, pas son optimisation.

### Réglages XGBoost, météo prévue, événements, pondération du proxy
Campagne `stream2_initial` : réglage sur septembre 2026, validation sur octobre. Score = moyenne des ratios RMSE / baseline (1 = baseline, plus petit = mieux).

| Candidat | Réglage (sept.) | Validation (oct.) | Semaines du 11/11 |
|---|---|---|---|
| baseline | 1,000 | 1,000 | 1,000 |
| xgb_reference | 0,771 | 0,868 | 0,840 |
| xgb_proxy_half (proxy pondéré 0,5) | 0,767 | 0,871 | 0,828 |
| xgb_depth4 | 0,772 | | |
| xgb_regularized | 0,771 | | |
| xgb_without_weather | 0,772 | | |
| xgb_without_events | 0,785 | | |

- Profondeur 4 et régularisation : pas mieux que la référence. Retirer la météo prévue ne change rien (0,772 contre 0,771) ; retirer les événements dégrade (0,785).
- Le proxy pondéré à 0,5 gagne 0,46 % au réglage, perd 0,25 % en validation : pas de gain stable, non adopté.
- Limite : une recherche ciblée sur peu de semaines, pas une optimisation exhaustive.

### Chantiers
Emprises de chantier de la Ville, 4 features (`chantier_*`), testées sur 39 jeudis de 2022 à 2024 (RMSE débit avec / sans) : Champs-Elysées 193 / 193, Convention 91 / 89, Saint-Pères 79 / 78. Les chantiers gagnent un fold sur deux.
Une emprise ne dit pas si une voie est fermée ni où part la déviation, et la moitié des jours ont une emprise à moins de 150 m.
Les colonnes restent générées dans `dataset.parquet` (si `data/chantiers/` existe), mais le modèle les ignore. Aucune emprise de voirie à moins de 150 m des 3 arcs sur la fenêtre du 07-11/11.

### « Que faire à Paris »
Agenda culturel de la Ville, téléchargé mais jamais intégré aux features : les événements retenus sont la liste manuelle `data/evenements/evenements_manuels.csv`. Source disponible avec `python -m src.data.update --all`.

### Anciennes chaînes de données
`data/build_traffic.py` et `data/build_external.py` ont précédé `src/data/build_dataset.py`. Seule la liste de périodes `EVENTS` a été reprise (dans `src/config.py`).

### Arcs similaires (Pétronille)
Pour élargir l'entraînement (plus de fériés et de 11 novembre), recherche des capteurs qui se comportent comme nos 3 arcs : `data/arcs/arcs_similaires.csv` (15 arcs par arc du brief, classés par un score de forme du profil horaire + niveau de débit + niveau d'occupation). Script `data/arcs_similaires.py` (~10 min ; `--local` pour recalculer sans retélécharger), justification de chaque choix dans `docs/arcs_similaires.md`.
Les Champs ressemblent aux grands boulevards (Madeleine, Sébastopol, Clichy, Foch), Convention au Bd Brune et au Quai d'Orsay, Saint-Pères à Raspail et aux Gobelins. Les arcs similaires réagissent aux fériés plus comme nos arcs que la moyenne (Convention, Saint-Pères ; non concluant pour les Champs, 4 jours seulement). Des tronçons voisins partagent le même compteur (Convention 5662/5664/5674 = 5671, Saint-Pères 192 = 191) : écartés.
