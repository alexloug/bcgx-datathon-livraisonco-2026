# Stream 2 — Modélisation reproductible

## Entrée et périmètre

Entrée unique : `data/processed/dataset_modele.parquet`, export final du stream 1.
Ne pas repartir des anciens CSV. Le dataset complet et ses sources restent inchangés.
Entraînement sur les lignes valides antérieures au cutoff ; prétraitement réappris à chaque fold.
Un jour sauté, puis cinq jours prédits. RMSE sur toutes les heures et séparée par rue et cible.
Les labels Champs-Elysées récents sont des proxys : leur score ne mesure pas l'erreur contre un capteur opérationnel.

## Campagne initiale

Configuration versionnée : `configs/stream2.json`.

1. Réglage : quatre jeudis de septembre 2026. Comparer baseline, XGBoost de référence,
   profondeur 4, régularisation renforcée, sans météo, sans événements, poids proxy à 0,5.
2. Sélection : moyenne des six ratios RMSE candidat / RMSE baseline (trois rues, deux cibles).
   Ce score interne évite de sommer véhicules/heure et points d'occupation ; ce n'est pas la métrique officielle.
   Garder les six scores individuels pour repérer les régressions.
3. Validation : quatre jeudis d'octobre, fenêtres distinctes et postérieures au réglage.
   Seuls baseline, référence et candidat sélectionné sont évalués. Le choix n'est pas recalculé sur cette validation.
4. Diagnostic : semaines du 11 novembre 2023–2025. Ces périodes sont déjà connues de l'équipe,
   ce sont des diagnostics, pas un test indépendant. Elles ne participent pas au classement.

La validation récente utilise elle aussi une période déjà explorée dans les docs : la séparation limite
le biais de la nouvelle recherche mais ne constitue pas une évaluation externe vierge.
Une nouvelle campagne après consultation des résultats est exploratoire et doit être présentée comme telle.
Cette première campagne reste ciblée ; elle ne remplace pas une validation sur les jeudis réguliers 2022–2024
avant de généraliser une amélioration à toutes les saisons.

## Exécution et artefacts

```bash
python -m src.experiments.run --config configs/stream2.json
```

Un dossier daté est créé dans `outputs/experiments/`. Une destination existante est refusée pour éviter
l'écrasement. `--output outputs/experiments/nom_unique` permet de choisir son nom.

- `config.json`, `metadata.json` : protocole, versions, état Git et empreinte SHA256 du dataset.
- `ranking_tuning.csv` : classement de réglage ; 1 = niveau de la baseline, plus petit = meilleur.
- `selected.json` : configuration retenue avant validation.
- `scores_validation.csv`, `scores_diagnostic.csv` : scores de contrôle.
- Dans chaque expérience : paramètres, prédictions Parquet, RMSE par arc, fold, horizon et origine de cible,
  plus nombre d'heures réellement évaluées (`coverage.csv`). Les heures sans vérité valide ne sont pas notées.

Une prédiction absente ou non finie sur une heure évaluable fait échouer la campagne ; elle ne doit pas
améliorer artificiellement le score en disparaissant d'une jointure.

## Décision finale

Lire les six RMSE, la stabilité par fold et les semaines du 11 novembre avant d'adopter une variante.
Une baisse du score moyen ne suffit pas à justifier une régression forte sur une rue.
Conserver XGBoost de référence si les gains sont faibles ou incohérents. Écrire la décision dans
`docs/resultats_stream2.md`, en indiquant la campagne et ses limites.

Le lanceur ne publie rien et ne génère pas le rendu final. `src.predict` conserve les paramètres par défaut ;
une configuration gagnante doit être intégrée explicitement au rendu après revue des résultats.

## Nettoyage effectué

Les CSV initiaux, leurs deux scripts, le notebook `02_Preprocessing` et les anciens PDF/figures sont déplacés
vers `archive/legacy_csv/` (local, ignoré par Git). Ils sont conservés pour consultation.
Les notebooks 01, 02_qualite_donnees et 03_construction_dataset restent actifs.
`build_traffic.py` et `build_external.py` restent disponibles comme outils historiques : le constructeur
unifié importe encore `EVENTS` depuis `build_external.py`, il ne faut donc pas supprimer ce module.
Les anciens plans et comptes rendus du stream data restent des documents historiques ; la livraison v2
et `features.md` font référence pour le schéma actuel.

## Environnement local macOS / Python 3.9

Les dépendances manquantes sont installées dans `.venv`. NumPy reste inférieur à 2 sous Python 3.9
pour être compatible avec la version PyTorch disponible sur cet environnement.
Homebrew étant absent, `libomp.dylib` a été installée uniquement dans `.venv/lib/` depuis le paquet
`conda-forge/llvm-openmp` (provenance et empreinte dans `.venv/lib/openmp-provenance.json`).
Dans ce terminal, avant les commandes d'entraînement ou les tests complets :

```bash
export DYLD_LIBRARY_PATH="$PWD/.venv/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

Sur une autre machine macOS équipée de Homebrew, la solution habituelle reste `brew install libomp`.

Après une campagne terminée, produire la note et le graphique comparatif :

```bash
python -m src.experiments.report outputs/experiments/NOM_CAMPAGNE
```

La note est écrite dans `docs/resultats_stream2.md` et le graphique dans le dossier de campagne.
