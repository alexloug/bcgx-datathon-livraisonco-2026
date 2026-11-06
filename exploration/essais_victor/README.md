# Essais de Victor (archive)

Scripts qui ont servi à tester des idées d'amélioration de XGBoost, du 3 au 5/11/2026. Les résultats et les décisions
sont dans [exploration/docs/resultats_essais_victor.md](../docs/resultats_essais_victor.md). Ce qui a été retenu est
passé dans `src/` : [src/models/ensemble.py](../../src/models/ensemble.py) et l'option `--manif` de
[src/predict.py](../../src/predict.py). Ces scripts ne servent pas au rendu.

Tous se lancent depuis la racine du dépôt, venv actif : `python -m exploration.essais_victor.<nom>`.
Ils lisent `data/processed/dataset_modele.parquet` : leurs chiffres ont été obtenus avant le passage des lags en heure
de Paris (dataset du 04/11), ils peuvent donc bouger légèrement si on les relance.

| Script | Ce qu'il teste |
|---|---|
| `anomalies.py` | Détection des jours anormaux (travaux, déviations) et des dérives du capteur d'occupation |
| `essai_xgb.py` | XGBoost de base et RMSE sur différents jeudis (réguliers 2022-2024, automnes 2024-2025, sept. 2026) |
| `recherche_xgb.py` | Recherche d'hyperparamètres (`RandomizedSearchCV`, validation jeudi par jeudi) |
| `analyse_features.py` | Corrélations, importance des colonnes (gain et permutation) |
| `essai_poids.py` | Plus de poids aux données récentes |
| `essai_decoupage.py` | Un modèle par rue, par tranche horaire, par heure |
| `essai_ferie.py` | Modèle par jour de semaine, modèle spécial fériés, poids des fériés (11/11) |
| `essai_proxy.py` | Poids des heures reconstruites des Champs-Elysées |
| `essai_ensemble.py` | Moyennes de modèles : graines, XGBoost + LightGBM, Random Forest |
| `essai_nettoyage.py` | Fonction de perte (Huber, absolue), retrait des jours anormaux, date de début |
| `nuit.py` | Batterie de 18 tests lancée la nuit du 4 au 5/11 (rapport : `outputs/experiments/nuit/rapport.md`) |

`nuit.py` est long parce qu'il enchaîne tous les tests sans s'arrêter en cas d'erreur. Pour comprendre un test, partir
plutôt du script simple correspondant ci-dessus.
