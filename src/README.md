# Code du pipeline

Toutes les commandes se lancent depuis la racine du dépôt.

| Module | Rôle |
|---|---|
| `config.py` | Arcs, chemins, fenêtre cible et décalage minimum de 144 h |
| `data/update.py`, `data/sources/` | Collecte et actualisation des sources |
| `data/build_dataset.py` | Dataset complet, nettoyage et features externes |
| `data/export_dataset.py` | Export contrôlé du dataset final à 44 features |
| `features/build.py` | Calendrier et historique décalé sans fuite |
| `models/preprocessing.py` | Encodage et imputation appris avant le cutoff |
| `models/baseline.py` | Référence saisonnière |
| `models/xgb.py`, `gbm.py`, `mlp.py` | XGBoost, LightGBM et réseau dense implémentés |
| `models/registry.py` | Noms des modèles disponibles |
| `experiments/run.py` | Réglage, validation distincte et traçabilité du stream 2 |
| `evaluate.py` | Backtest 24 h avec contrôle des prédictions |
| `predict.py` | CSV de rendu et contrôles de format |

`data/build_traffic.py` et `data/build_external.py` sont des outils complémentaires de l'ancienne chaîne.
Ils ne sont pas nécessaires pour générer le dataset unifié ; `build_external.EVENTS` est encore utilisé.

Voir [le README principal](../README.md) et [le protocole stream 2](../docs/stream2_modelisation.md).
