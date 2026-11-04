# Résultats du stream 2

Campagne : `outputs/experiments/stream2_initial`. Créée le 2026-11-04T17:59:02.663536+00:00.

Dataset : `dataset_modele.parquet`, empreinte SHA256 : `f6d2882347efabc793381c55e103a1b026005a54c0a19edd6de4fe3f6ef4b766`.

## Réglage

| candidate | relative_rmse |
| --- | --- |
| xgb_proxy_half | 0.767 |
| xgb_reference | 0.771 |
| xgb_regularized | 0.771 |
| xgb_depth4 | 0.772 |
| xgb_without_weather | 0.772 |
| xgb_without_events | 0.785 |
| baseline | 1.000 |

Configuration sélectionnée avant validation : **xgb_proxy_half**.

Score interne : moyenne des six ratios RMSE / baseline ; plus petit = meilleur, 1 = baseline. Ce score n'est pas l'agrégation officielle du concours.

## Validation temporelle récente

| candidate | relative_rmse |
| --- | --- |
| baseline | 1.000 |
| xgb_reference | 0.868 |
| xgb_proxy_half | 0.871 |

RMSE sur 24 h : débit en véhicules/heure ; occupation en points.

| candidate | arc | debit_horaire | taux_occupation |
| --- | --- | --- | --- |
| baseline | Champs-Elysées | 154.589 | 7.611 |
| baseline | Convention | 70.001 | 2.516 |
| baseline | Saint-Pères | 127.380 | 3.599 |
| xgb_proxy_half | Champs-Elysées | 142.666 | 5.817 |
| xgb_proxy_half | Convention | 72.987 | 2.091 |
| xgb_proxy_half | Saint-Pères | 103.765 | 3.052 |
| xgb_reference | Champs-Elysées | 141.999 | 5.769 |
| xgb_reference | Convention | 72.985 | 2.069 |
| xgb_reference | Saint-Pères | 104.838 | 3.045 |

## Diagnostic des semaines du 11 novembre

| candidate | relative_rmse |
| --- | --- |
| baseline | 1.000 |
| xgb_reference | 0.840 |
| xgb_proxy_half | 0.828 |

RMSE sur 24 h : débit en véhicules/heure ; occupation en points.

| candidate | arc | debit_horaire | taux_occupation |
| --- | --- | --- | --- |
| baseline | Champs-Elysées | 216.069 | 15.547 |
| baseline | Convention | 91.905 | 2.129 |
| baseline | Saint-Pères | 108.216 | 2.213 |
| xgb_proxy_half | Champs-Elysées | 192.830 | 13.510 |
| xgb_proxy_half | Convention | 63.797 | 1.763 |
| xgb_proxy_half | Saint-Pères | 90.600 | 1.880 |
| xgb_reference | Champs-Elysées | 195.743 | 13.271 |
| xgb_reference | Convention | 67.528 | 1.768 |
| xgb_reference | Saint-Pères | 90.945 | 1.939 |

## Interprétation et limites

La sélection repose seulement sur septembre 2026. La validation porte sur octobre, avec des fenêtres distinctes. Les semaines du 11 novembre sont diagnostiques, déjà explorées par l'équipe. Les périodes récentes ont également été étudiées dans les travaux précédents : ce n'est pas un test externe vierge.

Les cibles récentes des Champs-Elysées sont reconstruites. Les RMSE y mesurent la concordance au proxy. Seules les heures avec une vérité valide sont notées ; voir coverage.csv pour les effectifs.

Avant adoption : vérifier les régressions par rue/cible, étendre la validation aux jeudis réguliers 2022–2024 et confirmer l'agrégation officielle. Le modèle de rendu n'est pas remplacé automatiquement.

## Décision de cette campagne

**Conserver XGBoost de référence**, avec les paramètres actuels de `src/models/xgb.py`.
La configuration est rappelée dans `configs/model_retenu.json` ; `src.predict --model xgb` utilise déjà ces paramètres.
Ce fichier documente le choix, il n'est pas chargé automatiquement par `predict`.

La pondération proxy à 0,5 gagne seulement 0,46 % sur le score de réglage, puis perd 0,25 % par rapport à la
référence sur la validation d'octobre. Son gain de 1,4 % sur les trois semaines historiques du 11 novembre
ne suffit pas à établir une amélioration stable. Elle reste une piste, sans adoption pour cette campagne.

La référence améliore le score interne moyen de 13,2 % face à la baseline sur octobre, et de 16,0 % sur les
semaines du 11 novembre. Cela ne veut pas dire que tous les couples rue/cible progressent : le débit de
Convention régresse de 70,00 à 72,98 véhicules/heure de RMSE sur octobre (+4,3 %).

Les suppressions de météo et d'événements ne sont pas retenues : aucun gain moyen sur les semaines de réglage.
Les variantes de profondeur et de régularisation testées ne font pas mieux que la référence sur ce réglage.
Il s'agit d'une recherche ciblée, pas d'une optimisation exhaustive ni d'un choix définitif pour toutes les saisons.

Validation technique : 21 tests passent, plus une exécution de bout en bout du lanceur et du générateur de rapport.
Le jeu de données et les sources du stream 1 n'ont pas été modifiés par les expériences.
