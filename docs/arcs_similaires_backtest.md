# Arcs similaires : données, features et backtest

06/11/2026. Suite de [`arcs_similaires.md`](arcs_similaires.md) (choix des arcs, branche `petronille/arcs-similaires`) :
on télécharge leur historique, on construit le dataset augmenté et on mesure en backtest si ces arcs aident à
prédire les 3 arcs du brief.

**Conclusion : pas de gain robuste.** La meilleure variante (`xgb_pool_w5`) gagne 1 à 4 % sur Convention et
Saint-Pères, perd 2 % sur le débit des Champs-Elysées, et ne gagne que 6 à 7 cutoffs sur 10. C'est trop faible et
trop peu régulier pour changer le modèle du rendu.

## Reproduire

```bash
python -m src.data.sources.trafic_similaires        # API 13 mois + archives 2021-2025 (~10 min, 240 Mo par an)
python -m src.data.controle_similaires              # doublons, couverture, ruptures -> data/arcs/arcs_similaires_retenus.csv
python -m src.data.build_dataset_similaires         # -> data/processed/dataset_similaires.parquet
python -m src.backtest_similaires --jeux recents novembre etales   # ~25 min
```

Sorties : `outputs/backtest_similaires_<jeu>.csv` (RMSE) et `_predictions.parquet` (prédictions heure par heure).

## 1. Données

- 43 arcs (les 45 de la liste, moins 4281, déjà proxy de la cible des Champs, et Bd Brune 725, compté deux fois),
  2021 → 05/11/2026, `data/trafic/similaires/` (non versionné). `src/data/sources/trafic.py` filtre désormais les
  archives par identifiant d'arc (`arc_ids`) en plus du filtre par rue.
- **Doublons** : 5 paires d'arcs ont le même débit sur plus de 90 % des heures, donc un seul capteur :
  6812/6587 (Invalides), 724/725 (Brune), 342/344 (Raspail), 5061/4807 (Gobelins), 5025/5026 (Av. d'Italie). On
  garde le mieux couvert.
- **Couverture** : on exige au moins 70 % d'heures valides depuis le 21/06/2021. Beaucoup d'arcs ont des trous
  de plusieurs mois (2024 surtout).
- **Retenus : 17 arcs**, soit 710 000 heures valides contre 108 000 pour nos 3 arcs.

| Arc du brief | Arcs retenus |
|---|---|
| Champs-Elysées | Bd de Sébastopol (45, 1218), Bd de Strasbourg (527), Quai de Gesvres (9), 4-Septembre (1964) |
| Convention | Bd Brune (724), Bd Auguste-Blanqui (5538), Bd Richard-Lenoir (410), Bd de Port-Royal (610), Bd des Invalides (6584), Av. de Villars (6498), Bd Malesherbes (237) |
| Saint-Pères | Av. des Gobelins (5061), Julia-Bartet (1169), Av. d'Italie (5025), Av. de Flandre (6451), Av. de la Porte-Brancion (1154) |

- **Ruptures de niveau** : une dizaine d'arcs étaient 25 à 60 % plus chargés en 2021-2022 qu'aujourd'hui (baisse
  générale du trafic parisien). C'est une raison de prédire en relatif (ci-dessous) plutôt qu'en véh/h.

## 2. Features

`src/data/build_dataset_similaires.py`. Mêmes règles que pour les arcs du brief :
- **Lignes des arcs similaires** : nettoyage `clean_targets`, mêmes lags S-1/S-2/S-3, moyenne des lags, niveau
  28 j et lag 52 semaines (`src/features/build.py`), même date de dernière observation. Calendrier, vacances, météo
  et régimes repris du dataset (ils ne dépendent que de l'heure) ; événements à False (ils concernent nos 3 rues).
- **Features de groupe** sur nos 3 arcs : moyenne, sur leurs arcs similaires, des lags rapportés au niveau 28 j
  (`sim_<cible>_lag168_rel`, `_lag_mean_rel`, `_lag52s_rel`), et `sim_n`, le nombre d'arcs disponibles (4 à 6 en
  moyenne). Idée : un profil hebdomadaire et un effet férié de l'an dernier moins bruités que ceux d'un seul
  capteur, disponibles même quand notre capteur est en panne. Toutes décalées d'au moins 144 h : pas de fuite.

## 3. Variantes testées

`src/models/xgb_similaires.py`, mêmes hyperparamètres que `xgb` :

| Variante | Cible | Features de groupe | Lignes d'entraînement |
|---|---|---|---|
| `xgb` (référence) | véh/h et % | non | 3 arcs du brief |
| `xgb_rel` | rapport au niveau 28 j | non | 3 arcs |
| `xgb_groupe` | véh/h et % | oui | 3 arcs |
| `xgb_rel_groupe` | relative | oui | 3 arcs |
| `xgb_pool` | relative | non | 3 arcs + 17 similaires, poids égaux |
| `xgb_pool_w5` | relative | non | idem, nos 3 arcs pèsent 5 fois plus |
| `xgb_pool_groupe` | relative | oui | idem, poids 5 |

On ne prédit et on ne note que les 3 arcs du brief.

## 4. Résultats

Cutoffs le jeudi 23:00, 1 jour sauté, 5 jours prédits, RMSE sur 24 h. 23 cutoffs : les 8 derniers jeudis
(03/09 → 22/10/2026), les 3 semaines du 11/11 (2023-2025) et 12 jeudis étalés sur 2023-2026. Convention n'est noté
que sur 15 cutoffs et Saint-Pères sur 19, à cause des pannes de leurs capteurs.

**Écart de RMSE à `xgb`, en % (négatif = mieux), sur tous les cutoffs** (sauf un artefact, voir plus bas) :

| Variante | Débit Champs | Débit Convention | Débit Saint-Pères | Occ. Champs | Occ. Convention | Occ. Saint-Pères |
|---|---|---|---|---|---|---|
| `xgb_rel` | +2,4 | −1,4 | +2,1 | +0,2 | −2,8 | +0,2 |
| `xgb_groupe` | +0,6 | +2,1 | −1,8 | −1,1 | +2,3 | −0,8 |
| `xgb_rel_groupe` | +2,1 | +0,5 | +2,5 | 0,0 | −0,8 | −1,7 |
| `xgb_pool` | +1,7 | −0,9 | +0,2 | +0,2 | −3,1 | +1,0 |
| **`xgb_pool_w5`** | +1,8 | **−1,2** | **−1,8** | +0,2 | **−3,8** | −0,1 |
| `xgb_pool_groupe` | +1,6 | +1,4 | −1,4 | −0,7 | −1,1 | −1,3 |

Référence `xgb` sur ces cutoffs : débit 170 / 69 / 88 véh/h, occupation 9,7 / 2,7 / 2,7 points.

Part des cutoffs où `xgb_pool_w5` fait mieux que `xgb` :

| Arc | Débit | Occupation |
|---|---|---|
| Champs-Elysées | 52 % | 35 % |
| Convention | 67 % | 73 % |
| Saint-Pères | 63 % | 68 % |

Lecture :
- **Champs-Elysées : aucun apport**, voire une légère perte. C'était attendu : ses arcs « similaires » le sont peu
  (score 1,0 à 1,6, effet férié au niveau du hasard), et la cible y est reconstruite par proxy.
- **Convention et Saint-Pères : un léger mieux** avec la mise en commun pondérée. Le gain est de 1 à 4 %, et le
  modèle fait mieux sur environ 2 cutoffs sur 3. C'est cohérent avec la qualité des arcs choisis, mais l'écart
  reste du même ordre que la variation d'une semaine à l'autre.
- **Les features de groupe seules n'apportent rien de net** : elles gagnent sur Saint-Pères, perdent sur Convention.
- **La cible relative seule n'aide pas** : elle sert seulement à mettre en commun des arcs de niveaux différents.
- **Semaines du 11/11** : la mise en commun réduit l'erreur sur Saint-Pères (−4 à −5 % de débit, 3 semaines sur 3
  pour `xgb_pool`) et sur l'occupation des Champs (−4 à −7 %). Mais il n'y a que 3 semaines, et une seule pour
  Convention.

**Artefact exclu** : le cutoff du 15/05/2025 ne note, pour Convention, qu'une journée (21/05) : celle où le capteur
redémarre après plusieurs semaines de panne. Le niveau 28 j y est imputé, et les variantes relatives multiplient
par ce niveau imputé (+90 à +170 % d'erreur, contre une RMSE de 183 déjà anormale pour `xgb`). C'est un défaut des
variantes relatives au redémarrage d'un capteur, pas un comportement de la semaine type ; à corriger si on les garde
(revenir à la prédiction absolue quand le niveau 28 j est imputé).

## 5. Décision et suites

- **Rendu de ce soir** : on garde `xgb` sur `dataset_modele.parquet`. Le gain mesuré est trop faible et trop
  irrégulier pour justifier un changement le jour du rendu.
- Si on veut tout de même l'utiliser : `xgb_pool_w5` pour Convention et Saint-Pères, `xgb` pour les Champs. C'est
  un gain attendu de l'ordre de 1 à 4 % sur 4 des 6 RMSE.
- **Short-list de features (étape 3)** : à faire sur la référence `xgb`, avant de revenir aux arcs similaires. Avec
  44 features pour 108 000 lignes, on a plus de chances de gagner en retirant du bruit qu'en ajoutant des arcs.
- Pistes non testées : poids du brief entre 2 et 20, filtrer les arcs sur `percentile_feries` < 60 %, utiliser les
  arcs similaires uniquement pour les jours fériés (là où nos 3 arcs manquent d'exemples).
