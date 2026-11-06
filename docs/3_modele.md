# 3. Le modèle : ensemble XGBoost + LightGBM, protocole de validation, pistes écartées

Code : `src/models/ensemble.py` (modèle final, nom `ensemble`, par défaut dans `predict.py`), `src/models/xgb.py` (XGBoost
seul, nom `xgb`), `src/models/baseline.py`, `src/evaluate.py` (backtest). Les modèles et protocoles écartés, et les
essais de Victor, sont dans `exploration/`.

## 1. Le modèle retenu

1. **Features** : les 44 du dataset livré (voir 2_donnees.md) ; la rue est encodée en indicatrices. Les NaN sont laissés
   à XGBoost et LightGBM, qui savent les gérer (même score que l'imputation du prétraitement, écart < 2 %).
2. **Modèle** : par cible, la **moyenne de 6 modèles** : 3 XGBoost et 3 LightGBM (3 graines chacun), les 3 arcs ensemble,
   sur les heures valides ≤ cutoff, heures reconstruites des Champs comprises. 600 arbres, taux d'apprentissage 0,05,
   sous-échantillonnage 0,8 : **réglages par défaut** (profondeur 6 pour XGBoost, 63 feuilles pour LightGBM), car aucune
   recherche n'a fait mieux que le bruit (voir §4).
3. **Occupation nettoyée** : pour l'occupation seulement, on retire de l'entraînement les jours où le capteur dérive
   (occupation > 60 % au-dessus de l'habituel alors que le débit reste à moins de 20 % de l'habituel), plus 3 épisodes
   repérés à la main. Le débit n'est pas touché.
4. **Post-traitement** : prédictions bornées à ≥ 0, occupation ≤ 100 %.

Gains de l'ensemble sur le XGBoost d'origine (29 jeudis : réguliers 2022-2024, sept.-oct. 2026, semaines du 11/11) :
débit +0,7 à +2,0 %, occupation +4,8 à +6,2 % selon l'arc ; meilleur sur 71 % des (jeudi, arc, cible) et sur les 6 couples.
Le gain est modeste (le bruit d'une graine est de 2 à 3 %) mais régulier, et vient surtout de 2022-2024 pour le
nettoyage (nul sur 2026 et sur le 11/11). Détail : `exploration/docs/resultats_essais_victor.md`.

**Option `--manif AAAA-MM-JJ`** de `predict.py` : multiplie les prédictions d'une journée de manifestation de 12 h à 23 h
(× 0,65 Champs et Saint-Pères, × 0,8 Convention). **Désactivée par défaut** : à n'utiliser que si la manifestation est
confirmée (appliquée à tort un samedi normal, elle double l'erreur ; voir 4_rendu.md).

Baseline de référence : moyenne des 4 dernières semaines, même jour et même heure. Elle ignore les jours fériés ;
les modèles les apprennent grâce à `ferie` et au lag 52 semaines (sur le 11/11 seul, XGBoost : débit Convention
154 → 67, Saint-Pères 128 → 62, Champs 344 → 259).

Variables les plus utilisées (débit) : moyenne des lags S-1/S-2/S-3 (68 %), lag S-1, lag 52 semaines, férié, vacances,
JO 2024. Un poids de 0,5 sur les heures proxy ne change rien.

## 2. Protocole de validation

Même découpage que le rendu : **cutoff le jeudi 23:00 (heure de Paris), 1 jour sauté, 5 jours prédits**. Le
prétraitement et le modèle sont réappris à chaque fold. **RMSE sur les 24 heures**, comme la note du brief, par arc et
par cible. Un seul dataset sert à tous les folds, car toutes ses features sont décalées d'au moins 144 h.

`python -m src.evaluate --model xgb --folds all` compare toujours le modèle à la baseline sur exactement les mêmes
heures, et ajoute une ligne par run × fold × arc × cible à `outputs/backtest_runs.csv` (`--no-log` pour ne pas écrire).

| Type de fold | Contenu |
|---|---|
| `regular` | les 8 derniers jeudis (03/09 → 22/10/2026) ; cible des Champs entièrement reconstruite |
| `holiday` | les 34 jeudis depuis 2022 dont la fenêtre samedi → mercredi contient un férié |
| `--cutoffs` | des jeudis choisis, par exemple les semaines du 11/11 (2023-11-09, 2024-11-07, 2025-11-06) |

Une prédiction absente ou non finie sur une heure évaluable fait échouer le backtest : elle ne doit pas améliorer le
score en disparaissant d'une jointure. Les heures sans vérité valide ne sont pas notées.

Limites : les folds de 2022 sont appris sur 6 à 12 mois seulement ; Convention a moins d'heures (trous de capteur) ; le
« jour d'horizon 6 » affiché est un artefact des semaines de passage à l'heure d'hiver (121 h réelles) ; un rendu
unique, sur une fenêtre de 5 jours : toute différence de quelques % entre variantes est dans le bruit.

## 3. Résultats

XGBoost bat la baseline sur les deux familles de semaines, pour les 3 arcs et les 2 cibles. Références (RMSE 24 h,
débit en véh/h / occupation en points, données au 03/11/2026) :

| Arc | Baseline, 8 derniers jeudis | XGBoost, 8 derniers jeudis | Baseline, semaines du 11/11 | XGBoost, semaines du 11/11 |
|---|---|---|---|---|
| Champs-Elysées | 192 / 9,6 | 171 / 8,3 | 216 / 15,6 | 190 / 13,7 |
| Convention | 82 / 2,9 | 66 / 2,5 | 92 / 2,1 | 62 / 1,8 |
| Saint-Pères | 160 / 3,7 | 103 / 2,9 | 108 / 2,2 | 90 / 1,9 |

Les chiffres à jour de la dernière exécution sont dans le README. Par fold, XGBoost fait mieux que la baseline sur 68 des
82 couples (fold × arc) pour le débit des semaines fériées.

## 4. Ce qui a été essayé et écarté

La règle de décision pour garder une famille de features : à réglage inchangé, en RMSE 24 h, elle doit (1) améliorer
les jeudis réguliers 2022-2024, (2) ne pas dégrader les 9 derniers jeudis, (3) ne pas dégrader les 3 semaines du 11/11.
On n'a qu'un rendu : une feature qui aide en 2023 mais pas en 2026 est un pari. Détails et tableaux complets :
`exploration/README.md` et `exploration/docs/`.

| Piste | Résultat | Décision |
|---|---|---|
| LightGBM seul | ≈ XGBoost (écart 0 à 3 %), 3 fois plus lent (9,5 s contre 3 s par cutoff) | écarté seul ; entre dans l'ensemble |
| Réseau dense (MLP) | +5 à +15 % de RMSE sur le débit, instable selon la graine (±5 %) | écarté |
| Moyenne XGBoost + LightGBM + MLP | gagne 1 à 2 % (183 / 9,2 contre 186 / 9,5, Champs) | MLP écarté ; XGBoost + LightGBM retenu (§1) |
| Profondeur 4, régularisation renforcée | score relatif 0,772 / 0,771 contre 0,771 pour la référence | écarté |
| Sans météo prévue | 0,772 contre 0,771 | pas de gain à la retirer ; gardée (neige) |
| Sans événements | 0,785 contre 0,771 : dégrade | gardés |
| Proxy pondéré à 0,5 | 0,767 au réglage, 0,871 contre 0,868 en validation d'octobre | écarté |
| Chantiers (39 jeudis 2022-2024, RMSE débit avec / sans) | Champs 193 / 193, Convention 91 / 89, Saint-Pères 79 / 78 : un fold sur deux | exclus du modèle |
| Famille « rue » (arcs voisins) | ±3 %, sauf Convention occupation -9 % sur 3 semaines du 11/11 | refusée (condition 2) |
| Lags robustes (S-4, médiane) | ±2-4 % | refusée (conditions 1 et 2) |
| Régime de congestion | Champs débit -3 % en réguliers, +2 % sur les 9 derniers jeudis | refusée (condition 2) |
| Cible relative au niveau 28 j | Saint-Pères 109 contre 96 | rejetée |
| Correction du biais de la semaine précédente | le biais ne persiste pas (-0,45 à +0,54) | rejetée |

**Score interne de la campagne** (réglage sur 4 jeudis de septembre, validation sur 4 jeudis d'octobre, diagnostic sur les
semaines du 11/11 ; moyenne des six ratios RMSE / baseline, 1 = baseline) :

| Candidat | Réglage | Validation (oct.) | Semaines du 11/11 |
|---|---|---|---|
| baseline | 1,000 | 1,000 | 1,000 |
| xgb_reference | 0,771 | 0,868 | 0,840 |
| xgb_proxy_half | 0,767 | 0,871 | 0,828 |

Sur octobre, la référence améliore le score de 13,2 % face à la baseline, et de 16,0 % sur les semaines du 11/11, mais pas
chaque couple : le débit de Convention régresse de 70,0 à 73,0 véh/h (+4,3 %). La recherche est ciblée, pas exhaustive ; la
validation d'octobre n'est pas un test externe vierge (période déjà étudiée par l'équipe).

**Pourquoi XGBoost et pas un réseau** : ~100 000 lignes tabulaires aux features hétérogènes (booléens, compteurs, lags),
terrain où les boostings dominent ; les ruptures (JO, fériés rares) sont des seuils que les arbres apprennent
directement.

**Pourquoi les features d'historique ne s'enrichissent plus** : elles sont saturées. Le modèle tire déjà des lags et du
niveau 28 j ce que les arcs voisins savent (les anomalies sont locales à la rue : corrélation 0,07-0,28 entre rues
différentes). Le gain restant se joue sur les lignes (cible des Champs, jours anormaux), pas sur de nouvelles colonnes.

## 5. Correction des lags en heure de Paris : effet mesuré

RMSE du débit avant → après (un seul run) : 8 jeudis, Champs 182,8 → 182,6, Convention 65,7 → 65,9, Saint-Pères
101,9 → 102,0 ; 3 semaines du 11/11, 200,1 → 196,1, 68,2 → 68,0, 88,8 → 87,9. Sur le seul 11/11 (150 h), l'effet est
mélangé (Champs +3 %, Saint-Pères -4 %), dans le bruit d'un si petit échantillon. Les 3 tests ajoutés (traversée du
25/10, heures en double ou manquantes, lags fractionnaires) échouent sur l'ancien code.

## 6. Limites et pistes

- **Champs-Elysées** : l'erreur est dominée par la cible elle-même, reconstruite par proxy ; l'occupation des Champs est
  la prédiction la moins sûre.
- Hyperparamètres : non optimisés au final ; 44 combinaisons tirées au hasard donnent des RMSE à moins de 1,7 % les unes des autres (bruit d'une graine : 2 à 3 %).
- Seul gain stable mesuré : entraîner sur les arcs voisins de la rue des Champs (occupation des Champs -6 à -9 %, 3
  cutoffs sur 4), avec un modèle simplifié, non retenu faute de temps et de reconfirmation avec le XGBoost du dépôt.
- Prédire le couple de façon cohérente (occupation à partir du débit prédit) ; intervalles P10-P90 par créneau.
