# 3. Le modèle : XGBoost, protocole de validation, pistes écartées

Code : `src/models/preprocessing.py` (retraitement), `src/models/xgb.py` (modèle, nom `xgb`), `src/models/baseline.py`,
`src/evaluate.py` (backtest). Les modèles et protocoles écartés sont dans `exploration/`.

## 1. Le modèle retenu

1. **Features** : les 44 du dataset livré (voir 2_donnees.md), plus 4 ajoutées par le prétraitement.
2. **NaN** : imputés sans fuite, appris sur les lignes ≤ cutoff (voir 2_donnees.md §5).
3. **Modèle** : un `XGBRegressor` par cible, les 3 arcs ensemble (indicatrices d'arc), sur les heures valides ≤ cutoff,
   y compris les heures reconstruites des Champs (`proxy_weight`, 1 par défaut). 600 arbres, profondeur 6, taux
   d'apprentissage 0,05, sous-échantillonnage 0,8 : **réglages par défaut, jamais optimisés**.
4. **Post-traitement** : prédictions bornées à ≥ 0, occupation ≤ 100 %.

Baseline de référence : moyenne des 4 dernières semaines, même jour et même heure. Elle ignore les jours fériés ;
XGBoost les apprend grâce à `ferie` et au lag 52 semaines (sur le 11/11 seul : débit Convention 154 → 67, Saint-Pères
128 → 62, Champs 344 → 259).

Variables les plus utilisées (débit) : moyenne des lags S-1/S-2/S-3 (68 %), lag S-1, lag 52 semaines, férié, vacances,
JO 2024. Imputer les NaN ou les laisser à XGBoost donne des RMSE équivalentes (écart < 2 %) ; on garde l'imputation,
qui rend le pipeline indépendant du modèle. Un poids de 0,5 sur les heures proxy ne change rien non plus.

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
| LightGBM | ≈ XGBoost (écart 0 à 3 %), 3 fois plus lent (9,5 s contre 3 s par cutoff) | écarté |
| Réseau dense (MLP) | +5 à +15 % de RMSE sur le débit, instable selon la graine (±5 %) | écarté |
| Moyenne XGBoost + LightGBM + MLP | gagne 1 à 2 % (183 / 9,2 contre 186 / 9,5, Champs) | écarté |
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
- Hyperparamètres non optimisés (Victor y travaille) : prudence sur le surapprentissage des quelques semaines de réglage.
- Seul gain stable mesuré : entraîner sur les arcs voisins de la rue des Champs (occupation des Champs -6 à -9 %, 3
  cutoffs sur 4), avec un modèle simplifié, non retenu faute de temps et de reconfirmation avec le XGBoost du dépôt.
- Prédire le couple de façon cohérente (occupation à partir du débit prédit) ; intervalles P10-P90 par créneau.
