# Stream D : de la prévision du trafic au planning des livreurs

Ce module transforme les prédictions de trafic du livrable 1 en un planning de livreurs pour la semaine S+1. Il sert à chiffrer le gain attendu de la feuille de route (livrable 2).

## Fichiers

| Fichier | Rôle |
|---|---|
| `traffic_to_time.py` | Convertit débit et taux d'occupation en vitesse de parcours (km/h), par arc et par heure. Contient aussi un générateur de prédictions simulées au format du rendu. |
| `staffing.py` | PLNE de planning : shifts de 4 à 8 h, 10 EUR/h de 8 h à 20 h et 14 EUR/h sinon. Résolu avec `scipy.optimize.milp` (solveur HiGHS). |
| `run_demo.py` | Compare 4 scénarios sur 3 zones et 5 jours (07/11 -> 11/11), puis teste la sensibilité à la marge de sécurité. |

## Lancer

Seuls `numpy`, `pandas` et `scipy` (>= 1.9) sont nécessaires.

```bash
# sur nos prédictions (recommandé)
python -m src.roadmap.run_demo --pred output_victor.csv --calage relatif --out outputs/stream_d/victor
# sur du trafic simulé
python -m src.roadmap.run_demo
```

Le calcul prend environ 2 minutes. Les résultats sont écrits dans le dossier `--out` :

- `synthese_scenarios.csv` : coût, heures de livreurs, livraisons hors créneau et gain de chaque scénario ;
- `sensibilite_marge.csv` : coût de S2 selon la marge de sécurité ;
- `gain_annuel.csv` : coût et gain sur une année, à partir d'une semaine type (5 jours ouvrés, un samedi, un dimanche) ;
- `resultats_par_zone_jour.csv`, `vitesses_predites.csv`, `details.json` : détail par zone, par jour et par heure.

## Scénarios

- **S0** : planning actuel simulé. Le planificateur ne connaît que la vitesse moyenne de la semaine et prend 10 % de marge.
- **S0\*** : même logique, avec la plus petite marge qui évite toute livraison hors créneau. C'est la référence à service égal.
- **S1** : shifts calés heure par heure sur la vitesse prédite. Le B2B reste livré de 8 h à 18 h.
- **S2** : S1, en laissant l'optimiseur choisir l'heure des livraisons B2B entre 6 h et 19 h.

## Résultats sur nos prédictions (modèle final, 05/11)

| Scénario | Coût par livraison | Gain par rapport à S0\* |
|---|---|---|
| S0\* | 2,10 EUR | référence |
| S0 | 1,93 EUR, avec 1,5 % de livraisons hors créneau | non comparable |
| S1 | 1,81 EUR | -13,9 % |
| S2 | 1,69 EUR | -19,6 % |
| S2 avec 10 % de marge | | -11,8 % |

Sur une année (semaine type x 52), le coût livreurs actuel des 3 zones est d'environ 1 089 kEUR. Le gain est de 234 kEUR (-21,5 %) sans marge et de 150 kEUR (-13,7 %) avec 10 % de marge.

Sur le trafic simulé, on obtient -12,2 %, -19,3 % et -11,7 % : le résultat ne dépend pas de la source du trafic.

## Choix de modélisation

**Calage des vitesses.** La vitesse au capteur vaut v = q * L / tau, avec L la longueur d'un véhicule plus celle de la boucle (environ 5,5 m). Sur nos prédictions, le taux d'occupation de la Convention reste sous 6 % en journée, et cette formule donne plus de 50 km/h. Le calage `relatif` ne garde donc que la variation de q / tau dans la semaine, rapportée à son niveau en circulation libre (95e centile par arc) et ancrée sur 30 km/h * 0,6 = 18 km/h. Les vitesses obtenues vont de 5 à 18 km/h. Le calage absolu demandera les traces GPS des livreurs.

**Marge de sécurité.** En backtest, le modèle final se trompe d'environ 20 % sur le débit (RMSE de 109 véh/h, voir `pour_dannyl.csv`). Le temps de service à la porte ne dépend pas du trafic, donc une erreur de x % sur la vitesse ne pèse qu'environ x / 2 % sur la productivité. Nous retenons 10 % de marge. Le gain disparaît vers 25 %.

## Hypothèses à remplacer par les données de LivraisonCo

Elles sont définies en tête de `staffing.py`, `traffic_to_time.py` et `run_demo.py` : 4 min de service par livraison, 0,6 km entre deux livraisons, dépôt à 7 km, 700 / 600 / 450 livraisons par jour ouvré selon la zone, 40 % de B2B flexible les jours ouvrés.
