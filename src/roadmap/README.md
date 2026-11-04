# Stream D — Vision cible et feuille de route

Module qui transforme les prédictions de trafic du livrable 1 en planning de livreurs optimisé pour la semaine S+1. Il sert à chiffrer la feuille de route du livrable 2.

## Contenu

| Fichier | Rôle |
|---|---|
| `traffic_to_time.py` | D1 · débit + taux d'occupation → vitesse de parcours (v = q·L/τ, coefficient urbain 0,6). Contient aussi un générateur de prédictions de substitution au format exact du livrable 1. |
| `staffing.py` | D2 · PLNE de planning (shifts de 4 à 8 h, 10 €/h de 8 h à 20 h, 14 €/h sinon), résolu avec `scipy.optimize.milp` (HiGHS). |
| `run_demo.py` | Démo bout-en-bout : 4 scénarios (S0, S0*, S1, S2) sur 3 zones × 5 jours (07 → 11/11). |

## Lancer

```bash
pip install numpy pandas scipy        # scipy >= 1.9 pour milp
# avec des prédictions de substitution
python -m src.roadmap.run_demo
# avec nos vraies prédictions (fichier du rendu, séparateur ;)
python -m src.roadmap.run_demo --pred outputs/output_<groupe>.csv
```

Durée : environ 2 minutes, surtout à cause de la recherche de la marge S0*.

Sorties dans `outputs/stream_d/` :
- `synthese_scenarios.csv` : coût, heures livreurs, retards et gain par scénario
- `resultats_par_zone_jour.csv` : le détail par zone et par jour
- `vitesses_predites.csv` : vitesses et types de jour
- `details.json` : profils horaires et shifts retenus, pour les graphiques

## Scénarios

- **S0** : planning actuel simulé. Productivité moyenne constante et marge de 10 %.
- **S0\*** : même logique, avec la plus petite marge qui évite tout retard. C'est la référence « à service égal ».
- **S1** : shifts calés sur la vitesse prédite heure par heure. Le B2B reste livré de 8 h à 18 h.
- **S2** : S1, avec l'heure des livraisons B2B laissée à l'optimiseur (fenêtre 6 h-19 h).

## Résultats actuels (prédictions de substitution)

| Scénario | Coût / livraison | Livraisons hors créneau | Gain vs S0* |
|---|---|---|---|
| S0* | 1,94 € | 0 % | — |
| S0 | 1,72 € | 2,0 % | (service dégradé) |
| S1 | 1,70 € | 0 % | −12 % |
| S2 | 1,56 € | 0 % | −19 % |

## Hypothèses à remplacer

Toutes les hypothèses sont en tête de `staffing.py`, `traffic_to_time.py` et `run_demo.py`. La liste figure dans l'annexe A3 du deck : temps de service 4 min, 0,6 km entre deux livraisons, dépôt à 7 km, volumes 700 / 600 / 450 livraisons par jour ouvré, 40 % de B2B flexible.
