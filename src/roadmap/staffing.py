"""Stream D : planning des livreurs pour la semaine S+1, formulé comme un PLNE.

Un problème est résolu par zone et par jour.

Variables
    x[t, d] (entier) : nombre de livreurs qui commencent à l'heure t un shift de d heures (4 <= d <= 8)
    y[h] (réel >= 0) : nombre de livraisons B2B placées à l'heure h, dans leur fenêtre autorisée

Objectif
    minimiser la somme des x[t, d] * coût du shift (10 EUR/h de 8 h à 20 h, 14 EUR/h sinon)

Contraintes
    couverture : à chaque heure h, capacité des livreurs présents >= livraisons B2C[h] + y[h]
    B2B        : somme des y[h] = volume B2B du jour, et y[h] = 0 hors de la fenêtre

La capacité d'un livreur dépend de la vitesse prédite (productivite_stops_h). Elle est
réduite la première et la dernière heure du trajet entre le dépôt et la zone.
Le solveur est HiGHS, via scipy.optimize.milp (aucune dépendance en plus de scipy).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

# Hypothèses opérationnelles, à remplacer par les données de LivraisonCo.
TEMPS_SERVICE_MIN = 4.0     # minutes par livraison : se garer, sonner, remettre le colis
DIST_INTER_STOP_KM = 0.6    # distance moyenne entre deux livraisons en zone dense
DIST_DEPOT_KM = 7.0         # distance entre le dépôt et la zone de livraison
DUREES_SHIFT = range(4, 9)  # shifts de 4 à 8 heures (brief)


def tarif_horaire(h: int) -> float:
    """Coût horaire chargé d'un livreur (brief) : 10 EUR de 8 h à 20 h, 14 EUR la nuit."""
    return 10.0 if 8 <= h < 20 else 14.0


def productivite_stops_h(v_kmh: np.ndarray) -> np.ndarray:
    """Livraisons par heure et par livreur, en fonction de la vitesse de parcours."""
    return 60.0 / (TEMPS_SERVICE_MIN + 60.0 * DIST_INTER_STOP_KM / v_kmh)


def approche_min(v_kmh: np.ndarray) -> np.ndarray:
    """Minutes de trajet entre le dépôt et la zone, en fonction de la vitesse."""
    return 60.0 * DIST_DEPOT_KM / v_kmh


@dataclass(frozen=True)
class Shift:
    debut: int
    duree: int

    @property
    def heures(self) -> range:
        return range(self.debut, self.debut + self.duree)

    @property
    def cout(self) -> float:
        return sum(tarif_horaire(h) for h in self.heures)


def tous_les_shifts() -> list[Shift]:
    """Tous les shifts autorisés qui commencent et finissent dans la journée (95 au total)."""
    return [Shift(t, d) for d in DUREES_SHIFT for t in range(0, 24 - d + 1)]


def capacite(shift: Shift, prod: np.ndarray, appr: np.ndarray) -> np.ndarray:
    """Livraisons possibles à chaque heure (vecteur de 24) pour un livreur sur ce shift."""
    cap = np.zeros(24)
    for h in shift.heures:
        fraction = 1.0
        if h == shift.debut:  # trajet aller depuis le dépôt
            fraction -= appr[h] / 60.0
        if h == shift.debut + shift.duree - 1:  # trajet retour vers le dépôt
            fraction -= appr[h] / 60.0
        cap[h] = prod[h] * max(fraction, 0.0)
    return cap


@dataclass
class Plan:
    """Résultat du PLNE : nombre de livreurs par shift et heures des livraisons B2B."""

    effectifs: dict = field(default_factory=dict)  # Shift -> nombre de livreurs
    b2b_par_heure: np.ndarray = field(default_factory=lambda: np.zeros(24))
    cout: float = 0.0

    def capacite_reelle(self, prod: np.ndarray, appr: np.ndarray) -> np.ndarray:
        """Capacité du plan heure par heure, pour une productivité donnée."""
        cap = np.zeros(24)
        for shift, n in self.effectifs.items():
            cap += n * capacite(shift, prod, appr)
        return cap

    def livreurs_presents(self) -> np.ndarray:
        """Nombre de livreurs en service à chaque heure."""
        presents = np.zeros(24)
        for shift, n in self.effectifs.items():
            presents[list(shift.heures)] += n
        return presents

    @property
    def heures_livreur(self) -> float:
        return float(self.livreurs_presents().sum())


def optimiser_planning(
    b2c: np.ndarray,
    b2b_total: float,
    fenetre_b2b: tuple[int, int],
    prod: np.ndarray,
    appr: np.ndarray,
    b2b_profil_fixe: np.ndarray | None = None,
    marge: float = 0.0,
    time_limit: float = 30.0,
) -> Plan:
    """Résout le PLNE pour une zone et un jour.

    b2c             : livraisons B2C à faire à chaque heure (vecteur de 24)
    b2b_total       : volume B2B du jour
    fenetre_b2b     : heures (début, fin) où le B2B peut être livré
    prod, appr      : productivité (livraisons/h) et trajet dépôt-zone (min) à chaque heure
    b2b_profil_fixe : None si l'optimiseur choisit les heures B2B, sinon la répartition imposée
    marge           : sur-effectif de sécurité appliqué à la demande (0.10 = +10 %)
    """
    shifts = tous_les_shifts()
    n_shifts, n_heures = len(shifts), 24
    caps = np.array([capacite(s, prod, appr) for s in shifts])  # n_shifts x 24

    # Vecteur des variables : [x (un par shift), y (un par heure)].
    cout = np.concatenate([[s.cout for s in shifts], np.zeros(n_heures)])
    entiers = np.concatenate([np.ones(n_shifts), np.zeros(n_heures)])

    # Couverture : caps^T x - (1 + marge) * y >= (1 + marge) * b2c
    a_couverture = np.hstack([caps.T, -(1 + marge) * np.eye(n_heures)])
    contraintes = [LinearConstraint(a_couverture, lb=(1 + marge) * b2c, ub=np.inf)]

    y_min, y_max = np.zeros(n_heures), np.zeros(n_heures)
    if b2b_profil_fixe is None:
        debut, fin = fenetre_b2b
        y_max[debut:fin] = np.inf
        a_total = np.concatenate([np.zeros(n_shifts), np.ones(n_heures)])[None, :]
        contraintes.append(LinearConstraint(a_total, lb=b2b_total, ub=b2b_total))
    else:
        y_min = y_max = b2b_total * b2b_profil_fixe / b2b_profil_fixe.sum()

    bornes = Bounds(np.concatenate([np.zeros(n_shifts), y_min]),
                    np.concatenate([np.full(n_shifts, np.inf), y_max]))
    res = milp(cout, constraints=contraintes, integrality=entiers, bounds=bornes,
               options={"time_limit": time_limit, "disp": False})
    if res.x is None:
        raise RuntimeError(f"Le PLNE n'a pas de solution : {res.message}")

    x = np.round(res.x[:n_shifts]).astype(int)
    plan = Plan(
        effectifs={shifts[i]: int(x[i]) for i in range(n_shifts) if x[i] > 0},
        b2b_par_heure=res.x[n_shifts:],
    )
    plan.cout = sum(s.cout * n for s, n in plan.effectifs.items())
    return plan
