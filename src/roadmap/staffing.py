"""
Stream D · D2 — Planning des livreurs pour la semaine S+1 (PLNE).

Pour chaque zone (une zone = le quartier desservi par un arc capteur) et chaque jour :

Variables
    x[t,d] ∈ ℕ   nombre de livreurs démarrant à l'heure t pour un shift de d heures (4 ≤ d ≤ 8)
    y[h]   ≥ 0   nombre de stops B2B (flexibles) livrés à l'heure h, dans leur fenêtre autorisée

Objectif
    min Σ x[t,d] · Σ_{h∈shift} tarif(h)         tarif = 10 €/h de 8h à 20h, 14 €/h sinon

Contraintes
    couverture   Σ_{shifts actifs en h} x[t,d] · cap[t,d,h]  ≥  B2C[h] + y[h]     ∀h
    B2B          Σ_h y[h] = volume B2B du jour,   y[h] = 0 hors fenêtre B2B

cap[t,d,h] = productivité horaire (stops/h) issue de la vitesse prédite, réduite
la première et la dernière heure du temps d'approche dépôt ↔ zone.
Résolution : scipy.optimize.milp (solveur HiGHS), aucune dépendance externe.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

# --- Hypothèses opérationnelles (à remplacer par les données LivraisonCo) ---
TEMPS_SERVICE_MIN = 4.0     # minutes par stop : se garer, sonner, remettre, signer
DIST_INTER_STOP_KM = 0.6    # distance moyenne entre deux stops en zone dense
DIST_DEPOT_KM = 7.0         # dépôt en petite couronne → zone de livraison
DUREES_SHIFT = range(4, 9)  # 4h minimum, 8h maximum (brief)


def tarif_horaire(h: int) -> float:
    """Coût horaire chargé d'un livreur (brief) : 10 € de 8h à 20h, 14 € la nuit."""
    return 10.0 if 8 <= h < 20 else 14.0


def productivite_stops_h(v_kmh: np.ndarray) -> np.ndarray:
    """Stops livrés par heure et par livreur, selon la vitesse de parcours."""
    return 60.0 / (TEMPS_SERVICE_MIN + 60.0 * DIST_INTER_STOP_KM / v_kmh)


def approche_min(v_kmh: np.ndarray) -> np.ndarray:
    """Minutes perdues pour rejoindre (ou quitter) la zone depuis le dépôt."""
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
    return [Shift(t, d) for d in DUREES_SHIFT for t in range(0, 24 - d + 1)]


def capacite(shift: Shift, prod: np.ndarray, appr: np.ndarray) -> np.ndarray:
    """Stops livrables par heure (vecteur 24) par un livreur sur ce shift."""
    cap = np.zeros(24)
    for h in shift.heures:
        frac = 1.0
        if h == shift.debut:
            frac -= appr[h] / 60.0
        if h == shift.debut + shift.duree - 1:
            frac -= appr[h] / 60.0
        cap[h] = prod[h] * max(frac, 0.0)
    return cap


@dataclass
class Plan:
    effectifs: dict = field(default_factory=dict)  # Shift -> nb livreurs
    b2b_par_heure: np.ndarray = field(default_factory=lambda: np.zeros(24))
    cout: float = 0.0

    def capacite_reelle(self, prod: np.ndarray, appr: np.ndarray) -> np.ndarray:
        cap = np.zeros(24)
        for s, n in self.effectifs.items():
            cap += n * capacite(s, prod, appr)
        return cap

    def livreurs_presents(self) -> np.ndarray:
        pres = np.zeros(24)
        for s, n in self.effectifs.items():
            pres[list(s.heures)] += n
        return pres

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
    """
    Résout le PLNE d'un (zone, jour).
    - b2b_profil_fixe=None  → l'optimiseur choisit les heures B2B (scénario S2)
    - sinon                 → B2B imposé selon ce profil (scénarios S0 / S1)
    - marge                 → sur-effectif de sécurité appliqué à la demande
    """
    shifts = tous_les_shifts()
    nS, H = len(shifts), 24
    caps = np.array([capacite(s, prod, appr) for s in shifts])  # nS × 24

    c = np.concatenate([[s.cout for s in shifts], np.zeros(H)])
    integrality = np.concatenate([np.ones(nS), np.zeros(H)])

    # couverture : caps^T x − (1+marge)·y ≥ (1+marge)·b2c
    A_cov = np.hstack([caps.T, -(1 + marge) * np.eye(H)])
    cons = [LinearConstraint(A_cov, lb=(1 + marge) * b2c, ub=np.inf)]

    lo_y, hi_y = np.zeros(H), np.zeros(H)
    if b2b_profil_fixe is None:
        a, b = fenetre_b2b
        hi_y[a:b] = np.inf
        A_tot = np.concatenate([np.zeros(nS), np.ones(H)])[None, :]
        cons.append(LinearConstraint(A_tot, lb=b2b_total, ub=b2b_total))
    else:
        lo_y = hi_y = b2b_total * b2b_profil_fixe / b2b_profil_fixe.sum()

    bounds = Bounds(np.concatenate([np.zeros(nS), lo_y]), np.concatenate([np.full(nS, np.inf), hi_y]))
    res = milp(c, constraints=cons, integrality=integrality, bounds=bounds,
               options={"time_limit": time_limit, "disp": False})
    if res.x is None:
        raise RuntimeError(f"PLNE sans solution : {res.message}")

    x = np.round(res.x[:nS]).astype(int)
    plan = Plan(
        effectifs={shifts[i]: int(x[i]) for i in range(nS) if x[i] > 0},
        b2b_par_heure=res.x[nS:],
    )
    plan.cout = sum(s.cout * n for s, n in plan.effectifs.items())
    return plan
