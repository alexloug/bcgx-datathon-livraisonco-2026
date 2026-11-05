"""
Stream D · D1 — Du trafic prédit au temps de parcours.

Entrée : le CSV de prédictions du livrable 1 (format imposé par le brief)
    arc ; datetime ; debit_horaire ; taux_occupation
Sortie : vitesse de parcours (km/h) et productivité livreur (stops/h) par arc et par heure.

Physique utilisée (boucles électromagnétiques) :
    vitesse ponctuelle v = q · L_eff / τ
    q   = débit horaire (véh/h)
    τ   = taux d'occupation (fraction du temps où la boucle est occupée)
    L_eff = longueur effective véhicule + boucle (~5,5 m en ville)
La vitesse au capteur n'intègre ni feux ni intersections : on applique un
coefficient urbain ALPHA pour obtenir une vitesse de parcours "porte à porte".
ALPHA et L_EFF sont des hypothèses à calibrer (traces GPS livreurs / API TomTom-HERE).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --- Hypothèses physiques (à calibrer en phase 2) -------------------------
L_EFF_M = 5.5          # longueur effective véhicule + boucle, en mètres
ALPHA_URBAIN = 0.6     # vitesse de parcours / vitesse au capteur (feux, carrefours)
V_MIN, V_MAX = 4.0, 50.0  # bornes physiques (km/h) pour neutraliser les valeurs aberrantes

ARCS = ["Champs-Elysées", "Convention", "Saint-Pères"]


def vitesse_capteur_kmh(debit: pd.Series, taux_occupation_pct: pd.Series) -> pd.Series:
    """Vitesse ponctuelle au capteur : v = q · L_eff / τ (km/h)."""
    tau = (taux_occupation_pct / 100.0).clip(lower=1e-3)
    v = debit * (L_EFF_M / 1000.0) / tau
    return v.clip(V_MIN, V_MAX)


def vitesse_parcours_kmh(debit: pd.Series, taux_occupation_pct: pd.Series) -> pd.Series:
    """Vitesse de parcours effective d'un véhicule de livraison (km/h)."""
    return vitesse_capteur_kmh(debit, taux_occupation_pct) * ALPHA_URBAIN


# --- Calage relatif (utilisé sur les vraies prédictions) -------------------
# Sur les capteurs parisiens, le taux d'occupation est souvent très bas : la
# formule v = q·L/τ avec L = 5,5 m sature (> 50 km/h) sur certains arcs. Le calage
# absolu de L demandera des traces GPS (phase 2). En attendant, on ne garde que
# la variation relative de q/τ dans la semaine, ancrée sur la vitesse libre :
#   v_parcours = V_LIBRE × ALPHA × min(1, (q/τ) / quantile_95%(q/τ) de l'arc)
V_LIBRE_KMH = 30.0      # vitesse limite à Paris (30 km/h depuis 2021)
QUANTILE_LIBRE = 0.95   # niveau de q/τ considéré comme « circulation libre »


def vitesse_parcours_relative(df: pd.DataFrame) -> pd.Series:
    """Vitesse de parcours (km/h) à partir de la variation relative de q/τ par arc."""
    ratio = df["debit_horaire"] / (df["taux_occupation"] / 100.0).clip(lower=1e-3)
    ref = ratio.groupby(df["arc"]).transform(lambda s: s.quantile(QUANTILE_LIBRE))
    return (V_LIBRE_KMH * ALPHA_URBAIN * (ratio / ref).clip(upper=1.0)).clip(lower=V_MIN)


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Lit le fichier output_<groupe>.csv produit par les streams B/C."""
    df = pd.read_csv(path, sep=";", index_col=0)
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M")
    return df


# --- Prédictions de substitution (tant que le CSV de l'équipe n'est pas prêt) ---
# Profils horaires normalisés (0-24h) par type de jour, inspirés des profils
# typiques des axes parisiens : pointe du matin 8-9h, pointe du soir 17-18h.
_Q = {
    "ouvre": [0.25, 0.18, 0.14, 0.12, 0.13, 0.20, 0.40, 0.70, 0.95, 0.90, 0.80, 0.80,
              0.82, 0.80, 0.80, 0.85, 0.92, 1.00, 0.98, 0.85, 0.70, 0.60, 0.50, 0.38],
    "samedi": [0.35, 0.28, 0.22, 0.16, 0.12, 0.12, 0.18, 0.30, 0.45, 0.60, 0.72, 0.80,
               0.85, 0.85, 0.88, 0.92, 0.95, 0.95, 0.90, 0.80, 0.70, 0.62, 0.55, 0.45],
    "dimanche": [0.40, 0.32, 0.26, 0.20, 0.14, 0.12, 0.12, 0.16, 0.25, 0.38, 0.50, 0.60,
                 0.62, 0.60, 0.62, 0.66, 0.70, 0.72, 0.70, 0.62, 0.52, 0.45, 0.38, 0.30],
}
_TAU = {
    "ouvre": [0.10, 0.08, 0.06, 0.05, 0.06, 0.09, 0.20, 0.50, 0.90, 0.75, 0.55, 0.55,
              0.60, 0.55, 0.55, 0.62, 0.80, 1.00, 0.95, 0.70, 0.45, 0.35, 0.28, 0.18],
    "samedi": [0.12, 0.10, 0.08, 0.06, 0.05, 0.05, 0.07, 0.12, 0.20, 0.32, 0.45, 0.55,
               0.60, 0.60, 0.65, 0.72, 0.78, 0.75, 0.65, 0.50, 0.38, 0.30, 0.25, 0.18],
    "dimanche": [0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.04, 0.06, 0.10, 0.16, 0.24, 0.30,
                 0.32, 0.30, 0.32, 0.36, 0.40, 0.42, 0.38, 0.30, 0.22, 0.18, 0.15, 0.12],
}
# (débit max véh/h, τ min %, τ max %) par arc — ordres de grandeur plausibles
_ARC_PARAMS = {
    "Champs-Elysées": (1400, 1.5, 45.0),
    "Convention": (800, 0.5, 32.0),
    "Saint-Pères": (600, 0.4, 26.0),
}
JOURS_FERIES = {pd.Timestamp("2026-11-11").date()}


def type_de_jour(ts: pd.Timestamp) -> str:
    if ts.date() in JOURS_FERIES or ts.dayofweek == 6:
        return "dimanche"  # un férié se comporte comme un dimanche
    if ts.dayofweek == 5:
        return "samedi"
    return "ouvre"


def synthetic_predictions(start: str = "2026-11-07", days: int = 5, seed: int = 0) -> pd.DataFrame:
    """Génère 3 arcs × (24·days) heures au format exact du livrable 1."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range(start, periods=24 * days, freq="h")
    rows = []
    for arc, (qmax, tmin, tmax) in _ARC_PARAMS.items():
        for ts in hours:
            jt = type_de_jour(ts)
            q = qmax * _Q[jt][ts.hour] * rng.normal(1, 0.04)
            tau = (tmin + (tmax - tmin) * _TAU[jt][ts.hour]) * rng.normal(1, 0.05)
            rows.append((arc, ts, float(q), float(tau)))
    return pd.DataFrame(rows, columns=["arc", "datetime", "debit_horaire", "taux_occupation"])


def enrich(df: pd.DataFrame, calage: str = "physique") -> pd.DataFrame:
    """Ajoute vitesses et type de jour au DataFrame de prédictions.

    calage = "physique" : v = q·L/τ × ALPHA (profils simulés)
    calage = "relatif"  : variation relative de q/τ ancrée sur 30 km/h (vraies prédictions)
    """
    out = df.copy()
    out["v_capteur_kmh"] = vitesse_capteur_kmh(out["debit_horaire"], out["taux_occupation"])
    if calage == "relatif":
        out["v_parcours_kmh"] = vitesse_parcours_relative(out)
    else:
        out["v_parcours_kmh"] = vitesse_parcours_kmh(out["debit_horaire"], out["taux_occupation"])
    out["date"] = out["datetime"].dt.date
    out["heure"] = out["datetime"].dt.hour
    out["type_jour"] = out["datetime"].apply(type_de_jour)
    return out
