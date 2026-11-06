"""Stream D : convertit les prédictions de trafic (livrable 1) en vitesses de parcours.

Entrée : le CSV de rendu (colonnes arc, datetime, debit_horaire, taux_occupation).
Sortie : une vitesse de parcours en km/h par arc et par heure, utilisée par staffing.py.

Deux calages sont disponibles :
- "physique" : vitesse au capteur v = q * L / tau, réduite d'un coefficient urbain.
- "relatif"  : variation de q / tau dans la semaine, ancrée sur la vitesse libre.
  À utiliser sur les vraies prédictions, où le calage physique sature (voir plus bas).
"""

from pathlib import Path

import numpy as np
import pandas as pd

ARCS = ["Champs-Elysées", "Convention", "Saint-Pères"]
JOURS_FERIES = {pd.Timestamp("2026-11-11").date()}

# Hypothèses physiques, à caler sur les traces GPS des livreurs.
L_EFF_M = 5.5          # longueur d'un véhicule + longueur de la boucle de comptage (m)
ALPHA_URBAIN = 0.6     # vitesse de parcours / vitesse au capteur (feux, carrefours)
V_MIN, V_MAX = 4.0, 50.0  # bornes en km/h pour écarter les valeurs aberrantes

# Calage relatif. Sur les vraies prédictions, le taux d'occupation de la Convention
# reste sous 6 % en journée : avec L = 5,5 m, la vitesse calculée dépasse 50 km/h.
# On ne garde donc que la variation de q / tau par arc, rapportée à son niveau
# en circulation libre, et ancrée sur la vitesse limite à Paris.
V_LIBRE_KMH = 30.0      # vitesse limite à Paris depuis 2021
QUANTILE_LIBRE = 0.95   # niveau de q / tau considéré comme circulation libre


def vitesse_capteur_kmh(debit: pd.Series, taux_occupation_pct: pd.Series) -> pd.Series:
    """Vitesse au capteur v = q * L / tau, en km/h, bornée entre V_MIN et V_MAX."""
    tau = (taux_occupation_pct / 100.0).clip(lower=1e-3)
    return (debit * (L_EFF_M / 1000.0) / tau).clip(V_MIN, V_MAX)


def vitesse_parcours_kmh(debit: pd.Series, taux_occupation_pct: pd.Series) -> pd.Series:
    """Vitesse de parcours (calage physique) : vitesse au capteur * ALPHA_URBAIN."""
    return vitesse_capteur_kmh(debit, taux_occupation_pct) * ALPHA_URBAIN


def vitesse_parcours_relative(df: pd.DataFrame) -> pd.Series:
    """Vitesse de parcours (calage relatif), en km/h.

    v = V_LIBRE_KMH * ALPHA_URBAIN * min(1, (q / tau) / quantile 95 % de q / tau sur l'arc)
    """
    ratio = df["debit_horaire"] / (df["taux_occupation"] / 100.0).clip(lower=1e-3)
    niveau_libre = ratio.groupby(df["arc"]).transform(lambda s: s.quantile(QUANTILE_LIBRE))
    return (V_LIBRE_KMH * ALPHA_URBAIN * (ratio / niveau_libre).clip(upper=1.0)).clip(lower=V_MIN)


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Lit un CSV au format du rendu (séparateur ;, première colonne = index)."""
    df = pd.read_csv(path, sep=";", index_col=0)
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M")
    return df


def type_de_jour(ts: pd.Timestamp) -> str:
    """Type de jour pour les profils de trafic. Un férié se comporte comme un dimanche."""
    if ts.date() in JOURS_FERIES or ts.dayofweek == 6:
        return "dimanche"
    if ts.dayofweek == 5:
        return "samedi"
    return "ouvre"


# Profils horaires normalisés (0 h -> 23 h) utilisés pour simuler des prédictions
# tant que le modèle de l'équipe n'était pas prêt : pointes vers 8 h et 17-18 h.
_PROFIL_DEBIT = {
    "ouvre": [0.25, 0.18, 0.14, 0.12, 0.13, 0.20, 0.40, 0.70, 0.95, 0.90, 0.80, 0.80,
              0.82, 0.80, 0.80, 0.85, 0.92, 1.00, 0.98, 0.85, 0.70, 0.60, 0.50, 0.38],
    "samedi": [0.35, 0.28, 0.22, 0.16, 0.12, 0.12, 0.18, 0.30, 0.45, 0.60, 0.72, 0.80,
               0.85, 0.85, 0.88, 0.92, 0.95, 0.95, 0.90, 0.80, 0.70, 0.62, 0.55, 0.45],
    "dimanche": [0.40, 0.32, 0.26, 0.20, 0.14, 0.12, 0.12, 0.16, 0.25, 0.38, 0.50, 0.60,
                 0.62, 0.60, 0.62, 0.66, 0.70, 0.72, 0.70, 0.62, 0.52, 0.45, 0.38, 0.30],
}
_PROFIL_OCCUPATION = {
    "ouvre": [0.10, 0.08, 0.06, 0.05, 0.06, 0.09, 0.20, 0.50, 0.90, 0.75, 0.55, 0.55,
              0.60, 0.55, 0.55, 0.62, 0.80, 1.00, 0.95, 0.70, 0.45, 0.35, 0.28, 0.18],
    "samedi": [0.12, 0.10, 0.08, 0.06, 0.05, 0.05, 0.07, 0.12, 0.20, 0.32, 0.45, 0.55,
               0.60, 0.60, 0.65, 0.72, 0.78, 0.75, 0.65, 0.50, 0.38, 0.30, 0.25, 0.18],
    "dimanche": [0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.04, 0.06, 0.10, 0.16, 0.24, 0.30,
                 0.32, 0.30, 0.32, 0.36, 0.40, 0.42, 0.38, 0.30, 0.22, 0.18, 0.15, 0.12],
}
# Par arc : (débit max en véh/h, occupation min en %, occupation max en %).
_PARAMETRES_ARC = {
    "Champs-Elysées": (1400, 1.5, 45.0),
    "Convention": (800, 0.5, 32.0),
    "Saint-Pères": (600, 0.4, 26.0),
}


def synthetic_predictions(start: str = "2026-11-07", days: int = 5, seed: int = 0) -> pd.DataFrame:
    """Prédictions simulées au format du rendu (3 arcs x 24 h x days), avec un bruit de 4-5 %."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range(start, periods=24 * days, freq="h")
    rows = []
    for arc, (debit_max, occ_min, occ_max) in _PARAMETRES_ARC.items():
        for ts in hours:
            jour = type_de_jour(ts)
            debit = debit_max * _PROFIL_DEBIT[jour][ts.hour] * rng.normal(1, 0.04)
            occupation = (occ_min + (occ_max - occ_min) * _PROFIL_OCCUPATION[jour][ts.hour]) * rng.normal(1, 0.05)
            rows.append((arc, ts, float(debit), float(occupation)))
    return pd.DataFrame(rows, columns=["arc", "datetime", "debit_horaire", "taux_occupation"])


def enrich(df: pd.DataFrame, calage: str = "physique") -> pd.DataFrame:
    """Ajoute les vitesses (capteur et parcours), la date, l'heure et le type de jour."""
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
