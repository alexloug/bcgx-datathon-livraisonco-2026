"""Détection des jours anormaux (travaux, déviations, fermetures…) pour les retirer de l'entraînement.

Un jour est anormal sur un arc si son débit moyen s'écarte de plus de `seuil` (ex. 30 %) de la médiane des
mêmes jours de la semaine, 4 semaines avant et 4 semaines après. On ne marque PAS les jours fériés, ponts,
vacances et événements connus : le modèle a les colonnes pour les apprendre, il faut les garder.
Les épisodes repérés à la main par l'équipe (docs/determinants_trafic.md) sont ajoutés quoi qu'il arrive.

Usage : python -m exploration.essais_victor.anomalies          (affiche les épisodes détectés)
"""

import numpy as np
import pandas as pd

from src.config import TZ

# Épisodes connus (docs/determinants_trafic.md), dates de Paris incluses
EPISODES_CONNUS = [
    ("Saint-Pères", "2026-08-28", "2026-09-20", "déviation"),
    ("Saint-Pères", "2023-05-04", "2023-05-12", "travaux / déviation"),
    ("Convention", "2022-09-05", "2022-09-12", "travaux / déviation"),
    ("Champs-Elysées", "2024-09-19", "2024-09-30", "travaux / déviation"),
]


def jours_anormaux(df: pd.DataFrame, seuil: float = 0.30, connus: bool = True) -> pd.DataFrame:
    """Table (arc, date) des jours anormaux, à partir des lignes valides de `df` (aucune donnée hors de df)."""
    v = df[df["is_valid"].astype(bool)].copy()
    v["date"] = v["datetime"].dt.tz_convert(TZ).dt.date
    jour = v.groupby(["arc", "date"]).agg(debit=("debit_horaire", "mean"), n=("debit_horaire", "size"),
                                         dow=("dow", "first"), ferie=("ferie", "max"), pont=("pont", "max"),
                                         vacances=("vacances", "max"), evenement=("evenement", "max"))
    jour = jour[jour["n"] >= 18].reset_index()       # au moins 18 heures mesurées dans la journée
    jour["date"] = pd.to_datetime(jour["date"])

    morceaux = []
    for (arc, dow), g in jour.groupby(["arc", "dow"]):
        g = g.set_index("date").sort_index()
        # médiane des mêmes jours de semaine, ±4 semaines, sans le jour lui-même
        voisins = [g["debit"].shift(k, freq="7D") for k in (-4, -3, -2, -1, 1, 2, 3, 4)]
        ref = pd.concat(voisins, axis=1, sort=True).reindex(g.index).median(axis=1)
        g["ecart"] = g["debit"] / ref - 1
        morceaux.append(g.reset_index())
    jour = pd.concat(morceaux)

    explicable = (jour[["ferie", "pont", "vacances", "evenement"]].astype(bool)).any(axis=1)
    auto = jour[(jour["ecart"].abs() > seuil) & ~explicable][["arc", "date", "ecart"]].assign(cause="détecté")

    if connus:
        manuels = pd.concat([pd.DataFrame({"arc": a, "date": pd.date_range(d, f), "cause": c})
                             for a, d, f, c in EPISODES_CONNUS])
        auto = pd.concat([auto, manuels]).drop_duplicates(["arc", "date"], keep="last")
    return auto.sort_values(["arc", "date"]).reset_index(drop=True)


# Dérives probables du capteur d'occupation (occupation +60 à +240 % à débit normal), repérées à la main
DERIVES_OCC_CONNUES = [
    ("Champs-Elysées", "2023-10-29", "2023-11-15"),
    ("Champs-Elysées", "2021-11-20", "2021-11-23"),
    ("Convention", "2022-10-03", "2022-10-06"),
]


def derives_occupation(df: pd.DataFrame, seuil_occ: float = 0.60, seuil_debit: float = 0.20) -> pd.DataFrame:
    """Jours où l'occupation s'écarte de plus de `seuil_occ` de l'habituel alors que le débit reste à moins de
    `seuil_debit` : signe d'un capteur qui dérive plutôt que d'un vrai changement de trafic. + dérives connues."""
    v = df[df["is_valid"].astype(bool)].copy()
    v["date"] = v["datetime"].dt.tz_convert(TZ).dt.date
    jour = v.groupby(["arc", "date"]).agg(occ=("taux_occupation", "mean"), debit=("debit_horaire", "mean"),
                                         n=("debit_horaire", "size"), dow=("dow", "first"))
    jour = jour[jour["n"] >= 18].reset_index()
    jour["date"] = pd.to_datetime(jour["date"])
    morceaux = []
    for _, g in jour.groupby(["arc", "dow"]):
        g = g.set_index("date").sort_index()
        for col in ("occ", "debit"):
            voisins = [g[col].shift(k, freq="7D") for k in (-4, -3, -2, -1, 1, 2, 3, 4)]
            g[f"ecart_{col}"] = g[col] / pd.concat(voisins, axis=1, sort=True).reindex(g.index).median(axis=1) - 1
        morceaux.append(g.reset_index())
    jour = pd.concat(morceaux)
    auto = jour[(jour["ecart_occ"] > seuil_occ) & (jour["ecart_debit"].abs() < seuil_debit)][["arc", "date"]]
    connues = pd.concat([pd.DataFrame({"arc": a, "date": pd.date_range(d, f)}) for a, d, f in DERIVES_OCC_CONNUES])
    return pd.concat([auto, connues]).drop_duplicates().reset_index(drop=True)


def masque(df: pd.DataFrame, anormaux: pd.DataFrame) -> np.ndarray:
    """True pour les lignes de `df` qui tombent un jour anormal de leur arc."""
    cle = df["arc"] + "|" + df["datetime"].dt.tz_convert(TZ).dt.strftime("%Y-%m-%d")
    cles = set(anormaux["arc"] + "|" + anormaux["date"].dt.strftime("%Y-%m-%d"))
    return cle.isin(cles).to_numpy()


if __name__ == "__main__":
    from exploration.essais_victor.essai_xgb import DATASET, prepare
    df = prepare(pd.read_parquet(DATASET))
    for seuil in (0.20, 0.30, 0.40):
        a = jours_anormaux(df, seuil, connus=False)
        print(f"Seuil {seuil:.0%} : {len(a)} jours détectés  ->  " +
              ", ".join(f"{arc} {n}" for arc, n in a.groupby("arc").size().items()))
    a = jours_anormaux(df, 0.30)
    a["mois"] = a["date"].dt.to_period("M")
    print("\nSeuil 30 % + épisodes connus, jours par arc et par mois (mois avec >= 3 jours) :")
    t = a.groupby(["arc", "mois"]).size()
    print(t[t >= 3].to_string())
