"""Erreur du modèle retenu et valeur de ses prédictions pour choisir un créneau de livraison.

Entrées : prédictions heure par heure du backtest (23 cutoffs) de `xgb` (outputs/selection_features_valider.parquet,
variante `complet`) et de la baseline (moyenne des 4 dernières semaines, recalculée ici).

1. Erreur : RMSE 24 h et 8 h-20 h, rapportées au niveau moyen, comparées à la baseline.
2. Concentration : part de l'erreur² portée par les 10 % pires jours.
3. Décision : chaque jour, le livreur choisit un créneau de 4 h entre 8 h et 20 h (9 débuts possibles). Le coût d'un
   créneau est son indice de temps de parcours réel : (occupation / débit) rapporté à sa valeur de nuit
   (docs/reformulation_probleme.md §2 ; 1 = fluide, 2 = deux fois plus long). On compare le créneau choisi par :
   - l'oracle (le meilleur créneau réel, inatteignable) ;
   - le modèle (meilleur créneau prédit) ;
   - la baseline (meilleur créneau d'après la moyenne des 4 dernières semaines) ;
   - l'habitude (créneau fixe par arc et type de jour, le meilleur en moyenne sur 2021-2022, avant tout cutoff).

Usage : python -m src.analyse_business   -> outputs/analyse_business_*.csv
"""

import numpy as np
import pandas as pd

from src.config import DATASET_MODELE_PATH, OUTPUTS_DIR, TARGETS, TZ
from src.evaluate import backtest
from src.selection_features import cutoffs

SLOT_H = 4
JOUR = (8, 20)
DEBUTS = range(JOUR[0], JOUR[1] - SLOT_H + 1)
HABITUDE_FIN = "2022-12-31"


def predictions(data: pd.DataFrame) -> pd.DataFrame:
    xgb = pd.read_parquet(OUTPUTS_DIR / "selection_features_valider.parquet")
    xgb = xgb[xgb["variante"] == "complet"].drop(columns="variante")
    base = pd.concat([backtest("baseline", data, c).assign(jeu=j) for j, c in cutoffs(data).items()])
    base = base[["arc", "datetime", "cutoff", *[f"{t}_pred" for t in TARGETS]]]
    base.columns = ["arc", "datetime", "cutoff", *[f"{t}_base" for t in TARGETS]]
    p = xgb.merge(base, on=["arc", "datetime", "cutoff"], how="left")
    local = p["datetime"].dt.tz_convert(TZ)
    return p.assign(heure=local.dt.hour, jour=local.dt.date, dow=local.dt.dayofweek)


def erreurs(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arc, g in p.groupby("arc"):
        jour = g[g["heure"].between(JOUR[0], JOUR[1] - 1)]
        for t in TARGETS:
            e, eb = g[t] - g[f"{t}_pred"], g[t] - g[f"{t}_base"]
            ej = jour[t] - jour[f"{t}_pred"]
            d2 = (e ** 2).groupby(g["jour"]).sum().sort_values(ascending=False)
            rows.append({
                "arc": arc, "cible": t, "niveau_moyen_24h": g[t].mean(), "niveau_moyen_8_20h": jour[t].mean(),
                "rmse_24h": np.sqrt((e ** 2).mean()), "rmse_baseline_24h": np.sqrt((eb ** 2).mean()),
                "rmse_8_20h": np.sqrt((ej ** 2).mean()), "mae_8_20h": ej.abs().mean(),
                "erreur_relative_8_20h": ej.abs().mean() / jour[t].mean(),
                "gain_vs_baseline": 1 - np.sqrt((e ** 2).mean()) / np.sqrt((eb ** 2).mean()),
                "part_erreur2_10pc_pires_jours": d2.head(max(1, len(d2) // 10)).sum() / d2.sum(),
            })
    return pd.DataFrame(rows).round(3)


def reference_nuit(data: pd.DataFrame) -> pd.Series:
    v = data[data["is_valid"] & (data["debit_horaire"] > 0)]
    h = v["datetime"].dt.tz_convert(TZ).dt.hour
    v = v[(h >= 1) & (h <= 4)]
    return (v["taux_occupation"] / v["debit_horaire"]).groupby(v["arc"]).median()


def creneaux(df: pd.DataFrame, occ: str, debit: str, ref: pd.Series) -> pd.DataFrame:
    """Indice moyen de chaque créneau de 4 h, par (arc, jour) ; NaN si une heure manque."""
    tti = (df[occ] / df[debit].clip(lower=1)) / df["arc"].map(ref)
    w = tti.groupby([df["arc"], df["jour"], df["heure"]]).mean().unstack("heure")
    out = {d: w.reindex(columns=range(d, d + SLOT_H)).mean(axis=1, skipna=False) for d in DEBUTS}
    return pd.DataFrame(out)


def habitude(data: pd.DataFrame, ref: pd.Series) -> pd.Series:
    """Créneau fixe par (arc, type de jour) : le meilleur en moyenne avant 2023."""
    h = data[data["is_valid"] & (data["datetime"] <= pd.Timestamp(HABITUDE_FIN, tz=TZ))].copy()
    local = h["datetime"].dt.tz_convert(TZ)
    h = h.assign(heure=local.dt.hour, jour=local.dt.date, dow=local.dt.dayofweek)
    c = creneaux(h, "taux_occupation", "debit_horaire", ref)
    dow = pd.Series(pd.to_datetime(c.index.get_level_values("jour")).dayofweek, index=c.index)
    type_jour = np.select([dow == 5, dow == 6], ["samedi", "dimanche"], "semaine")
    moy = c.groupby([c.index.get_level_values("arc"), type_jour]).mean()
    return moy.idxmin(axis=1)


def decisions(p: pd.DataFrame, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = reference_nuit(data)
    reel = creneaux(p, "taux_occupation", "debit_horaire", ref).dropna()
    pred = creneaux(p, "taux_occupation_pred", "debit_horaire_pred", ref).reindex(reel.index)
    base = creneaux(p, "taux_occupation_base", "debit_horaire_base", ref).reindex(reel.index)
    hab = habitude(data, ref)
    dow = pd.to_datetime(reel.index.get_level_values("jour")).dayofweek
    type_jour = np.select([dow == 5, dow == 6], ["samedi", "dimanche"], "semaine")
    choix = pd.DataFrame({
        "oracle": reel.idxmin(axis=1),
        "modele": pred.idxmin(axis=1),
        "baseline": base.idxmin(axis=1),
        "habitude": [hab.get((a, t)) for a, t in zip(reel.index.get_level_values("arc"), type_jour)],
    }, index=reel.index)
    cout = pd.DataFrame({k: [reel.loc[i, s] for i, s in v.items()] for k, v in choix.items()}, index=reel.index)
    cout["pire"] = reel.max(axis=1)
    res = []
    for arc, g in cout.groupby(level="arc"):
        for pol in ["modele", "baseline", "habitude"]:
            res.append({
                "arc": arc, "politique": pol, "jours": len(g),
                "indice_moyen": g[pol].mean(), "indice_oracle": g["oracle"].mean(),
                "surcout_vs_oracle": (g[pol] / g["oracle"] - 1).mean(),
                "gain_vs_habitude": 1 - g[pol].mean() / g["habitude"].mean(),
                "bon_creneau": (choix.loc[g.index, pol] == choix.loc[g.index, "oracle"]).mean(),
                "creneau_a_20pc_du_meilleur": (g[pol] <= 1.2 * g["oracle"]).mean(),
                "ecart_pire_meilleur": (g["pire"] / g["oracle"] - 1).mean(),
            })
    return pd.DataFrame(res).round(3), cout


def main() -> None:
    data = pd.read_parquet(DATASET_MODELE_PATH)
    p = predictions(data)
    err = erreurs(p)
    dec, cout = decisions(p, data)
    err.to_csv(OUTPUTS_DIR / "analyse_business_erreurs.csv", index=False)
    dec.to_csv(OUTPUTS_DIR / "analyse_business_creneaux.csv", index=False)
    cout.to_csv(OUTPUTS_DIR / "analyse_business_creneaux_par_jour.csv")
    pd.set_option("display.width", 250)
    print(err.to_string(index=False))
    print()
    print(dec.to_string(index=False))


if __name__ == "__main__":
    main()
