"""Contrôles de l'historique des arcs similaires (data/trafic/similaires/) -> data/arcs/arcs_similaires_retenus.csv.

1. Doublons : deux arcs dont le débit est identique sur au moins SAME_COUNTER des heures communes partagent un
   compteur. On garde le mieux couvert depuis 2021 (puis le meilleur score), l'autre est marqué `doublon_de`.
2. Couverture : part d'heures valides (mêmes règles que build_dataset.clean_targets) par année, depuis TRAIN_START.
3. Ruptures : débit moyen par année rapporté à celui des 12 derniers mois ; un écart > RUPTURE est signalé.

Usage : python -m src.data.controle_similaires
"""

import pandas as pd

from src.config import DATA_DIR, TARGETS, TRAIN_START, TZ
from src.data.sources.trafic_similaires import ARCS_PATH, EXCLUS, load

SAME_COUNTER = 0.9      # même seuil que src/data/arcs_similaires.py
MIN_COUVERTURE = 0.7    # part d'heures valides depuis TRAIN_START en dessous de laquelle l'arc n'est pas retenu
RUPTURE = 0.25          # écart de niveau annuel signalé
OUT_PATH = DATA_DIR / "arcs" / "arcs_similaires_retenus.csv"


def valides(raw: pd.DataFrame) -> pd.DataFrame:
    ok = raw[TARGETS].notna().all(axis=1) & ~raw["etat_arc"].isin(["Invalide", "Barré"])
    return raw[ok]


def doublons(df: pd.DataFrame, ordre: list[str]) -> dict[str, str]:
    """{arc écarté: arc gardé}, en parcourant les arcs du meilleur au moins bon rang."""
    w = df.pivot_table(index="datetime", columns="arc_id", values="debit_horaire")
    gardes, ecartes = [], {}
    for a in ordre:
        if a not in w:
            continue
        for g in gardes:
            both = w[[a, g]].dropna()
            if len(both) > 500 and (both[a] == both[g]).mean() >= SAME_COUNTER:
                ecartes[a] = g
                break
        else:
            gardes.append(a)
    return ecartes


def main() -> None:
    arcs = pd.read_csv(ARCS_PATH, sep=";", dtype={"arc_id": str})
    arcs = arcs[~arcs["arc_id"].isin(EXCLUS)]
    start = pd.Timestamp(TRAIN_START, tz=TZ).tz_convert("UTC")
    raw = load()
    raw = raw[raw["datetime"] >= start]
    ok = valides(raw)

    grille = pd.date_range(start, raw["datetime"].max(), freq="h")
    n_heures = len(grille)
    heures_an = pd.Series(grille.tz_convert(TZ).year).value_counts()
    ok = ok.assign(annee=ok["datetime"].dt.tz_convert(TZ).dt.year)
    couv = ok.groupby("arc_id").size().div(n_heures).rename("couverture_2021")
    par_an = ok.pivot_table(index="arc_id", columns="annee", values="datetime", aggfunc="size")
    couv_an = par_an.div(heures_an, axis=1)[par_an.columns].round(2).add_prefix("couv_")

    recent = ok[ok["datetime"] >= ok["datetime"].max() - pd.Timedelta("365D")]
    niveau_recent = recent.groupby("arc_id")["debit_horaire"].mean()
    niveau_an = ok.pivot_table(index="arc_id", columns="annee", values="debit_horaire", aggfunc="mean")
    niveau_an = niveau_an.where(par_an >= 24 * 60)   # au moins 60 jours valides dans l'année pour juger
    ecart_an = niveau_an.div(niveau_recent, axis=0) - 1
    ruptures = ecart_an.abs().gt(RUPTURE).apply(
        lambda r: " ".join(f"{a}:{ecart_an.loc[r.name, a]:+.0%}" for a in r.index[r]), axis=1).rename("ruptures")

    # Deux doublons partagent un compteur : on garde le mieux couvert (à couverture égale, le meilleur score).
    ordre = (arcs.assign(c=arcs["arc_id"].map(couv).fillna(0)).sort_values(["c", "score"], ascending=[False, True])
             ["arc_id"].drop_duplicates().tolist())
    dbl = doublons(ok, ordre)

    out = arcs.merge(couv, left_on="arc_id", right_index=True, how="left") \
              .merge(couv_an, left_on="arc_id", right_index=True, how="left") \
              .merge(ruptures, left_on="arc_id", right_index=True, how="left")
    out["doublon_de"] = out["arc_id"].map(dbl)
    out["couverture_2021"] = out["couverture_2021"].fillna(0).round(3)
    out["retenu"] = out["doublon_de"].isna() & (out["couverture_2021"] >= MIN_COUVERTURE)
    cols = ["arc_brief", "rang", "arc_id", "libelle", "score", "percentile_feries", "couverture_2021",
            *couv_an.columns, "ruptures", "doublon_de", "retenu"]
    out[cols].to_csv(OUT_PATH, sep=";", index=False)

    pd.set_option("display.width", 250)
    print(out[cols].to_string(index=False))
    print(f"\ndoublons : {dbl}")
    r = out[out["retenu"]]
    print(f"retenus : {r['arc_id'].nunique()} arcs distincts ; par arc du brief : {r.groupby('arc_brief').size().to_dict()}")
    print(f"lignes valides retenues : {ok[ok['arc_id'].isin(r['arc_id'])].shape[0]:,} -> {OUT_PATH.name}")


if __name__ == "__main__":
    main()
