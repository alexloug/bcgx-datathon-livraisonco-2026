"""Stream A : construit data/processed/traffic_hourly.parquet à partir de data/trafic/ (archives + API).

Sortie (une ligne par arc du brief et par heure, index horaire continu) :
    arc, datetime (UTC, tz-aware), debit_horaire, taux_occupation, is_valid, etat_arc

Usage : python -m exploration.data.build_traffic   (après python -m src.data.update)
"""

import pandas as pd

from src.config import ARCS, PROCESSED_DIR, TARGETS

TRAFFIC_PATH = PROCESSED_DIR / "traffic_hourly.parquet"
from src.data.sources import trafic


def filter_arc(raw: pd.DataFrame, arc: str) -> pd.DataFrame:
    spec = ARCS[arc]
    df = raw[(raw["nd_amont"] == spec["nd_amont"]) & (raw["nd_aval"] == spec["nd_aval"])]
    out = df[["datetime", *TARGETS, "etat_arc"]].assign(arc=arc)
    return out.drop_duplicates(["arc", "datetime"], keep="last")


def to_hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Réindexe chaque arc sur une grille horaire continue (les trous deviennent des NaN)."""
    frames = []
    for arc, g in df.groupby("arc"):
        grid = pd.date_range(g["datetime"].min(), g["datetime"].max(), freq="h", tz="UTC")
        g = g.set_index("datetime").reindex(grid).rename_axis("datetime").reset_index()
        g["arc"] = arc
        frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    out["is_valid"] = out[TARGETS].notna().all(axis=1) & (out["etat_arc"] != "Invalide")
    return out[["arc", "datetime", *TARGETS, "is_valid", "etat_arc"]].sort_values(["arc", "datetime"])


def build() -> pd.DataFrame:
    raw = trafic.load()
    return to_hourly_grid(pd.concat([filter_arc(raw, arc) for arc in ARCS], ignore_index=True))


if __name__ == "__main__":
    traffic = build()
    TRAFFIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    traffic.to_parquet(TRAFFIC_PATH, index=False)
    summary = traffic.groupby("arc").agg(
        debut=("datetime", "min"), fin=("datetime", "max"), heures=("datetime", "size"),
        part_valide=("is_valid", "mean"),
        derniere_valide=("datetime", lambda s: s[traffic.loc[s.index, "is_valid"]].max()),
    )
    print(summary.to_string())
    print(f"-> {TRAFFIC_PATH}")
