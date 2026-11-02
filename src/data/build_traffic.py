"""Stream A : construit data/processed/traffic_hourly.parquet à partir des exports open data.

Sortie (une ligne par arc et par heure, index horaire continu) :
    arc, datetime (UTC, tz-aware), debit_horaire, taux_occupation, is_valid, etat_arc

Usage : python -m src.data.build_traffic
"""

import pandas as pd

from src.config import ARCS, RAW_DIR, RECENT_DIR, TRAFFIC_PATH, TARGETS

# Les exports API et les archives annuelles n'ont pas les mêmes en-têtes.
COLUMN_MAP = {
    "Date et heure de comptage": "t_1h",
    "Débit horaire": "q",
    "Taux d'occupation": "k",
    "Libelle noeud amont": "libelle_nd_amont",
    "Libelle noeud aval": "libelle_nd_aval",
    "Etat arc": "etat_barre",
}


def read_raw(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)
    return df.rename(columns=COLUMN_MAP)


def filter_arc(raw: pd.DataFrame, arc: str) -> pd.DataFrame:
    spec = ARCS[arc]
    df = raw[(raw["libelle_nd_amont"] == spec["nd_amont"]) & (raw["libelle_nd_aval"] == spec["nd_aval"])]
    out = pd.DataFrame({
        "arc": arc,
        "datetime": pd.to_datetime(df["t_1h"], utc=True),
        "debit_horaire": pd.to_numeric(df["q"], errors="coerce"),
        "taux_occupation": pd.to_numeric(df["k"], errors="coerce"),
        "etat_arc": df["etat_barre"],
    })
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


def raw_path(spec: dict):
    """Export frais de fetch_recent.py s'il existe, sinon l'instantané versionné dans data/."""
    recent = RECENT_DIR / spec["raw_file"]
    return recent if recent.exists() else RAW_DIR / spec["raw_file"]


def build() -> pd.DataFrame:
    paths = {arc: raw_path(spec) for arc, spec in ARCS.items()}
    for arc, path in paths.items():
        print(f"{arc} <- {path.relative_to(RAW_DIR.parent)}")
    parts = [filter_arc(read_raw(path), arc) for arc, path in paths.items()]
    # TODO stream A : ajouter ici les archives annuelles (data/archives/) avant le réindexage.
    return to_hourly_grid(pd.concat(parts, ignore_index=True))


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
