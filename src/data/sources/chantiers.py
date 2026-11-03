"""Chantiers sur la voirie parisienne (open data Paris) -> data/chantiers/.

- chantiers_perturbants.parquet : chantiers à fort impact sur la circulation, avec niveau de perturbation
  et impact (ex. RESTREINTE). Petit jeu (~100 lignes), dates 2018 -> 2031.
- chantiers_a_paris.parquet : toutes les emprises de chantier en cours ou à venir (~6 000 lignes).
- chantiers_2024.parquet : emprises de 2024 (seule année archivée publiée).

Les coordonnées (lat, lon) servent à garder les chantiers proches de nos arcs dans la phase features.

Usage : python -m src.data.sources.chantiers
"""

import pandas as pd

from src.config import CHANTIERS_DIR
from src.data.sources.opendatasoft import PARIS, export_csv

DATASETS = {
    "chantiers_perturbants": "chantiers-perturbants",
    "chantiers_a_paris": "chantiers-a-paris",
    "chantiers_2024": "historique-chantiers-a-paris-en-2024",
}


def fetch(dataset: str) -> pd.DataFrame:
    df = export_csv(PARIS, dataset).drop(columns=["geo_shape"], errors="ignore")
    latlon = df.pop("geo_point_2d").str.split(",", expand=True)
    df["lat"] = pd.to_numeric(latlon[0], errors="coerce")
    df["lon"] = pd.to_numeric(latlon[1], errors="coerce")
    for col in ["date_debut", "date_fin"]:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df


def update() -> None:
    CHANTIERS_DIR.mkdir(parents=True, exist_ok=True)
    for name, dataset in DATASETS.items():
        df = fetch(dataset)
        df.to_parquet(CHANTIERS_DIR / f"{name}.parquet", index=False)
        print(f"chantier {name:20s} {len(df)} lignes, {df['date_debut'].min()} -> {df['date_fin'].max()} -> {name}.parquet")


def load(name: str = "chantiers_perturbants") -> pd.DataFrame:
    return pd.read_parquet(CHANTIERS_DIR / f"{name}.parquet")


if __name__ == "__main__":
    update()
