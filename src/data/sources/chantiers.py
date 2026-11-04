"""Chantiers sur la voirie parisienne (open data Paris) -> data/chantiers/.

- chantiers_perturbants.parquet : chantiers à fort impact sur la circulation, avec niveau de perturbation
  et impact (ex. RESTREINTE). Petit jeu (~100 lignes) en cours ou à venir, sans historique.
- chantiers_a_paris.parquet : toutes les emprises de chantier en cours ou à venir (~6 000 lignes).
- historique/<année>.parquet : emprises actives pendant l'année (archives annuelles 2019 -> 2024, même schéma).

Couverture des emprises : complète de 2019 à 2024 et à partir du jour du téléchargement ; entre les deux
(01/2025 -> J), seuls les chantiers encore en cours au téléchargement sont connus (pas d'archive 2025 publiée).
`load_emprises` assemble archives et jeu courant ; `coverage_gap` donne la période incomplète.

Usage : python -m src.data.sources.chantiers
"""

import pandas as pd

from src.config import CHANTIERS_DIR
from src.data.sources.opendatasoft import PARIS, export_csv

DATASETS = {
    "chantiers_perturbants": "chantiers-perturbants",
    "chantiers_a_paris": "chantiers-a-paris",
}
# Les archives annuelles sont publiées sous des identifiants peu parlants (copies du jeu courant).
HISTORIQUE = {
    2019: "chantiers-a-paris-copie",
    2020: "chantiers-a-paris-copie0",
    2021: "chantiers-a-paris-copie1",
    2022: "chantiers-a-paris-copie2",
    2023: "chantiers-a-paris-copie3",
    2024: "historique-chantiers-a-paris-en-2024",
}
HISTORIQUE_DIR = CHANTIERS_DIR / "historique"


def fetch(dataset: str) -> pd.DataFrame:
    df = export_csv(PARIS, dataset).drop(columns=["geo_shape"], errors="ignore")
    latlon = df.pop("geo_point_2d").str.split(",", expand=True)
    df["lat"] = pd.to_numeric(latlon[0], errors="coerce")
    df["lon"] = pd.to_numeric(latlon[1], errors="coerce")
    for col in ["date_debut", "date_fin"]:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df


def update(force: bool = False) -> None:
    HISTORIQUE_DIR.mkdir(parents=True, exist_ok=True)
    for name, dataset in DATASETS.items():
        df = fetch(dataset)
        df.to_parquet(CHANTIERS_DIR / f"{name}.parquet", index=False)
        print(f"chantier {name:20s} {len(df)} lignes, {df['date_debut'].min()} -> {df['date_fin'].max()} -> {name}.parquet")
    for year, dataset in HISTORIQUE.items():
        path = HISTORIQUE_DIR / f"{year}.parquet"
        if path.exists() and not force:
            continue
        df = fetch(dataset)
        df.to_parquet(path, index=False)
        print(f"chantier historique {year}  {len(df)} lignes -> historique/{year}.parquet")


def load(name: str = "chantiers_perturbants") -> pd.DataFrame:
    return pd.read_parquet(CHANTIERS_DIR / f"{name}.parquet")


def load_emprises() -> pd.DataFrame:
    """Toutes les emprises connues (archives + jeu courant), une ligne par `num_emprise` (la plus récente)."""
    frames = [pd.read_parquet(p) for p in sorted(HISTORIQUE_DIR.glob("*.parquet"))]
    frames.append(load("chantiers_a_paris"))
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates("num_emprise", keep="last").reset_index(drop=True)


def coverage_gap() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Période (dates de Paris, bornes incluses) où les emprises sont incomplètes : après la dernière archive,
    avant le jour où le jeu courant a été téléchargé."""
    last_year = max(int(p.stem) for p in HISTORIQUE_DIR.glob("*.parquet"))
    downloaded = pd.Timestamp((CHANTIERS_DIR / "chantiers_a_paris.parquet").stat().st_mtime, unit="s")
    return pd.Timestamp(f"{last_year + 1}-01-01"), downloaded.normalize() - pd.Timedelta(days=1)


if __name__ == "__main__":
    update()
