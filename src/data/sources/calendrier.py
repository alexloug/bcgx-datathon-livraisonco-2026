"""Calendrier officiel -> data/calendrier/.

- jours_feries.parquet        : API calendrier.api.gouv.fr (métropole, 2006-2031).
- vacances_scolaires.parquet  : data.education.gouv.fr, académie de Paris (zone C), à partir de 2017-2018.
  Avant 2017, rien en open data structuré : à compléter à la main si on utilise les archives anciennes.

Usage : python -m src.data.sources.calendrier
"""

import pandas as pd
import requests

from src.config import CALENDRIER_DIR, TZ
from src.data.sources.opendatasoft import EDUCATION, export_csv

FERIES_URL = "https://calendrier.api.gouv.fr/jours-feries/metropole.json"


def fetch_feries() -> pd.DataFrame:
    resp = requests.get(FERIES_URL, timeout=60)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json().items(), columns=["date", "nom"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values("date").reset_index(drop=True)


def fetch_vacances() -> pd.DataFrame:
    """Une ligne par période. `debut` = premier jour sans cours, `fin` = jour de reprise (exclu), heure de Paris."""
    df = export_csv(EDUCATION, "fr-en-calendrier-scolaire", where='location="Paris"', order_by="start_date")
    # Une même période est parfois dupliquée par population (élèves / enseignants).
    df = df[df["population"].isin(["-", "Élèves"])]
    out = pd.DataFrame({
        "nom": df["description"],
        "annee_scolaire": df["annee_scolaire"],
        "debut": pd.to_datetime(df["start_date"], utc=True).dt.tz_convert(TZ).dt.date,
        "fin": pd.to_datetime(df["end_date"], utc=True).dt.tz_convert(TZ).dt.date,
        "zone": df["zones"],
    })
    return out.drop_duplicates(["nom", "debut"]).reset_index(drop=True)


def update() -> None:
    CALENDRIER_DIR.mkdir(parents=True, exist_ok=True)
    for name, df, col in [("jours_feries", fetch_feries(), "date"), ("vacances_scolaires", fetch_vacances(), "debut")]:
        path = CALENDRIER_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"calendr. {name:20s} {len(df)} lignes, {df[col].min()} -> {df[col].max()} -> {path.name}")


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(CALENDRIER_DIR / f"{name}.parquet")


if __name__ == "__main__":
    update()
