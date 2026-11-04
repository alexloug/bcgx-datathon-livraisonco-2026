"""Événements -> data/evenements/.

- evenements_manuels.csv (versionné, à compléter à la main) : événements qui ferment ou chargent nos rues
  et qu'aucune source ouverte ne publie (cérémonies, défilés, manifestations, arrivée du Tour de France…).
  `recurrence` = « annuelle » (répété chaque année à la même date) ou « ponctuelle ».
  Colonnes facultatives (valeur par défaut si vide ou absente) :
  `type` = « fermeture » (moins de voitures : avenue fermée) ou « perturbation » (trafic reporté ou chargé :
  grève, salon), défaut « fermeture » ;
  `heure_debut` / `heure_fin` = heures de Paris (0-23), défaut 0 / 23. L'événement couvre en continu
  date_debut heure_debut -> date_fin heure_fin, heure de fin incluse (ex. 7 -> 14 : de 07:00 à 14:59).
  -> evenements.parquet : une ligne par occurrence, années 2010 à N+1.
- que_faire_a_paris.parquet : agenda culturel de la Ville (« Que faire à Paris »). Ne contient que les
  événements encore en ligne (pas d'historique fiable) et surtout des petits événements : à évaluer.

Pas d'open data trouvé pour les manifestations (Préfecture de police) ni pour les grèves de transport.

Usage : python -m src.data.sources.evenements
"""

from datetime import date

import pandas as pd

from src.config import EVENEMENTS_DIR
from src.data.sources.opendatasoft import PARIS, export_csv

MANUAL_PATH = EVENEMENTS_DIR / "evenements_manuels.csv"
FIRST_YEAR = 2010
TYPES = ("fermeture", "perturbation")
DEFAULTS = {"type": "fermeture", "heure_debut": 0, "heure_fin": 23}


def expand_manual(df: pd.DataFrame, last_year: int) -> pd.DataFrame:
    df = df.copy()
    for col, default in DEFAULTS.items():
        df[col] = df[col].fillna(default) if col in df else default
    unknown = set(df["type"]) - set(TYPES)
    assert not unknown, f"type inconnu dans {MANUAL_PATH.name} : {unknown} (attendu : {TYPES})"
    rows = []
    for ev in df.itertuples(index=False):
        start, end = pd.Timestamp(ev.date_debut), pd.Timestamp(ev.date_fin)
        years = range(FIRST_YEAR, last_year + 1) if ev.recurrence == "annuelle" else [start.year]
        for y in years:
            shift = pd.DateOffset(years=y - start.year)
            rows.append({"nom": ev.nom, "debut": (start + shift).date(), "fin": (end + shift).date(),
                         "heure_debut": int(ev.heure_debut), "heure_fin": int(ev.heure_fin), "type": ev.type,
                         "rues": ev.rues, "a_verifier": ev.a_verifier == "oui"})
    return pd.DataFrame(rows).sort_values("debut").reset_index(drop=True)


def fetch_que_faire() -> pd.DataFrame:
    df = export_csv(PARIS, "que-faire-a-paris-", where="date_start is not null",
                    select="id,title,date_start,date_end,address_name,address_street,address_zipcode,"
                           "lat_lon,qfap_tags,price_type,access_type,updated_at")
    for col in ["date_start", "date_end", "updated_at"]:
        df[col] = pd.to_datetime(df[col], utc=True)
    latlon = df.pop("lat_lon").str.split(",", expand=True)
    df["lat"] = pd.to_numeric(latlon[0], errors="coerce")
    df["lon"] = pd.to_numeric(latlon[1], errors="coerce")
    return df.sort_values("date_start").reset_index(drop=True)


def update() -> None:
    EVENEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    manual = expand_manual(pd.read_csv(MANUAL_PATH, sep=";"), date.today().year + 1)
    manual.to_parquet(EVENEMENTS_DIR / "evenements.parquet", index=False)
    print(f"evenem.  evenements           {len(manual)} occurrences ({manual['nom'].nunique()} événements) -> evenements.parquet")
    qfap = fetch_que_faire()
    qfap.to_parquet(EVENEMENTS_DIR / "que_faire_a_paris.parquet", index=False)
    print(f"evenem.  que_faire_a_paris    {len(qfap)} lignes, {qfap['date_start'].min():%Y-%m-%d} -> "
          f"{qfap['date_start'].max():%Y-%m-%d} -> que_faire_a_paris.parquet")


def load(name: str = "evenements") -> pd.DataFrame:
    return pd.read_parquet(EVENEMENTS_DIR / f"{name}.parquet")


if __name__ == "__main__":
    update()
