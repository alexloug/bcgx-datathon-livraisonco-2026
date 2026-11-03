"""Météo horaire au centre de Paris via Open-Meteo (sans clé, usage non commercial) -> data/meteo/.

- historique.parquet : observé (réanalyse ERA5 / ERA5-Land), de 2010 à ~J-2.
- prevision.parquet  : modèles de prévision, 7 jours passés + 16 jours à venir. Indispensable pour la
  fenêtre à prédire : on ne connaîtra pas la météo observée du 07 au 11/11 au moment du rendu.
- prevision_historique.parquet : prévisions archivées (depuis 2021), pour backtester avec la même
  qualité d'information qu'au moment du rendu plutôt qu'avec l'observé.

Usage : python -m src.data.sources.meteo
"""

from datetime import date, timedelta

import pandas as pd
import requests

from src.config import METEO_DIR, PARIS_LAT, PARIS_LON

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

VARIABLES = ["temperature_2m", "apparent_temperature", "precipitation", "rain", "snowfall",
             "cloud_cover", "wind_speed_10m", "wind_gusts_10m", "relative_humidity_2m",
             "weather_code", "is_day"]
HISTORY_START = 2010
HISTORICAL_FORECAST_START = 2021


def fetch(url: str, **params) -> pd.DataFrame:
    params = {"latitude": PARIS_LAT, "longitude": PARIS_LON, "hourly": ",".join(VARIABLES),
              "timezone": "UTC", **params}
    resp = requests.get(url, params=params, timeout=300)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()["hourly"])
    df.insert(0, "datetime", pd.to_datetime(df.pop("time"), utc=True))
    return df.dropna(subset=VARIABLES, how="all")


def by_year(url: str, start_year: int, end: date) -> pd.DataFrame:
    """Une requête par an pour rester sous les limites de taille de réponse."""
    parts = [fetch(url, start_date=f"{y}-01-01", end_date=min(date(y, 12, 31), end).isoformat())
             for y in range(start_year, end.year + 1)]
    return pd.concat(parts, ignore_index=True)


def save(df: pd.DataFrame, name: str) -> None:
    path = METEO_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"meteo    {name:20s} {len(df)} heures, {df['datetime'].min():%Y-%m-%d %H:%M} -> "
          f"{df['datetime'].max():%Y-%m-%d %H:%M} UTC -> {path.name}")


def update() -> None:
    METEO_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today()
    save(by_year(ARCHIVE_URL, HISTORY_START, today - timedelta(days=1)), "historique")
    save(by_year(HISTORICAL_FORECAST_URL, HISTORICAL_FORECAST_START, today - timedelta(days=1)), "prevision_historique")
    forecast = fetch(FORECAST_URL, past_days=7, forecast_days=16)
    forecast["fetched_at"] = pd.Timestamp.now(tz="UTC").floor("h")
    save(forecast, "prevision")


def load(name: str = "historique") -> pd.DataFrame:
    return pd.read_parquet(METEO_DIR / f"{name}.parquet")


if __name__ == "__main__":
    update()
