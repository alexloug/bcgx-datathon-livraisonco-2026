"""Stream A : construit data/processed/external.parquet (une ligne par heure, heure de Paris).

Colonnes déjà produites : datetime (UTC), ferie, confinement, jo_2024.
TODO stream A : météo (historique + prévision du 07 au 11/11), vacances scolaires zone C,
chantiers, autres événements.

Usage : python -m src.data.build_external
"""

from datetime import date, timedelta

import pandas as pd

from src.config import EXTERNAL_PATH, TZ

# Périodes à marquer (dates incluses). À compléter et vérifier par le stream A.
EVENTS = {
    "confinement": [
        ("2020-03-17", "2020-05-10"),
        ("2020-10-30", "2020-12-14"),
        ("2021-04-03", "2021-05-02"),
    ],
    "jo_2024": [
        ("2024-07-18", "2024-08-11"),   # JO, périmètres de sécurité compris
        ("2024-08-28", "2024-09-08"),   # Jeux paralympiques
    ],
}


def easter(year: int) -> date:
    """Dimanche de Pâques (algorithme de Meeus/Jones/Butcher)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def french_holidays(year: int) -> list[date]:
    e = easter(year)
    fixed = [(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)]
    return [date(year, m, d) for m, d in fixed] + [e + timedelta(days=n) for n in (1, 39, 50)]


def build(start: str = "2010-01-01", end: str = "2026-12-31") -> pd.DataFrame:
    local = pd.date_range(start, f"{end} 23:00", freq="h", tz=TZ)
    days = local.tz_localize(None).normalize()
    holidays = {d for y in range(local[0].year, local[-1].year + 1) for d in french_holidays(y)}
    out = pd.DataFrame({"datetime": local.tz_convert("UTC")})
    out["ferie"] = pd.Series(days.date).isin(holidays).to_numpy()
    for name, periods in EVENTS.items():
        flag = pd.Series(False, index=out.index)
        for lo, hi in periods:
            flag |= (days >= pd.Timestamp(lo)) & (days <= pd.Timestamp(hi))
        out[name] = flag.to_numpy()
    return out


if __name__ == "__main__":
    external = build()
    EXTERNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    external.to_parquet(EXTERNAL_PATH, index=False)
    print(external.drop(columns="datetime").mean().round(4).to_string())
    print(f"-> {EXTERNAL_PATH}")
