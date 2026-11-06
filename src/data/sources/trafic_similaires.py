"""Historique des arcs similaires (data/arcs/arcs_similaires.csv) -> data/trafic/similaires/.

Même schéma que src/data/sources/trafic.py, filtré par identifiant d'arc et non par rue :
- archives annuelles -> similaires/<année>.parquet
- API 13 mois glissants -> similaires/recent.parquet

On exclut 4281, déjà téléchargé (rue des Champs-Elysées) et utilisé pour reconstruire la cible des Champs.

Usage :
    python -m src.data.sources.trafic_similaires                     # API + archives 2021-2025 manquantes
    python -m src.data.sources.trafic_similaires --history 2021 2025 --force
"""

import argparse
import time

import pandas as pd

from src.config import DATA_DIR, TRAFIC_DIR
from src.data.sources.opendatasoft import quote_list
from src.data.sources.trafic import fetch_api, update_year

ARCS_PATH = DATA_DIR / "arcs" / "arcs_similaires.csv"
OUT_DIR = TRAFIC_DIR / "similaires"
EXCLUS = {"4281"}   # proxy de la cible des Champs-Elysées


def arc_ids() -> list[str]:
    arcs = pd.read_csv(ARCS_PATH, sep=";", dtype={"arc_id": str})
    return sorted(set(arcs["arc_id"]) - EXCLUS, key=int)


def update_recent(ids: list[str]) -> pd.DataFrame:
    df = fetch_api(f"iu_ac in {quote_list(ids)}")
    df.to_parquet(OUT_DIR / "recent.parquet", index=False)
    print(f"similaires recent  {df['arc_id'].nunique()} arcs, {len(df)} lignes, "
          f"{df['datetime'].min():%Y-%m-%d} -> {df['datetime'].max():%Y-%m-%d %H:%M} UTC")
    return df


def update_history(ids: list[str], years, force: bool = False) -> None:
    failed = []
    for year in years:
        out = OUT_DIR / f"{year}.parquet"
        if out.exists() and not force:
            print(f"similaires {year}    déjà présent")
            continue
        for essai in range(1, 4):   # les archives (~240 Mo) coupent parfois en cours de route
            try:
                df = update_year(year, OUT_DIR, ids)
                break
            except Exception as exc:
                print(f"similaires {year}    essai {essai} en échec : {exc!r}", flush=True)
                time.sleep(10 * essai)
        else:
            failed.append(year)
            continue
        print(f"similaires {year}    {df['arc_id'].nunique()} arcs, {len(df)} lignes", flush=True)
    if failed:
        raise RuntimeError(f"Archives en échec : {failed}")


def load(ids: list[str] | None = None) -> pd.DataFrame:
    """Archives puis API ; sur les heures communes, l'API l'emporte (comme trafic.load)."""
    filters = [("arc_id", "in", ids)] if ids else None
    parts = [pd.read_parquet(p, filters=filters) for p in sorted(OUT_DIR.glob("2*.parquet"))]
    parts.append(pd.read_parquet(OUT_DIR / "recent.parquet", filters=filters))
    df = pd.concat(parts, ignore_index=True)
    return df.drop_duplicates(["arc_id", "datetime"], keep="last").sort_values(["arc_id", "datetime"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Historique trafic des arcs similaires")
    parser.add_argument("--history", nargs=2, type=int, default=[2021, 2025], metavar=("DEBUT", "FIN"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-recent", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids = arc_ids()
    print(f"similaires {len(ids)} arcs : {', '.join(ids)}")
    if not args.no_recent:
        update_recent(ids)
    update_history(ids, range(args.history[0], args.history[1] + 1), args.force)
