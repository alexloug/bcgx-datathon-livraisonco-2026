"""Comptages routiers permanents de la Ville de Paris -> data/trafic/.

Deux sources, ramenées au même schéma (une ligne par arc et par heure, `datetime` en UTC) :
- API « comptages-routiers-permanents » : 13 mois glissants, publiés en J-1 -> recent.parquet
- Archives annuelles « comptages-routiers-permanents-historique » (zips de ~200 Mo, 2010-2025)
  -> historique/<année>.parquet. Chaque zip est lu en flux, filtré sur nos rues, puis supprimé.

On garde toutes les rues de config.STREETS (proxys et arcs voisins), pas seulement les 3 arcs du brief.

Usage :
    python -m src.data.sources.trafic                          # 13 mois glissants
    python -m src.data.sources.trafic --history 2019 2025      # archives 2019 à 2025 (manquantes seulement)
    python -m src.data.sources.trafic --amont Lecourbe-Convention --since 2026-10-01   # requête ponctuelle
"""

from __future__ import annotations

import argparse
import io
import subprocess
import zipfile
from contextlib import contextmanager

import pandas as pd
import requests

from src.config import STREETS, TRAFIC_DIR
from src.data.sources.opendatasoft import PARIS, distinct, export_csv, quote_list

DATASET = "comptages-routiers-permanents"
HISTORY_URL = f"{PARIS}/comptages-routiers-permanents-historique/attachments/opendata_txt_{{year}}_zip"
HISTORY_YEARS = range(2010, 2026)
RECENT_PATH = TRAFIC_DIR / "recent.parquet"
HISTORY_DIR = TRAFIC_DIR / "historique"

COLUMNS = ["arc_id", "libelle", "nd_amont", "nd_aval", "datetime", "debit_horaire", "taux_occupation",
           "etat_trafic", "etat_arc", "source"]

# Codes des archives, vérifiés contre l'API sur oct.-nov. 2025 (mêmes valeurs heure par heure).
ETAT_TRAFIC = {0: "Inconnu", 1: "Fluide", 2: "Pré-saturé", 3: "Saturé", 4: "Bloqué"}
# etat_barre : seuls 2 et 3 apparaissent sur la période de recouvrement. Les autres codes restent
# bruts (« code_1 »…) en attendant l'analyse de qualité.
ETAT_ARC = {2: "Barré", 3: "Ouvert"}


def street_labels() -> list[str]:
    """Libellés exacts de l'API qui contiennent une de nos rues (le `like` ODSQL ne gère pas les `_`)."""
    return sorted(l for l in distinct(PARIS, DATASET, "libelle") if any(s in l for s in STREETS))


def fetch_api(where: str) -> pd.DataFrame:
    df = export_csv(PARIS, DATASET, where=where, order_by="t_1h",
                    select="iu_ac,libelle,t_1h,q,k,etat_trafic,libelle_nd_amont,libelle_nd_aval,etat_barre",
                    iu_ac=str)
    out = pd.DataFrame({
        "arc_id": df["iu_ac"],
        "libelle": df["libelle"],
        "nd_amont": df["libelle_nd_amont"],
        "nd_aval": df["libelle_nd_aval"],
        "datetime": pd.to_datetime(df["t_1h"], utc=True),
        "debit_horaire": pd.to_numeric(df["q"], errors="coerce"),
        "taux_occupation": pd.to_numeric(df["k"], errors="coerce"),
        "etat_trafic": df["etat_trafic"],
        "etat_arc": df["etat_barre"],
        "source": "api",
    })
    return out[COLUMNS]


def update_recent() -> pd.DataFrame:
    labels = street_labels()
    df = fetch_api(f"libelle in {quote_list(labels)}")
    if df.empty:
        raise RuntimeError(f"Aucune ligne pour {labels}")
    TRAFIC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RECENT_PATH, index=False)
    print(f"trafic   recent      {df['arc_id'].nunique()} arcs, {len(df)} lignes, "
          f"{df['datetime'].min():%Y-%m-%d %H:%M} -> {df['datetime'].max():%Y-%m-%d %H:%M} UTC -> {RECENT_PATH.name}")
    return df


@contextmanager
def open_member(zf: zipfile.ZipFile, zip_path, member: str):
    """Certains zips (2021…) sont en Deflate64, que `zipfile` ne lit pas : on passe alors par `unzip -p`."""
    try:
        f = zf.open(member)
    except NotImplementedError:
        proc = subprocess.Popen(["unzip", "-p", str(zip_path), member], stdout=subprocess.PIPE)
        try:
            yield proc.stdout
        finally:
            proc.stdout.close()
            if proc.wait() != 0:
                raise RuntimeError(f"unzip a échoué sur {member}")
        return
    with f:
        yield f


def read_archive(zip_path, streets=STREETS) -> pd.DataFrame:
    """Lit les fichiers hebdomadaires d'un zip annuel et ne garde que les lignes de nos rues."""
    keys = [s.encode() for s in streets]
    header, rows = None, []
    with zipfile.ZipFile(zip_path) as zf:
        for member in sorted(zf.namelist()):
            with open_member(zf, zip_path, member) as f:
                header = f.readline()
                for line in f:
                    if any(k in line for k in keys) and any(k in line.split(b";", 2)[1] for k in keys):
                        rows.append(line)
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(io.BytesIO(header + b"".join(rows)), sep=";", dtype={"iu_ac": str}, usecols=range(11))
    # Les archives sont en heure de Paris (l'API en UTC). Au passage à l'heure d'hiver, une seule
    # ligne 02:00 existe pour deux heures réelles : on la retire, comme les heures inexistantes de mars.
    local = pd.to_datetime(df["t_1h"]).dt.tz_localize("Europe/Paris", ambiguous="NaT", nonexistent="NaT")
    out = pd.DataFrame({
        "arc_id": df["iu_ac"],
        "libelle": df["libelle"],
        "nd_amont": df["libelle_nd_amont"],
        "nd_aval": df["libelle_nd_aval"],
        "datetime": local.dt.tz_convert("UTC"),
        "debit_horaire": pd.to_numeric(df["q"], errors="coerce"),
        "taux_occupation": pd.to_numeric(df["k"], errors="coerce"),
        "etat_trafic": df["etat_trafic"].map(ETAT_TRAFIC).fillna("code_" + df["etat_trafic"].astype(str)),
        "etat_arc": df["etat_barre"].map(ETAT_ARC).fillna("code_" + df["etat_barre"].astype(str)),
        "source": "archive",
    })
    return out.dropna(subset=["datetime"]).drop_duplicates(["arc_id", "datetime"], keep="last")[COLUMNS]


def update_year(year: int) -> pd.DataFrame:
    tmp = HISTORY_DIR / f"_{year}.zip"
    try:
        with requests.get(HISTORY_URL.format(year=year), stream=True, timeout=600) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        df = read_archive(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    df.to_parquet(HISTORY_DIR / f"{year}.parquet", index=False)
    return df


def update_history(years=HISTORY_YEARS, force: bool = False) -> None:
    """Une année en échec n'empêche pas les suivantes ; l'erreur est levée à la fin."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for year in years:
        out = HISTORY_DIR / f"{year}.parquet"
        if out.exists() and not force:
            print(f"trafic   {year}        déjà présent -> {out.name}")
            continue
        try:
            df = update_year(year)
        except Exception as exc:
            print(f"trafic   {year}        ÉCHEC : {exc!r}")
            failed.append(year)
            continue
        print(f"trafic   {year}        {df['arc_id'].nunique()} arcs, {len(df)} lignes -> {out.name}")
    if failed:
        raise RuntimeError(f"Archives trafic en échec : {failed}")


def update(history: list[int] | None = None, force: bool = False) -> None:
    update_recent()
    if history:
        update_history(history, force)


def load(arc_ids: list[str] | None = None) -> pd.DataFrame:
    """Archives puis API ; sur les heures communes, l'API l'emporte. `arc_ids` filtre à la lecture."""
    filters = [("arc_id", "in", arc_ids)] if arc_ids else None
    parts = [pd.read_parquet(p, filters=filters) for p in sorted(HISTORY_DIR.glob("*.parquet"))]
    if RECENT_PATH.exists():
        parts.append(pd.read_parquet(RECENT_PATH, filters=filters))
    if not parts:
        raise FileNotFoundError(f"Rien dans {TRAFIC_DIR} : lancer python -m src.data.update")
    df = pd.concat(parts, ignore_index=True)
    return df.drop_duplicates(["arc_id", "datetime"], keep="last").sort_values(["arc_id", "datetime"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comptages routiers : API 13 mois et archives annuelles")
    parser.add_argument("--history", nargs=2, type=int, metavar=("DEBUT", "FIN"),
                        help=f"années d'archives à récupérer, bornes incluses ({HISTORY_YEARS[0]}-{HISTORY_YEARS[-1]})")
    parser.add_argument("--force", action="store_true", help="retélécharger les années déjà présentes")
    parser.add_argument("--amont", help="requête ponctuelle : libellé du nœud amont")
    parser.add_argument("--aval", help="requête ponctuelle : libellé du nœud aval")
    parser.add_argument("--iu-ac", nargs="+", help="requête ponctuelle : identifiants d'arc")
    parser.add_argument("--since", help="requête ponctuelle : date de début incluse (ISO, UTC)")
    parser.add_argument("--out", help="requête ponctuelle : fichier parquet de sortie")
    args = parser.parse_args()

    if args.amont or args.aval or args.iu_ac:
        clauses = [f'libelle_nd_amont="{args.amont}"' if args.amont else None,
                   f'libelle_nd_aval="{args.aval}"' if args.aval else None,
                   f"iu_ac in {quote_list(args.iu_ac)}" if args.iu_ac else None,
                   f't_1h >= "{args.since}"' if args.since else None]
        df = fetch_api(" and ".join(filter(None, clauses)))
        print(df.groupby(["arc_id", "nd_amont", "nd_aval"]).agg(
            debut=("datetime", "min"), fin=("datetime", "max"), heures=("datetime", "size"),
            debit_renseigne=("debit_horaire", lambda s: s.notna().mean())).to_string())
        if args.out:
            df.to_parquet(args.out, index=False)
    else:
        years = range(args.history[0], args.history[1] + 1) if args.history else None
        update(list(years) if years else None, args.force)
