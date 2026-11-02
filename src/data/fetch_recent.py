"""Stream A : télécharge les comptages routiers (13 mois glissants, publiés en J-1) depuis l'API open data Paris.

Utilise l'endpoint d'export (pas de limite de lignes, contrairement à /records limité à 10 000)
et filtre côté serveur. Sans argument, récupère les 3 arcs du brief dans data/recent/,
lus en priorité par build_traffic.

Usage :
    python -m src.data.fetch_recent                                   # les 3 arcs du brief
    python -m src.data.fetch_recent --amont Av_Champs_Elysees-Washington
    python -m src.data.fetch_recent --amont Lecourbe-Convention --aval Convention-Blomet --since 2026-10-01
    python -m src.data.fetch_recent --iu-ac 4401 4402 --out data/recent/champs_proxies.csv
"""

import argparse
import io
from pathlib import Path

import pandas as pd
import requests

from src.config import ARCS, RECENT_DIR

EXPORT_URL = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/comptages-routiers-permanents/exports/csv"
FIELDS = "iu_ac,libelle,t_1h,q,k,etat_trafic,libelle_nd_amont,libelle_nd_aval,etat_barre"


def build_where(nd_amont=None, nd_aval=None, iu_ac=None, since=None, until=None) -> str:
    clauses = []
    if nd_amont:
        clauses.append(f'libelle_nd_amont="{nd_amont}"')
    if nd_aval:
        clauses.append(f'libelle_nd_aval="{nd_aval}"')
    if iu_ac:
        clauses.append("iu_ac in (" + ", ".join(f'"{i}"' for i in iu_ac) + ")")
    if since:
        clauses.append(f't_1h >= "{since}"')
    if until:
        clauses.append(f't_1h <= "{until}"')
    if not clauses:
        raise ValueError("Préciser au moins un filtre (nœud amont/aval ou iu_ac) : le jeu complet fait ~30 M de lignes")
    return " and ".join(clauses)


def fetch_traffic(nd_amont: str | None = None, nd_aval: str | None = None, iu_ac: list[str] | None = None,
                  since: str | None = None, until: str | None = None, timeout: int = 300) -> pd.DataFrame:
    """Renvoie les comptages horaires correspondant aux filtres (dates ISO, interprétées en UTC)."""
    params = {
        "where": build_where(nd_amont, nd_aval, iu_ac, since, until),
        "select": FIELDS,
        "order_by": "t_1h",
        "timezone": "UTC",
        "delimiter": ";",
    }
    resp = requests.get(EXPORT_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.content.decode("utf-8-sig")), sep=";", dtype={"iu_ac": str})


def summarize(name: str, df: pd.DataFrame) -> str:
    """Une ligne par arc : période couverte, part d'heures renseignées et dernière valeur, débit et occupation séparément."""
    if df.empty:
        return f"{name:15s} aucune ligne"
    lines = []
    for iu_ac, g in df.groupby("iu_ac"):
        stats = []
        for col, label in [("q", "débit"), ("k", "occup.")]:
            ok = g[col].notna()
            last = g.loc[ok, "t_1h"].max()[:13] if ok.any() else "jamais"
            stats.append(f"{label} {ok.mean():4.0%} dernier {last}")
        lines.append(f"{name:15s} arc {iu_ac:5s} {g['libelle_nd_amont'].iat[0]} -> {g['libelle_nd_aval'].iat[0]}  "
                     f"{len(g)} h ({g['t_1h'].min()[:10]} -> {g['t_1h'].max()[:10]})  " + " | ".join(stats))
    return "\n".join(lines)


def fetch_brief_arcs() -> None:
    """Historique complet des 3 arcs : ces fichiers alimentent build_traffic, on ne les tronque jamais."""
    RECENT_DIR.mkdir(parents=True, exist_ok=True)
    for arc, spec in ARCS.items():
        df = fetch_traffic(spec["nd_amont"], spec["nd_aval"])
        if df.empty:
            raise RuntimeError(f"Aucune ligne renvoyée pour {arc} : vérifier les nœuds dans config.ARCS")
        df.to_csv(RECENT_DIR / spec["raw_file"], sep=";", index=False)
        print(summarize(arc, df))
    print(f"-> {RECENT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Télécharge des comptages routiers depuis l'API open data Paris")
    parser.add_argument("--amont", help="libellé du nœud amont, ex. Av_Champs_Elysees-Washington")
    parser.add_argument("--aval", help="libellé du nœud aval, ex. Av_Champs_Elysees-Berri")
    parser.add_argument("--iu-ac", nargs="+", help="identifiants d'arc, ex. 4401 4402")
    parser.add_argument("--since", help="date de début incluse (ISO), ex. 2026-10-01")
    parser.add_argument("--until", help="date de fin incluse (ISO)")
    parser.add_argument("--out", type=Path, help="fichier CSV de sortie (défaut : data/recent/<filtre>.csv)")
    args = parser.parse_args()

    if not (args.amont or args.aval or args.iu_ac):
        if args.since or args.until or args.out:
            parser.error("--since, --until et --out demandent un filtre (--amont, --aval ou --iu-ac)")
        fetch_brief_arcs()
    else:
        df = fetch_traffic(args.amont, args.aval, args.iu_ac, args.since, args.until)
        if df.empty:
            raise SystemExit("Aucune ligne renvoyée : vérifier l'orthographe exacte des nœuds ou des identifiants")
        name = "_".join(filter(None, [args.amont, args.aval, *(args.iu_ac or [])]))
        out = args.out or RECENT_DIR / f"{name}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, sep=";", index=False)
        print(summarize("requête", df))
        print(f"-> {out}")
