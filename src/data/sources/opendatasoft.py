"""Client minimal pour les portails Opendatasoft (opendata.paris.fr, data.education.gouv.fr).

L'endpoint /exports n'a pas de limite de lignes, contrairement à /records (10 000 max).
"""

from __future__ import annotations

import io

import pandas as pd
import requests

PARIS = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets"
EDUCATION = "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets"


def export_csv(portal: str, dataset: str, where: str | None = None, select: str | None = None,
               order_by: str | None = None, timeout: int = 600, **dtype) -> pd.DataFrame:
    params = {"delimiter": ";", "timezone": "UTC"}
    for key, value in [("where", where), ("select", select), ("order_by", order_by)]:
        if value:
            params[key] = value
    resp = requests.get(f"{portal}/{dataset}/exports/csv", params=params, timeout=timeout)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.content.decode("utf-8-sig")), sep=";", dtype=dtype or None, low_memory=False)


def distinct(portal: str, dataset: str, field: str, timeout: int = 120) -> list[str]:
    params = {"select": field, "group_by": field, "limit": -1}
    resp = requests.get(f"{portal}/{dataset}/records", params=params, timeout=timeout)
    resp.raise_for_status()
    return [r[field] for r in resp.json()["results"] if r[field] is not None]


def quote_list(values) -> str:
    return "(" + ", ".join(f'"{v}"' for v in values) + ")"
