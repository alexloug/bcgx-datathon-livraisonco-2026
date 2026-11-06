"""Arcs similaires aux 3 arcs du brief -> data/arcs/.

Justification chiffrée de chaque choix (filtres, score, exclusions) : exploration/docs/arcs_similaires.md.

But : trouver, parmi les ~3 000 capteurs de Paris, ceux qui se comportent comme Champs-Elysées, Convention et
Saint-Pères, pour élargir l'entraînement (plus de jours fériés, de 11 novembre…). On ne télécharge pas encore leurs
données horaires : on décrit chaque arc, puis on classe les plus proches.

1. Caractéristiques de chaque arc (`caracteristiques.parquet`) :
   - tracé (`referentiel-comptages-routiers`) : position, longueur, période d'existence ;
   - profil de trafic sur 12 mois, agrégé côté serveur (API) : débit et occupation moyens par heure de Paris
     × jour de la semaine. Le capteur 4264 (Champs-Elysées) étant muet depuis le 19/11/2025, son profil vient
     des 12 mois qui précèdent, dans les données locales (`trafic.load`).
2. Voirie OpenStreetMap (`osm.parquet`) : type de route, voies, vitesse, sens unique, couloir de bus. Chaque capteur
   est associé au segment OSM le plus proche, parallèle et dans son sens de circulation (`match_osm`).
3. Similarité (`arcs_similaires.csv`, 15 arcs par arc du brief), somme de quatre écarts, chacun divisé par sa
   moyenne pour qu'ils pèsent autant : forme du profil normalisé (24 h semaine + 24 h week-end), niveau de débit,
   niveau d'occupation (capacité de la voie à débit égal), nombre de voies dans le sens du capteur (OSM).
   Candidats : couverture ≥ 80 % sur 12 mois, débit moyen ≥ 100 véh/h, ni bretelle ni périphérique, pas de copie du
   débit de l'arc du brief (compteur partagé : Convention 5662/5664/5674, Saint-Pères 192), un seul arc par groupe de
   débits identiques, au plus 2 tronçons par rue.
4. Vérification (`jours_feries.parquet`, colonnes `ecart_feries` / `percentile_feries`) : les arcs retenus
   réagissent-ils aux jours fériés comme l'arc du brief ? Contrôle indépendant, pas utilisé dans le score.

Usage : python -m exploration.data.arcs_similaires            (télécharge, ~10 min pour le profil, ~1 min OSM)
        python -m exploration.data.arcs_similaires --local    (recalcule la similarité sans retélécharger)
"""

import argparse
import io
import json
import time
import unicodedata

import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import KDTree

from src.config import ARCS, CALENDRIER_DIR, CHAMPS_PROXIES, DATA_DIR
from src.data.sources import trafic
from src.data.sources.opendatasoft import PARIS, export_csv

ARCS_DIR = DATA_DIR / "arcs"
DAYS = 365
MIN_COVERAGE = 0.8   # part des heures mesurées sur 12 mois : 1 856 arcs -> 1 028 (à 0,9 : 805)
MIN_DEBIT = 100      # véh/h ; nos arcs font 400 à 750, en dessous ce sont des rues d'une autre nature
TOP = 15
MAX_PAR_RUE = 2
ECARTS = ["ecart_forme", "ecart_debit", "ecart_occupation", "ecart_voies"]
SAME_COUNTER = 0.9   # part d'heures au débit identique au-delà de laquelle deux arcs partagent le même compteur
DEAD_ARCS = {"4264": ("2024-11-19", "2025-11-18")}   # arc muet : profil sur les 12 mois avant l'arrêt
# Vérification sur les jours fériés : 4264 étant muet sur la période, on prend ses proxys (moyenne des deux).
FERIES_REF = {"4264": CHAMPS_PROXIES["debit_horaire"]["arcs"]}
FERIES_HOURS = (7, 21)     # heures de Paris comparées (journée), fin exclue
MIN_FERIES = 3             # jours fériés en commun minimum pour comparer deux arcs


def fetch_referentiel() -> pd.DataFrame:
    df = export_csv(PARIS, "referentiel-comptages-routiers", iu_ac=str,
                    select="iu_ac,libelle,libelle_nd_amont,libelle_nd_aval,date_debut,date_fin,geo_point_2d,geo_shape")
    lat, lon = zip(*df.pop("geo_point_2d").str.split(",").map(lambda p: (float(p[0]), float(p[1]))))
    df["lat"], df["lon"] = lat, lon
    shape = df.pop("geo_shape").map(lambda g: np.array(json.loads(g)["coordinates"], dtype=float))
    df["longueur_m"] = shape.map(line_length)
    # Le tracé va de l'amont vers l'aval (4264 finit là où commence 4274) : il donne le sens de circulation.
    df["lon_amont"], df["lat_amont"] = shape.map(lambda c: c[0][0]), shape.map(lambda c: c[0][1])
    df["lon_aval"], df["lat_aval"] = shape.map(lambda c: c[-1][0]), shape.map(lambda c: c[-1][1])
    for col in ["date_debut", "date_fin"]:
        df[col] = pd.to_datetime(df[col], utc=True)
    # Un arc peut avoir plusieurs versions (dates) : on garde la plus récente.
    return df.sort_values("date_debut").drop_duplicates("iu_ac", keep="last").rename(columns={"iu_ac": "arc_id"})


def line_length(coords: np.ndarray) -> float:
    if coords.ndim != 2 or len(coords) < 2:
        return np.nan
    return float(sum(haversine(*a[::-1], *b[::-1]) for a, b in zip(coords[:-1], coords[1:])))


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6_371_000 * 2 * np.arcsin(np.sqrt(a))


def fetch_profils(since: pd.Timestamp) -> pd.DataFrame:
    """Débit / occupation moyens par arc × heure de Paris × jour (1 = lundi), calculés par l'API."""
    params = {"select": "iu_ac, avg(q) as q, avg(k) as k, count(q) as n",
              "group_by": "iu_ac, hour(t_1h) as h, date_format(t_1h, 'e') as dow",
              "where": f"t_1h >= date'{since:%Y-%m-%d}' and q is not null and k is not null "
                       f"and etat_barre != 'Barré' and etat_barre != 'Invalide'",
              "timezone": "Europe/Paris", "delimiter": ";"}
    resp = requests.get(f"{PARIS}/{trafic.DATASET}/exports/csv", params=params, timeout=1800)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.content.decode("utf-8-sig")), sep=";", dtype={"iu_ac": str})
    return df.rename(columns={"iu_ac": "arc_id"})


def local_profil(arc_id: str, start: str, end: str) -> pd.DataFrame:
    df = trafic.load([arc_id])
    t = df["datetime"].dt.tz_convert("Europe/Paris")
    start, end = pd.Timestamp(start, tz="Europe/Paris"), pd.Timestamp(end, tz="Europe/Paris") + pd.Timedelta(days=1)
    ok = (t >= start) & (t < end) & ~df["etat_arc"].isin(["Barré", "Invalide"])
    df = df[ok & df["debit_horaire"].notna() & df["taux_occupation"].notna()].assign(
        h=t[ok].dt.hour, dow=t[ok].dt.day_of_week + 1)
    out = df.groupby(["h", "dow"]).agg(q=("debit_horaire", "mean"), k=("taux_occupation", "mean"),
                                       n=("debit_horaire", "size")).reset_index()
    return out.assign(arc_id=arc_id)


OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
PARIS_BBOX = (48.810, 2.220, 48.910, 2.475)   # sud, ouest, nord, est : Paris intra-muros + périphérique
OSM_QUERY = ('[out:json][timeout:240];way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|'
             'residential|living_street)(_link)?$"]({},{},{},{});out tags geom;').format(*PARIS_BBOX)
MAX_MATCH_M = 30     # distance maximale entre le milieu d'un arc et la route OSM retenue
BUS_KEYS = ("busway", "busway:right", "busway:left", "busway:both")


def fetch_osm() -> pd.DataFrame:
    """Routes de Paris depuis OpenStreetMap (Overpass), une ligne par way.

    Les serveurs publics Overpass sont souvent saturés (réponse « at capacity ») : on essaie plusieurs serveurs,
    3 fois chacun, en attendant de plus en plus longtemps. On interroge un rectangle (bbox) plutôt que le contour
    administratif de Paris, que certains miroirs ne connaissent pas (ils renvoient alors 0 route)."""
    for attempt in range(3):
        for url in OVERPASS:
            try:
                resp = requests.post(url, data={"data": OSM_QUERY}, timeout=300,
                                     headers={"User-Agent": "datathon-livraisonco/0.1"})
                elements = resp.json()["elements"] if resp.ok else []
            except (requests.RequestException, ValueError):
                elements = []
            if len(elements) > 10_000:
                return pd.DataFrame([osm_row(e) for e in elements])
        time.sleep(30 * 2 ** attempt)
    raise RuntimeError("OpenStreetMap injoignable (serveurs Overpass saturés) : relancer plus tard")


def osm_row(e: dict) -> dict:
    t = e["tags"]
    oneway = t.get("oneway", "yes" if t.get("junction") == "roundabout" or t["highway"] == "motorway" else "no")
    return {"osm_id": e["id"], "osm_highway": t["highway"], "osm_nom": t.get("name"),
            "osm_voies": pd.to_numeric(t.get("lanes"), errors="coerce"),
            "osm_voies_avant": pd.to_numeric(t.get("lanes:forward"), errors="coerce"),
            "osm_voies_arriere": pd.to_numeric(t.get("lanes:backward"), errors="coerce"),
            "osm_vitesse": pd.to_numeric(t.get("maxspeed"), errors="coerce"),
            "osm_sens": {"yes": 1, "1": 1, "true": 1, "-1": -1}.get(oneway, 0),
            "osm_bus": any(t.get(k) in ("lane", "opposite_lane") for k in BUS_KEYS)
                       or any(k in t for k in ("lanes:bus", "lanes:psv", "bus:lanes", "psv:lanes")),
            "lon": [p["lon"] for p in e["geometry"]], "lat": [p["lat"] for p in e["geometry"]]}


def to_xy(lon, lat):
    """Projection locale en mètres autour de Paris (suffisante à l'échelle de la ville)."""
    return np.c_[(np.asarray(lon) - 2.35) * 111_320 * np.cos(np.radians(48.86)), (np.asarray(lat) - 48.86) * 110_574]


def match_osm(ref: pd.DataFrame, osm: pd.DataFrame) -> pd.DataFrame:
    """Route OSM de chaque arc : le segment le plus proche du milieu de l'arc (≤ MAX_MATCH_M), parallèle à l'arc
    (angle < 60°) et, pour un sens unique, orienté dans le sens de circulation de l'arc. Le sens compte : les grands
    axes à chaussées séparées (Champs-Elysées…) ont une route OSM par sens."""
    seg_way, a, b = [], [], []
    for i, w in enumerate(osm.itertuples()):
        xy = to_xy(w.lon, w.lat)
        a.append(xy[:-1]), b.append(xy[1:]), seg_way.append(np.full(len(xy) - 1, i))
    a, b, seg_way = np.vstack(a), np.vstack(b), np.concatenate(seg_way)
    seg_dir = (b - a) / np.maximum(np.linalg.norm(b - a, axis=1, keepdims=True), 1e-9)
    sens = osm["osm_sens"].to_numpy()[seg_way]
    tree = KDTree((a + b) / 2)
    p0, p1 = to_xy(ref["lon_amont"], ref["lat_amont"]), to_xy(ref["lon_aval"], ref["lat_aval"])
    mid, u = (p0 + p1) / 2, (p1 - p0) / np.maximum(np.linalg.norm(p1 - p0, axis=1, keepdims=True), 1e-9)
    _, idx = tree.query(mid, k=60)
    rows = []
    for m, uu, cand in zip(mid, u, idx):
        ab = b[cand] - a[cand]
        t = np.clip(((m - a[cand]) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-9), 0, 1)
        dist = np.linalg.norm(a[cand] + t[:, None] * ab - m, axis=1)
        cos = seg_dir[cand] @ uu
        ok = np.where(sens[cand] == 0, np.abs(cos) > 0.5, cos * sens[cand] > 0.5) & (dist <= MAX_MATCH_M)
        if not ok.any():
            rows.append({"osm_id": np.nan, "osm_distance_m": np.nan, "osm_dans_le_sens": np.nan})
            continue
        j = cand[ok][np.argmin(dist[ok])]
        rows.append({"osm_row": seg_way[j], "osm_distance_m": dist[ok].min(),
                     "osm_dans_le_sens": bool(seg_dir[j] @ uu > 0)})
    m = pd.DataFrame(rows, index=ref.index)
    out = osm.drop(columns=["lon", "lat"]).iloc[m["osm_row"].dropna().astype(int)].set_index(m.index[m["osm_row"].notna()])
    out = out.join(m[["osm_distance_m", "osm_dans_le_sens"]], how="right")
    # Voies dans le sens du capteur : `lanes` pour un sens unique, `lanes:forward/backward` sinon, à défaut lanes / 2.
    avant = out["osm_voies_avant"].where(out["osm_dans_le_sens"] == True, out["osm_voies_arriere"])
    out["voies_sens"] = np.where(out["osm_sens"] != 0, out["osm_voies"], avant.fillna(out["osm_voies"] / 2))
    out["osm_nom_ok"] = [name_match(l, n) for l, n in zip(ref["libelle"], out["osm_nom"])]
    return out.drop(columns=["osm_voies_avant", "osm_voies_arriere"])


STOP = {"av", "ave", "avenue", "bd", "boulevard", "rue", "quai", "pt", "pont", "place", "pl", "des", "de", "du",
        "la", "le", "les", "rte", "route", "porte", "pte", "bretelle", "voie", "georges", "saint", "sts", "st"}


def name_match(libelle, osm_nom) -> bool | float:
    """Contrôle : un mot significatif du libellé de la Ville se retrouve dans le nom OSM."""
    if not isinstance(osm_nom, str):
        return np.nan
    norm = lambda x: unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode().lower()
    words = {w for w in norm(libelle).replace("-", "_").replace("'", "_").split("_") if len(w) >= 4 and w not in STOP}
    return any(w in norm(osm_nom) for w in words) if words else np.nan


def caracteristiques(ref: pd.DataFrame, prof: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par arc : tracé, niveaux, couverture et profil normalisé (colonnes sem_00…23, we_00…23)."""
    prof = prof.assign(we=prof["dow"] >= 6)
    tot = prof.groupby("arc_id").agg(n=("n", "sum"))
    tot["couverture"] = (tot["n"] / (DAYS * 24)).clip(upper=1)
    w = prof.assign(qn=prof["q"] * prof["n"], kn=prof["k"] * prof["n"]).groupby("arc_id")[["qn", "kn", "n"]].sum()
    tot["debit_moyen"] = w["qn"] / w["n"]
    tot["occupation_moyenne"] = w["kn"] / w["n"]
    shape = prof.groupby(["arc_id", "we", "h"])["q"].mean().unstack(["we", "h"])
    shape.columns = [f"{'we' if we else 'sem'}_{h:02d}" for we, h in shape.columns]
    shape = shape.div(tot["debit_moyen"], axis=0)
    tot["ratio_weekend"] = shape.filter(like="we_").mean(axis=1) / shape.filter(like="sem_").mean(axis=1)
    tot["pointe_matin"] = shape[["sem_08", "sem_09"]].mean(axis=1)
    tot["pointe_soir"] = shape[["sem_18", "sem_19"]].mean(axis=1)
    return ref.set_index("arc_id").join(tot, how="right").join(shape).reset_index()


def same_counter(arc_id: str, others: list[str]) -> list[str]:
    """Arcs de `others` dont le débit est identique à celui de `arc_id` sur au moins SAME_COUNTER des heures
    communes. Vérifié sur les données locales (trafic.load), qui contiennent toutes les rues du brief."""
    if not others:
        return []
    w = trafic.load([arc_id, *others]).pivot_table(index="datetime", columns="arc_id", values="debit_horaire")
    w = w[w.index >= w.index.max() - pd.Timedelta(days=DAYS)] if arc_id not in DEAD_ARCS else \
        w[(w.index >= pd.Timestamp(DEAD_ARCS[arc_id][0], tz="Europe/Paris"))
          & (w.index < pd.Timestamp(DEAD_ARCS[arc_id][1], tz="Europe/Paris"))]
    same = [o for o in others if o in w and (w[o] == w[arc_id])[w[[o, arc_id]].notna().all(axis=1)].mean() >= SAME_COUNTER]
    return same


def similaires(car: pd.DataFrame) -> pd.DataFrame:
    shape_cols = [c for c in car if c.startswith(("sem_", "we_"))]
    car = car.set_index("arc_id")
    brief = {spec["arc_id"]: name for name, spec in ARCS.items()}
    # Bretelles et périphérique : comportement de voie rapide, pas comparable à nos avenues.
    rapide = car["osm_highway"].fillna("").str.contains("motorway|_link") | car["libelle"].str.contains("bretelle", case=False)
    cand = car[(car["couverture"] >= MIN_COVERAGE) & (car["debit_moyen"] >= MIN_DEBIT) & ~rapide
               & car[shape_cols].notna().all(axis=1)].drop(list(brief), errors="ignore")
    # Voies OSM seulement si la route associée porte le bon nom (sinon on a pu prendre la rue d'à côté).
    voies = car["voies_sens"].where(car["osm_nom_ok"] != False)
    out = []
    for arc_id, name in brief.items():
        ref = car.loc[arc_id]
        ref_shape = ref[shape_cols].astype(float)
        d = pd.DataFrame(index=cand.index)
        d["ecart_forme"] = np.sqrt(((cand[shape_cols].astype(float) - ref_shape) ** 2).mean(axis=1))
        d["ecart_debit"] = np.abs(np.log(cand["debit_moyen"] / float(ref["debit_moyen"])))
        d["ecart_occupation"] = np.abs(np.log(cand["occupation_moyenne"] / float(ref["occupation_moyenne"])))
        # Voies inconnues (OSM vide, ou pour l'arc du brief : Convention) -> écart moyen, c.-à-d. neutre.
        d["ecart_voies"] = np.abs(np.log(voies[cand.index] / voies[arc_id]))
        d["ecart_voies"] = d["ecart_voies"].fillna(d["ecart_voies"].mean()).fillna(0)
        # Chaque écart divisé par sa moyenne : aucun ne domine à cause de son unité (la médiane ne marche pas pour
        # les voies, souvent égales donc d'écart médian nul).
        d["score"] = sum(d[c] / (d[c].mean() or 1) for c in ECARTS)
        d["distance_km"] = haversine(ref["lat"], ref["lon"], cand["lat"], cand["lon"]) / 1000
        d["meme_rue"] = cand["libelle"].str.contains(ARCS[name]["rue"], regex=False)
        # Copies du débit de l'arc du brief (même compteur de débit partagé par des tronçons qui se suivent) :
        # elles n'apportent aucun exemple de plus, on les écarte.
        d = d.drop(same_counter(arc_id, list(d.index[d["meme_rue"]])))
        # Même phénomène entre deux candidats (débits moyens identiques) : on n'en garde qu'un.
        d = d.sort_values("score")
        d = d[~cand.loc[d.index, "debit_moyen"].round(3).duplicated()]
        # Au plus MAX_PAR_RUE tronçons d'une même rue : des tronçons qui se suivent apportent peu d'information neuve.
        d = d[d.groupby(cand.loc[d.index, "libelle"].str.lower()).cumcount() < MAX_PAR_RUE]
        top = d.head(TOP).join(
            cand[["libelle", "libelle_nd_amont", "libelle_nd_aval", "debit_moyen", "occupation_moyenne",
                  "ratio_weekend", "couverture", "osm_nom", "osm_highway", "voies_sens", "osm_vitesse", "osm_bus",
                  "date_debut"]])
        out.append(top.reset_index().assign(arc_brief=name, rang=range(1, len(top) + 1)))
    cols = ["arc_brief", "rang", "arc_id", "libelle", "libelle_nd_amont", "libelle_nd_aval", "score",
            *ECARTS, "debit_moyen", "occupation_moyenne", "ratio_weekend", "couverture", "osm_nom", "osm_highway",
            "voies_sens", "osm_vitesse", "osm_bus", "distance_km", "meme_rue", "date_debut"]
    out = pd.concat(out)[cols]
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(3)
    return out


def jours_feries() -> set[pd.Timestamp]:
    return set(pd.to_datetime(pd.read_parquet(CALENDRIER_DIR / "jours_feries.parquet")["date"]))


def fetch_feries(since: pd.Timestamp) -> pd.DataFrame:
    """Débit moyen 7h-21h par arc et par jour, pour chaque férié de semaine depuis `since` et les mêmes jours de
    semaine 1 et 2 semaines avant / après (jours de référence). Agrégé par l'API : quelques secondes."""
    jf = jours_feries()
    today = pd.Timestamp.now().normalize()
    feries = sorted(d for d in jf if since.tz_localize(None) <= d < today - pd.Timedelta(weeks=2) and d.day_of_week < 5)
    days = sorted({d + pd.Timedelta(weeks=w) for d in feries for w in (-2, -1, 0, 1, 2)})
    where = " or ".join(f"(t_1h >= date'{d:%Y-%m-%d}' and t_1h < date'{d + pd.Timedelta(days=1):%Y-%m-%d}')"
                        for d in days)
    params = {"select": "iu_ac, avg(q) as q, count(q) as n",
              "group_by": "iu_ac, date_format(t_1h, 'yyyy-MM-dd') as jour",
              "where": f"({where}) and hour(t_1h) >= {FERIES_HOURS[0]} and hour(t_1h) < {FERIES_HOURS[1]} "
                       f"and q is not null and etat_barre != 'Barré' and etat_barre != 'Invalide'",
              "timezone": "Europe/Paris", "delimiter": ";"}
    resp = requests.get(f"{PARIS}/{trafic.DATASET}/exports/csv", params=params, timeout=1800)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.content.decode("utf-8-sig")), sep=";", dtype={"iu_ac": str})
    return df.rename(columns={"iu_ac": "arc_id"}).assign(jour=lambda x: pd.to_datetime(x["jour"]))


def effet_feries(jours: pd.DataFrame) -> pd.DataFrame:
    """Effet férié = débit du férié / médiane des jours de référence (jours x arcs). Un jour ne compte que s'il a au
    moins 12 des 14 heures, et au moins 2 jours de référence (eux-mêmes non fériés)."""
    jf = jours_feries()
    feries = sorted(d for d in jours["jour"].unique() if d in jf)
    full = FERIES_HOURS[1] - FERIES_HOURS[0]
    q = jours[jours["n"] >= full - 2].pivot_table(index="jour", columns="arc_id", values="q")
    out = {}
    for d in feries:
        refs = [r for r in (d + pd.Timedelta(weeks=w) for w in (-2, -1, 1, 2)) if r in q.index and r not in jf]
        if d in q.index and refs:
            out[d] = q.loc[d] / q.loc[refs].median().where(q.loc[refs].notna().sum() >= 2)
    return pd.DataFrame(out).T


def verif_feries(sim: pd.DataFrame, car: pd.DataFrame, effet: pd.DataFrame) -> pd.DataFrame:
    """Ajoute à chaque arc similaire :
    - `ecart_feries` : écart moyen entre son effet férié et celui de l'arc du brief (jours en commun) ;
    - `percentile_feries` : part des arcs comparables (mêmes filtres de couverture et de débit) qui font mieux.
      Au hasard on attendrait 50 % ; plus c'est bas, plus l'arc réagit aux fériés comme le nôtre."""
    elig = car.index[(car["couverture"] >= MIN_COVERAGE) & (car["debit_moyen"] >= MIN_DEBIT)]
    parts = []
    for name, g in sim.groupby("arc_brief", sort=False):
        b = ARCS[name]["arc_id"]
        ref_ids = [a for a in FERIES_REF.get(b, [b]) if a in effet]
        ref = effet[ref_ids].mean(axis=1)
        common = effet.notna() & ref.notna().to_numpy()[:, None]
        e = effet.sub(ref, axis=0).abs().where(common).mean()
        e = e[(common.sum() >= MIN_FERIES) & e.index.isin(elig)].drop([b, *ref_ids], errors="ignore")
        g = g.assign(ecart_feries=g["arc_id"].map(e))
        parts.append(g.assign(percentile_feries=g["ecart_feries"].map(lambda v: 100 * (e < v).mean() if pd.notna(v) else np.nan)))
        print(f"feries   {name:15s} {int(ref.notna().sum())} fériés, similaires : écart {g['ecart_feries'].mean():.3f} "
              f"contre {e.median():.3f} pour l'ensemble des arcs, percentile moyen "
              f"{parts[-1]['percentile_feries'].mean():.0f} % (hasard : 50 %)")
    return pd.concat(parts)


def main(local: bool = False) -> None:
    ARCS_DIR.mkdir(parents=True, exist_ok=True)
    if local:
        ref = pd.read_parquet(ARCS_DIR / "referentiel.parquet")
        prof = pd.read_parquet(ARCS_DIR / "profils.parquet")
        osm = pd.read_parquet(ARCS_DIR / "osm.parquet")
        jours = pd.read_parquet(ARCS_DIR / "jours_feries.parquet")
    else:
        since = pd.Timestamp.now(tz="Europe/Paris").normalize() - pd.Timedelta(days=DAYS)
        ref = fetch_referentiel()
        ref.to_parquet(ARCS_DIR / "referentiel.parquet", index=False)
        prof = fetch_profils(since)
        prof.to_parquet(ARCS_DIR / "profils.parquet", index=False)
        osm = fetch_osm()
        osm.to_parquet(ARCS_DIR / "osm.parquet", index=False)
        jours = fetch_feries(since)
        jours.to_parquet(ARCS_DIR / "jours_feries.parquet", index=False)
    ref = ref[ref["arc_id"].isin(prof["arc_id"])].reset_index(drop=True)
    ref = pd.concat([ref, match_osm(ref, osm)], axis=1)
    for arc_id, (start, end) in DEAD_ARCS.items():
        prof = pd.concat([prof[prof["arc_id"] != arc_id], local_profil(arc_id, start, end)], ignore_index=True)
    car = caracteristiques(ref, prof)
    car.to_parquet(ARCS_DIR / "caracteristiques.parquet", index=False)
    sim = verif_feries(similaires(car), car.set_index("arc_id"), effet_feries(jours))
    sim[["ecart_feries", "percentile_feries"]] = sim[["ecart_feries", "percentile_feries"]].round(3)
    sim.to_csv(ARCS_DIR / "arcs_similaires.csv", sep=";", index=False)
    print(f"arcs     {len(car)} arcs décrits, {len(sim)} similaires -> {ARCS_DIR.relative_to(DATA_DIR.parent)}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="réutilise les .parquet déjà téléchargés")
    main(parser.parse_args().local)
