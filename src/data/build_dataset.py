"""Construit le dataset unifié d'entraînement et de prédiction -> data/processed/dataset.parquet.

Une ligne par arc du brief et par heure, de TRAIN_START à la fin de la fenêtre à prédire, avec les deux cibles
(nettoyées, Champs-Elysées reconstruit par proxy après l'arrêt du capteur) et toutes les features disponibles
au moment de prédire. Hypothèses, choix et ruptures : docs/dataset_entrainement.md.

    from src.data.build_dataset import build_dataset
    df, meta = build_dataset()                                   # défauts de config.py
    df, meta = build_dataset(start="2022-01-01", end="2026-11-11 23:00")

Usage : python -m src.data.build_dataset [--start "2021-06-21 00:00"] [--end "2026-11-11 23:00"]
(après python -m src.data.update)
"""

import argparse
import json
from datetime import timedelta

import numpy as np
import pandas as pd

from src.config import (ARCS, CHAMPS_PROXIES, DATASET_PATH, MIN_LAG_HOURS, SUBMISSION_END, SUBMISSION_START,
                        TARGETS, TRAIN_START, TZ)
from src.data.build_external import EVENTS
from src.data.sources import calendrier, evenements, meteo, trafic
from src.features.build import calendar_features, lag_features, level_features, yearly_lag_features

STUCK_HOURS = 6          # séquence de valeurs identiques (> 0) à partir de laquelle le capteur est jugé figé
SPIKE_FACTOR = 2         # pic = valeur > SPIKE_FACTOR x quantile 99,9 % de l'arc
METEO_COLS = ["temperature_2m", "precipitation", "rain", "snowfall", "cloud_cover", "wind_speed_10m", "weather_code"]
POST_JO = pd.Timestamp("2024-09-09", tz=TZ)
VILLE_30 = pd.Timestamp("2021-08-30", tz=TZ)


# --- Cibles -----------------------------------------------------------------------------------------------

def clean_targets(df: pd.DataFrame) -> pd.Series:
    """Masque de validité d'une série d'arc (triée, grille horaire continue). Règles D3 et D5 du plan."""
    ok = df[TARGETS].notna().all(axis=1) & ~df["etat_arc"].isin(["Invalide", "Barré"])
    for col in TARGETS:
        x = df[col]
        run = (x != x.shift()).cumsum()
        stuck = (x.groupby(run).transform("size") >= STUCK_HOURS) & (x > 0)
        spike = x > SPIKE_FACTOR * x[ok].quantile(0.999)
        ok &= ~stuck & ~spike
    return ok


def arc_series(raw: pd.DataFrame, arc_id: str, hours: pd.DatetimeIndex) -> pd.DataFrame:
    """Série d'un arc sur la grille `hours`, avec la colonne `valide` après nettoyage."""
    s = raw[raw["arc_id"] == arc_id].set_index("datetime")[[*TARGETS, "etat_arc"]]
    s = s[~s.index.duplicated(keep="last")].reindex(hours)
    s["valide"] = clean_targets(s)
    return s


def day_type(index: pd.DatetimeIndex) -> np.ndarray:
    dow = index.tz_convert(TZ).day_of_week
    return np.select([dow == 5, dow == 6], ["samedi", "dimanche"], "semaine")


def fit_ratios(target: pd.Series, proxy: pd.Series) -> pd.Series:
    """Ratio médian cible / proxy par (heure locale, type de jour), sur les heures où les deux sont valides."""
    both = target.notna() & (proxy > 0)
    idx = target.index[both]
    key = pd.MultiIndex.from_arrays([idx.tz_convert(TZ).hour, day_type(idx)], names=["heure", "type_jour"])
    return pd.Series((target[both] / proxy[both]).to_numpy(), index=key).groupby(level=[0, 1]).median()


def apply_ratios(proxy: pd.Series, ratios: pd.Series) -> pd.Series:
    key = pd.MultiIndex.from_arrays([proxy.index.tz_convert(TZ).hour, day_type(proxy.index)])
    return pd.Series(proxy.to_numpy() * ratios.reindex(key).to_numpy(), index=proxy.index)


def reconstruct_champs(target: pd.DataFrame, proxies: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Remplit les heures invalides de 4264 par ses proxys recalés (CHAMPS_PROXIES). Renvoie aussi les ratios."""
    out = target.copy()
    out["source_cible"] = np.where(out["valide"], "mesure", "absent")
    fill = ~out["valide"]
    rebuilt, meta = {}, {}
    for col, spec in CHAMPS_PROXIES.items():
        recs = []
        for pid in spec["arcs"]:
            p = proxies[pid][col].where(proxies[pid]["valide"])
            ratios = fit_ratios(target[col].where(target["valide"]), p)
            meta[f"{col}/{pid}"] = {f"{h:02d}h {t}": round(r, 4) for (h, t), r in ratios.items()}
            recs.append(apply_ratios(p, ratios))
        stacked = pd.concat(recs, axis=1)
        rebuilt[col] = stacked.mean(axis=1) if spec["methode"] == "moyenne" else stacked.bfill(axis=1).iloc[:, 0]
    # Une heure n'est reconstruite que si les deux cibles le sont (le couple reste cohérent).
    can = fill & rebuilt["debit_horaire"].notna() & rebuilt["taux_occupation"].notna()
    for col in TARGETS:
        out.loc[can, col] = rebuilt[col][can]
    out.loc[can, "valide"] = True
    out.loc[can, "source_cible"] = "proxy"
    return out, meta


def build_targets(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    proxy_ids = sorted({pid for spec in CHAMPS_PROXIES.values() for pid in spec["arcs"]})
    raw = trafic.load([spec["arc_id"] for spec in ARCS.values()] + proxy_ids)
    # On garde un an d'historique avant `start` pour que les lags et le niveau 52 semaines existent dès le début.
    hours = pd.date_range(start - timedelta(days=371), end, freq="h", tz="UTC")
    frames, meta = [], {}
    for arc, spec in ARCS.items():
        s = arc_series(raw, spec["arc_id"], hours)
        if arc == "Champs-Elysées":
            s, meta["ratios_proxy"] = reconstruct_champs(s, {pid: arc_series(raw, pid, hours) for pid in proxy_ids})
        else:
            s["source_cible"] = np.where(s["valide"], "mesure", "absent")
        s.loc[~s["valide"], TARGETS] = np.nan
        frames.append(s.drop(columns="etat_arc").rename_axis("datetime").reset_index().assign(arc=arc))
    out = pd.concat(frames, ignore_index=True).rename(columns={"valide": "is_valid"})
    return out.sort_values(["arc", "datetime"]).reset_index(drop=True), meta


# --- Features exogènes ------------------------------------------------------------------------------------

def calendar_extra(datetimes: pd.Series) -> pd.DataFrame:
    """Fériés, veilles/lendemains, ponts, vacances zone C, heure d'été."""
    local = datetimes.dt.tz_convert(TZ)
    day = local.dt.normalize().dt.tz_localize(None)
    feries = set(pd.to_datetime(calendrier.load("jours_feries")["date"]))
    is_ferie = lambda d: d.isin(feries)
    dow = local.dt.day_of_week
    out = pd.DataFrame({
        "ferie": is_ferie(day),
        "veille_ferie": is_ferie(day + pd.Timedelta(days=1)),
        "lendemain_ferie": is_ferie(day - pd.Timedelta(days=1)),
        # Pont : lundi avant un mardi férié, ou vendredi après un jeudi férié.
        "pont": ((dow == 0) & is_ferie(day + pd.Timedelta(days=1))) | ((dow == 4) & is_ferie(day - pd.Timedelta(days=1))),
        "heure_ete": local.map(lambda t: t.dst() != timedelta(0)),
    }, index=datetimes.index)
    vac = calendrier.load("vacances_scolaires")
    out["vacances"] = False
    for r in vac.itertuples():
        out["vacances"] |= (day >= pd.Timestamp(r.debut)) & (day < pd.Timestamp(r.fin))
    return out


def weather(datetimes: pd.Series) -> pd.DataFrame:
    """Météo **prévue** (prévisions archivées, puis prévision courante au-delà) et, pour analyse, l'observée.

    Le modèle doit utiliser `meteo_*` (disponible au moment de prédire), pas `meteo_obs_*`.
    """
    past = meteo.load("prevision_historique")
    current = meteo.load("prevision")
    fc = pd.concat([past, current[current["datetime"] > past["datetime"].max()]]).set_index("datetime")[METEO_COLS]
    obs = meteo.load("historique").set_index("datetime")[METEO_COLS]
    out = pd.concat([fc.add_prefix("meteo_").reindex(datetimes).reset_index(drop=True),
                     obs.add_prefix("meteo_obs_").reindex(datetimes).reset_index(drop=True)], axis=1)
    out["meteo_pluie"] = out["meteo_precipitation"] > 0.1
    out.index = datetimes.index
    return out


def event_flags(df: pd.DataFrame) -> pd.Series:
    """Événement manuel en cours sur la rue de l'arc (cérémonies, défilés…)."""
    ev = evenements.load("evenements")
    day = df["datetime"].dt.tz_convert(TZ).dt.normalize().dt.tz_localize(None)
    rue = df["arc"].map({arc: spec["rue"] for arc, spec in ARCS.items()})
    flag = pd.Series(False, index=df.index)
    for r in ev.itertuples():
        flag |= (day >= pd.Timestamp(r.debut)) & (day <= pd.Timestamp(r.fin)) & rue.apply(lambda x: x in r.rues)
    return flag


def regime_flags(df: pd.DataFrame) -> pd.DataFrame:
    t = df["datetime"].dt.tz_convert(TZ)
    jo = pd.Series(False, index=df.index)
    for a, b in EVENTS["jo_2024"]:
        jo |= (t >= pd.Timestamp(a, tz=TZ)) & (t < pd.Timestamp(b, tz=TZ) + pd.Timedelta(days=1))
    return pd.DataFrame({"jo_2024": jo, "post_jo_2024": t >= POST_JO, "post_30kmh": t >= VILLE_30}, index=df.index)


# --- Assemblage -------------------------------------------------------------------------------------------

def build_dataset(start: str = TRAIN_START, end: str = SUBMISSION_END,
                  last_observed: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    """Dataset unifié (cibles + features), heures de Paris pour `start`/`end`.

    `last_observed` (UTC) : dernière heure connue ; les cibles postérieures sont mises à NaN (backtest ou rendu).
    Par défaut, la dernière heure valide de Convention ou Saint-Pères.
    """
    start_utc = pd.Timestamp(start, tz=TZ).tz_convert("UTC")
    end_utc = pd.Timestamp(end, tz=TZ).tz_convert("UTC")
    targets, meta = build_targets(start_utc, end_utc)
    if last_observed is None:
        last_observed = targets.loc[targets["is_valid"] & (targets["source_cible"] == "mesure"), "datetime"].max()
    future = targets["datetime"] > last_observed
    targets.loc[future, TARGETS] = np.nan
    targets.loc[future, "is_valid"] = False
    targets.loc[future, "source_cible"] = "futur"

    feats = pd.concat([
        targets,
        calendar_features(targets["datetime"]),
        calendar_extra(targets["datetime"]),
        lag_features(targets),
        level_features(targets),
        yearly_lag_features(targets),
        weather(targets["datetime"]),
        regime_flags(targets),
    ], axis=1)
    feats["evenement"] = event_flags(feats)

    feats = feats[feats["datetime"] >= start_utc].reset_index(drop=True)
    window = (feats["datetime"] >= pd.Timestamp(SUBMISSION_START, tz=TZ)) & (feats["datetime"] <= pd.Timestamp(SUBMISSION_END, tz=TZ))
    feats["partie"] = np.select([feats["datetime"] <= last_observed, window], ["historique", "a_predire"], "non_predit")

    meta.update({
        "start": str(start_utc), "end": str(end_utc), "last_observed": str(last_observed),
        "min_lag_hours": MIN_LAG_HOURS, "lignes": len(feats),
        "lignes_valides_par_arc": feats[feats["is_valid"]].groupby("arc").size().to_dict(),
        "source_cible_par_arc": feats.groupby(["arc", "source_cible"]).size().unstack(fill_value=0).to_dict("index"),
        "generated_at": str(pd.Timestamp.now(tz="UTC").floor("s")),
    })
    return feats, meta


def save(df: pd.DataFrame, meta: dict, path=DATASET_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construit data/processed/dataset.parquet")
    parser.add_argument("--start", default=TRAIN_START, help="début, heure de Paris")
    parser.add_argument("--end", default=SUBMISSION_END, help="fin, heure de Paris")
    args = parser.parse_args()
    df, meta = build_dataset(args.start, args.end)
    save(df, meta)
    print(f"{len(df)} lignes, {df.shape[1]} colonnes, {df['datetime'].min()} -> {df['datetime'].max()}")
    print(f"dernière heure observée : {meta['last_observed']}")
    print(pd.crosstab([df["arc"]], [df["partie"], df["source_cible"]]).to_string())
    print(f"-> {DATASET_PATH}")
