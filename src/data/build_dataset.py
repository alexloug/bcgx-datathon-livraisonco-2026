"""Construit le dataset unifié d'entraînement et de prédiction -> data/processed/dataset.parquet.

Une ligne par arc du brief et par heure, de TRAIN_START à la fin de la fenêtre à prédire, avec les deux cibles
(nettoyées, Champs-Elysées reconstruit par proxy après l'arrêt du capteur) et toutes les features disponibles
au moment de prédire. Hypothèses, choix et ruptures : docs/2_donnees.md.

    from src.data.build_dataset import build_dataset
    df, meta = build_dataset()                                   # défauts de config.py
    df, meta = build_dataset(start="2022-01-01", end="2026-11-11 23:00")

Usage : python -m src.data.build_dataset [--start "2021-06-21 00:00"] [--end "2026-11-11 23:00"]
(après python -m src.data.update)
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta

import numpy as np
import pandas as pd

from src.config import (ARCS, CHAMPS_PROXIES, CHANTIERS_DIR, DATASET_PATH, EVENTS, MIN_LAG_HOURS, SUBMISSION_END, SUBMISSION_START,
                        TARGETS, TRAIN_START, TZ)
from src.data.sources import calendrier, chantiers, evenements, meteo, trafic
from src.features.build import calendar_features, lag_features, level_features, yearly_lag_features

STUCK_HOURS = 6          # séquence de valeurs identiques (> 0) à partir de laquelle le capteur est jugé figé
SPIKE_FACTOR = 2         # pic = valeur > SPIKE_FACTOR x quantile 99,9 % de l'arc
METEO_COLS = ["temperature_2m", "precipitation", "rain", "snowfall", "cloud_cover", "wind_speed_10m", "weather_code"]
POST_JO = pd.Timestamp("2024-09-09", tz=TZ)
VILLE_30 = pd.Timestamp("2021-08-30", tz=TZ)
CHANTIER_ARC_M = 30          # emprise sur la chaussée de l'arc lui-même
CHANTIER_PROCHE_M = 150      # carrefours et rues adjacentes
CHANTIER_NOUVEAU_JOURS = 7   # un chantier qui commence perturbe plus qu'un chantier installé
VACANCES_TYPES = {"Toussaint": 1, "Noël": 2, "Hiver": 3, "Printemps": 4, "Été": 5}
SCOLAIRE_MAX_JOURS = 42      # au-delà de 6 semaines d'une reprise ou des vacances, plus d'effet visible


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
    # Une heure n'est reconstruite que si les deux cibles le sont (le couple reste cohérent), et si l'occupation
    # reconstruite reste possible : ratio x forte occupation du proxy peut dépasser 100 % (6 heures depuis 2021).
    can = (fill & rebuilt["debit_horaire"].notna() & rebuilt["taux_occupation"].notna()
           & (rebuilt["taux_occupation"] <= 100))
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
    return pd.concat([out, school_calendar(day)], axis=1)


def school_calendar(day: pd.Series) -> pd.DataFrame:
    """Position dans le calendrier scolaire (zone C), en jours, plafonnée à `SCOLAIRE_MAX_JOURS` ; -1 = sans objet.

    `vacances` seul traite chaque période comme un bloc. Or le trafic bouge à l'intérieur : creux de la mi-août,
    retour dès la dernière semaine d'août, semaine de reprise, semaines qui précèdent les vacances
    (docs/3_modele.md §4). Les ponts de l'Ascension ne comptent pas comme une période de vacances ici.
    """
    vac = calendrier.load("vacances_scolaires")
    vac = vac[~vac["nom"].str.contains("Ascension|Début")].assign(
        debut=lambda v: pd.to_datetime(v["debut"]), fin=lambda v: pd.to_datetime(v["fin"])).sort_values("debut")
    types = vac["nom"].map(lambda n: next((c for k, c in VACANCES_TYPES.items() if k in n), 0)).to_numpy()
    debut, fin = vac["debut"].to_numpy(), vac["fin"].to_numpy()

    days = pd.DatetimeIndex(day.unique())
    i = np.searchsorted(debut, days.to_numpy(), side="right") - 1      # dernière période commencée
    known = (i >= 0) & (i + 1 < len(debut))
    i = np.clip(i, 0, len(debut) - 2)
    en_vac = known & (days.to_numpy() < fin[i])
    jours = lambda delta: np.minimum(delta / np.timedelta64(1, "D"), SCOLAIRE_MAX_JOURS)
    daily = pd.DataFrame({
        "vacances_type": np.where(en_vac, types[i], 0),
        "jours_avant_reprise": np.where(en_vac, jours(fin[i] - days.to_numpy()), -1),
        "jours_depuis_reprise": np.where(known & ~en_vac, jours(days.to_numpy() - fin[i]), -1),
        "jours_avant_vacances": np.where(known & ~en_vac, jours(debut[i + 1] - days.to_numpy()), -1),
    }, index=days)
    daily[~known] = np.nan
    return pd.DataFrame(daily.reindex(day).to_numpy(), columns=daily.columns, index=day.index)


def level_vacation_share(df: pd.DataFrame, window_days: int = 28) -> pd.Series:
    """Part d'heures de vacances dans la fenêtre du niveau `<cible>_niveau28j` (mêmes bornes, cf. level_features).

    Le niveau d'une semaine qui suit des vacances est tiré vers le bas : c'est le cas de la fenêtre du 07-11/11,
    dont le niveau couvre en partie la Toussaint. Cette part permet au modèle de corriger.
    """
    shifted = df["vacances"].astype(float).groupby(df["arc"]).shift(MIN_LAG_HOURS)
    return shifted.groupby(df["arc"]).transform(lambda s: s.rolling(24 * window_days, min_periods=24 * 7).mean())


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
    # Neige et gel jouent à la journée (chaussée glissante, report vers les transports) et le lendemain, alors que
    # la chute horaire est presque toujours nulle : agrégats journaliers de la prévision, jours de Paris.
    fc_day = fc.groupby(fc.index.tz_convert(TZ).normalize().tz_localize(None))
    daily = pd.DataFrame({"meteo_neige_jour": fc_day["snowfall"].sum(), "meteo_tmin_jour": fc_day["temperature_2m"].min()})
    daily["meteo_neige_veille"] = daily["meteo_neige_jour"].shift(1)
    day = datetimes.dt.tz_convert(TZ).dt.normalize().dt.tz_localize(None)
    out[daily.columns] = daily.reindex(day).to_numpy()
    return out


def event_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Événement manuel en cours sur la rue de l'arc, à l'heure près, séparé par type.

    `evenement_fermeture` : avenue fermée ou vidée (cérémonies, défilés, Champs piétons…) ;
    `evenement_perturbation` : trafic reporté ou chargé (grèves, salons) ; `evenement` : l'un ou l'autre.
    """
    ev = evenements.load("evenements")
    for col, default in evenements.DEFAULTS.items():   # parquet généré avant l'ajout des colonnes
        if col not in ev:
            ev[col] = default
    hour = df["datetime"].dt.tz_convert(TZ).dt.tz_localize(None).dt.floor("h")
    rue = df["arc"].map({arc: spec["rue"] for arc, spec in ARCS.items()})
    flags = {t: pd.Series(False, index=df.index) for t in evenements.TYPES}
    for r in ev.itertuples():
        start = pd.Timestamp(r.debut) + pd.Timedelta(hours=int(r.heure_debut))
        end = pd.Timestamp(r.fin) + pd.Timedelta(hours=int(r.heure_fin))
        flags[r.type] |= (hour >= start) & (hour <= end) & rue.apply(lambda x: x in r.rues)
    out = pd.DataFrame({f"evenement_{t}": f for t, f in flags.items()}, index=df.index)
    out.insert(0, "evenement", out.any(axis=1))
    return out


def distance_to_segment(lon, lat, segment) -> np.ndarray:
    """Distance en mètres de points (lon, lat) à un segment [(lon, lat), (lon, lat)] (approximation locale plane)."""
    (x1, y1), (x2, y2) = segment
    kx, ky = 111_320 * np.cos(np.radians(y1)), 110_540
    px, py = (np.asarray(lon) - x1) * kx, (np.asarray(lat) - y1) * ky
    vx, vy = (x2 - x1) * kx, (y2 - y1) * ky
    t = np.clip((px * vx + py * vy) / (vx * vx + vy * vy), 0, 1)
    return np.hypot(px - t * vx, py - t * vy)


def chantier_features(df: pd.DataFrame) -> pd.DataFrame:
    """Emprises de chantier sur la chaussée autour de l'arc, jour par jour (dates de Paris).

    Surfaces en log(1 + m²). `chantier_voirie_*` exclut les chantiers de bâtiment (bennes, échafaudages) pour ne
    garder que la voirie et les réseaux. NaN pendant `chantiers.coverage_gap()` : les emprises y sont incomplètes,
    et un « pas de chantier » serait faux ; XGBoost apprend alors sur 2019-2024.
    """
    em = chantiers.load_emprises().dropna(subset=["lat", "lon", "date_debut", "date_fin"])
    em = em[em["localisation_detail"].str.contains("CHAUSSEE", na=False)]
    debut = pd.to_datetime(em["date_debut"]).to_numpy()
    fin = pd.to_datetime(em["date_fin"]).to_numpy()
    surface = pd.to_numeric(em["surface"], errors="coerce").fillna(0).to_numpy()
    voirie = (em["chantier_categorie"] != "Tiers (travaux sur bâtiment)").to_numpy()
    day = df["datetime"].dt.tz_convert(TZ).dt.normalize().dt.tz_localize(None)

    cols = ["chantier_surface_arc", "chantier_surface_150m", "chantier_voirie_150m", "chantier_nouveau_7j"]
    out = pd.DataFrame(np.nan, index=df.index, columns=cols)
    for arc, spec in ARCS.items():
        rows = df["arc"] == arc
        days = pd.DatetimeIndex(day[rows].unique())
        dist = distance_to_segment(em["lon"], em["lat"], spec["segment"])
        near = dist <= CHANTIER_PROCHE_M
        d = days.to_numpy()[:, None]
        active = (d >= debut[near]) & (d <= fin[near])
        recent = active & (d - debut[near] < np.timedelta64(CHANTIER_NOUVEAU_JOURS, "D"))
        s = surface[near]
        daily = pd.DataFrame({
            "chantier_surface_arc": np.log1p((active * (s * (dist[near] <= CHANTIER_ARC_M))).sum(axis=1)),
            "chantier_surface_150m": np.log1p((active * s).sum(axis=1)),
            "chantier_voirie_150m": np.log1p((active * (s * voirie[near])).sum(axis=1)),
            "chantier_nouveau_7j": recent.any(axis=1).astype(float),
        }, index=days)
        out.loc[rows] = daily.reindex(day[rows]).to_numpy()
    gap_start, gap_end = chantiers.coverage_gap()
    out[((day >= gap_start) & (day <= gap_end)).to_numpy()] = np.nan
    return out


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
    feats = pd.concat([feats, event_flags(feats)], axis=1)
    feats["vacances_niveau28j"] = level_vacation_share(feats)
    if (CHANTIERS_DIR / "chantiers_a_paris.parquet").exists():   # optionnel : le modèle n'utilise pas ces colonnes
        feats = pd.concat([feats, chantier_features(feats)], axis=1)

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
