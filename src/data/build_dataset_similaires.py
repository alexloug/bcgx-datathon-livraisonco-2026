"""Dataset augmenté par les arcs similaires -> data/processed/dataset_similaires.parquet.

Part du dataset livré au modèle (dataset_modele.parquet, 3 arcs du brief) et ajoute :

1. **Les lignes des arcs similaires retenus** (data/arcs/arcs_similaires_retenus.csv, sans doublons ni arcs trop
   troués), avec les mêmes règles que les arcs du brief : nettoyage `clean_targets`, mêmes lags et niveau 28 j
   (src/features/build.py), même date de dernière observation. Les features qui ne dépendent que de l'heure
   (calendrier, vacances, météo, régimes) sont reprises du dataset ; les événements, propres aux 3 rues du brief,
   sont à False. `arc` = « sim_<arc_id> ».
2. **Des features de groupe sur les lignes des 3 arcs du brief** : moyenne, sur les arcs similaires de cet arc du
   brief, de leurs lags rapportés à leur niveau 28 j (`sim_<cible>_lag168_rel`, `_lag_mean_rel`, `_lag52s_rel`)
   et nombre d'arcs disponibles (`sim_n`). Toutes décalées d'au moins 144 h, comme les lags dont elles viennent.
   Sur les lignes des arcs similaires, ces colonnes sont calculées de même, sur leur propre groupe.

Colonnes ajoutées : `role` (`brief` / `similaire`), `groupe` (arc du brief de rattachement). Un arc similaire à deux
arcs du brief (ex. Bd Brune 725) n'apparaît qu'une fois, rattaché à son meilleur score.

Usage : python -m src.data.build_dataset_similaires   (après trafic_similaires et controle_similaires)
"""

import json
from datetime import timedelta

import numpy as np
import pandas as pd

from src.config import ARCS, DATASET_MODELE_PATH, PROCESSED_DIR, SUBMISSION_END, TARGETS, TRAIN_START, TZ
from src.data.build_dataset import arc_series
from src.data.controle_similaires import OUT_PATH as RETENUS_PATH
from src.data.sources import trafic_similaires
from src.features.build import lag_features, level_features, yearly_lag_features

OUT = PROCESSED_DIR / "dataset_similaires.parquet"
PAR_ARC = ["lag168", "lag336", "lag504", "lag_mean", "niveau28j", "lag52s"]
GROUPE_LAGS = ["lag168", "lag_mean", "lag52s"]
EVENEMENTS = ["evenement", "evenement_fermeture", "evenement_perturbation"]


def retenus() -> pd.DataFrame:
    r = pd.read_csv(RETENUS_PATH, sep=";", dtype={"arc_id": str})
    r = r[r["retenu"]].sort_values("score").drop_duplicates("arc_id")
    return r[["arc_id", "arc_brief", "libelle", "score"]]


def series_similaires(base: pd.DataFrame, arcs: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp(TRAIN_START, tz=TZ).tz_convert("UTC")
    end = pd.Timestamp(SUBMISSION_END, tz=TZ).tz_convert("UTC")
    hours = pd.date_range(start - timedelta(days=371), end, freq="h", tz="UTC")
    last_observed = base.loc[base["is_valid"], "datetime"].max()
    raw = trafic_similaires.load(arcs["arc_id"].tolist())
    frames = []
    for a in arcs.itertuples():
        s = arc_series(raw, a.arc_id, hours)
        s.loc[s.index > last_observed, "valide"] = False
        s.loc[~s["valide"], TARGETS] = np.nan
        s["source_cible"] = np.where(s["valide"], "mesure", np.where(s.index > last_observed, "futur", "absent"))
        frames.append(s.drop(columns="etat_arc").rename_axis("datetime").reset_index()
                      .assign(arc=f"sim_{a.arc_id}", groupe=a.arc_brief))
    t = pd.concat(frames, ignore_index=True).rename(columns={"valide": "is_valid"})
    t = t.sort_values(["arc", "datetime"]).reset_index(drop=True)
    t = pd.concat([t, lag_features(t), level_features(t), yearly_lag_features(t)], axis=1)
    t = t[t["datetime"] >= start]

    # Features ne dépendant que de l'heure : celles du dataset, prises sur un arc du brief.
    par_heure = [c for c in base.columns if c not in t.columns and c not in EVENEMENTS + ["partie"]]
    heure = base.drop_duplicates("datetime").set_index("datetime")[par_heure + ["partie"]]
    t = t.join(heure, on="datetime")
    t[EVENEMENTS] = False
    return t


def groupe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Moyenne, par (groupe, heure), des lags relatifs des arcs similaires du groupe."""
    sim = df[df["role"] == "similaire"]
    rel = pd.DataFrame({"datetime": sim["datetime"], "groupe": sim["groupe"]})
    for col in TARGETS:
        niveau = sim[f"{col}_niveau28j"].where(sim[f"{col}_niveau28j"] > 0)
        for lag in GROUPE_LAGS:
            rel[f"sim_{col}_{lag}_rel"] = sim[f"{col}_{lag}"] / niveau
    agg = rel.groupby(["groupe", "datetime"]).mean()
    agg["sim_n"] = rel.groupby(["groupe", "datetime"])[f"sim_debit_horaire_lag168_rel"].count()
    return agg


def build() -> tuple[pd.DataFrame, dict]:
    base = pd.read_parquet(DATASET_MODELE_PATH).assign(role="brief")
    base["groupe"] = base["arc"]
    arcs = retenus()
    sim = series_similaires(base, arcs).assign(role="similaire")
    df = pd.concat([base, sim[base.columns]], ignore_index=True)
    df = df.join(groupe_features(df), on=["groupe", "datetime"])
    bools = [c for c in df.columns if df[c].dtype == object and set(df[c].dropna().unique()) <= {True, False}]
    df[bools] = df[bools].astype(bool)
    df = df.sort_values(["arc", "datetime"]).reset_index(drop=True)
    meta = {
        "arcs_similaires": arcs.groupby("arc_brief")["arc_id"].apply(list).to_dict(),
        "lignes_valides": df[df["is_valid"]].groupby("role").size().to_dict(),
        "sim_n_moyen_brief": df[df["role"] == "brief"].groupby("arc")["sim_n"].mean().round(1).to_dict(),
    }
    return df, meta


if __name__ == "__main__":
    df, meta = build()
    df.to_parquet(OUT, index=False)
    OUT.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"{len(df)} lignes, {df.shape[1]} colonnes -> {OUT}")
