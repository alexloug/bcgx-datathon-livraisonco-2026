"""Exporte le dataset livré à l'équipe modèle -> data/processed/dataset_modele.parquet (+ .json).

Même lignes que dataset.parquet, mais seulement les colonnes utiles au modèle : clés, cibles, colonnes de suivi et
exactement les features que reçoit le prétraitement (`feature_columns` : sans chantiers ni météo observée).
Contrôles bloquants avant écriture. Doc : docs/livraison_dataset.md.

Les 4 colonnes chantiers (`chantier_*`, `chantier_features` dans build_dataset.py) sont toujours générées dans
dataset.parquet. `--avec-chantiers` les ajoute au fichier livré, pour analyse : le prétraitement du modèle les ignore
tant que `chantier_` reste dans `EXCLUDED_PREFIXES` (src/models/preprocessing.py).

Usage : python -m src.data.export_dataset                    (après python -m src.data.build_dataset)
        python -m src.data.export_dataset --avec-chantiers
"""

import argparse
import json

import pandas as pd

from src.config import ARCS, DATASET_MODELE_PATH, DATASET_PATH, MIN_LAG_HOURS, TARGETS
from src.models.preprocessing import EXCLUDED_PREFIXES, feature_columns

VERSION = "v2"
KEYS = ["arc", "datetime"]
SUIVI = ["is_valid", "source_cible", "partie"]
WINDOW_ROWS = len(ARCS) * 120   # 3 arcs x 5 jours x 24 h, comme le CSV de rendu

# Famille de chaque feature, par préfixe (dans cet ordre) : sert à la doc et au .json.
FAMILLES = [
    ("historique", ("debit_horaire_", "taux_occupation_")),
    ("calendrier scolaire", ("vacances", "jours_")),
    ("météo prévue", ("meteo_",)),
    ("événements", ("evenement",)),
    ("régimes", ("jo_2024", "post_")),
    ("calendrier", ("hour", "dow", "month", "is_weekend", "ferie", "veille_ferie", "lendemain_ferie", "pont",
                    "heure_ete")),
]


def famille(col: str) -> str:
    return next(nom for nom, prefixes in FAMILLES if col.startswith(prefixes))


def check(df: pd.DataFrame, features: list[str]) -> None:
    """Contrôles bloquants : une erreur ici veut dire qu'il ne faut pas livrer."""
    assert not df.duplicated(KEYS).any(), "doublons (arc, datetime)"
    steps = df.groupby("arc")["datetime"].diff().dropna()
    assert (steps == pd.Timedelta(hours=1)).all(), "grille horaire non continue"
    assert set(df["arc"]) == set(ARCS), "arcs manquants ou en trop"
    window = df[df["partie"] == "a_predire"]
    assert len(window) == WINDOW_ROWS, f"{len(window)} lignes à prédire au lieu de {WINDOW_ROWS}"
    assert window[TARGETS].isna().all().all(), "cibles présentes sur la fenêtre à prédire"
    valid = df["is_valid"]
    assert df.loc[valid, TARGETS].notna().all().all(), "ligne valide sans cible"
    assert df.loc[~valid, TARGETS].isna().all().all(), "cible présente sur une ligne non valide"
    assert (df[TARGETS].min() >= 0).all() and df["taux_occupation"].max() <= 100, "cible hors bornes"
    assert not [c for c in features if c.startswith(EXCLUDED_PREFIXES)], "feature exclue présente"
    assert window[features].notna().mean().min() > 0, "feature entièrement vide sur la fenêtre"


def export(df: pd.DataFrame, avec_chantiers: bool = False) -> tuple[pd.DataFrame, dict]:
    features = feature_columns(df)
    chantiers = [c for c in df if c.startswith("chantier_")] if avec_chantiers else []
    out = df[KEYS + TARGETS + SUIVI + features + chantiers].reset_index(drop=True)
    check(out, features)
    window = out[out["partie"] == "a_predire"]
    hist = out[out["partie"] == "historique"]
    meta = {
        "version": VERSION,
        "genere_le": str(pd.Timestamp.now(tz="UTC").floor("s")),
        "periode_utc": [str(out["datetime"].min()), str(out["datetime"].max())],
        "derniere_heure_observee_utc": str(hist.loc[hist["is_valid"], "datetime"].max()),
        "min_lag_hours": MIN_LAG_HOURS,
        "lignes": len(out),
        "lignes_par_arc_et_origine": out.groupby(["arc", "source_cible"]).size().unstack(fill_value=0).to_dict("index"),
        "lignes_valides_par_arc": out[out["is_valid"]].groupby("arc").size().to_dict(),
        "features": {f: [c for c in features if famille(c) == f] for f, _ in FAMILLES},
        "colonnes_hors_modele": chantiers,
        "fenetre_part_renseignee": window[features].notna().mean().round(3)[lambda s: s < 1].to_dict(),
    }
    return out, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exporte data/processed/dataset_modele.parquet")
    parser.add_argument("--avec-chantiers", action="store_true",
                        help="ajoute les 4 colonnes chantier_* (non utilisées par le modèle)")
    args = parser.parse_args()
    df, meta = export(pd.read_parquet(DATASET_PATH), avec_chantiers=args.avec_chantiers)
    df.to_parquet(DATASET_MODELE_PATH, index=False)
    DATASET_MODELE_PATH.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    print(f"{meta['version']} : {len(df)} lignes, {df.shape[1]} colonnes "
          f"({sum(len(v) for v in meta['features'].values())} features), "
          f"dernière heure observée {meta['derniere_heure_observee_utc']}")
    print("Features incomplètes sur la fenêtre :", meta["fenetre_part_renseignee"] or "aucune")
    print(f"-> {DATASET_MODELE_PATH}")
