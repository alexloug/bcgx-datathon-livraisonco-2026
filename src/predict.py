"""Point d'entrée du rendu : produit outputs/output_<groupe>.csv (07/11 00:00 -> 11/11 23:00), vérifié avec le test du brief.

Usage :
    python -m src.predict --group <groupe>              # part des données déjà présentes
    python -m src.predict --group <groupe> --refresh    # met d'abord à jour trafic et prévision météo, reconstruit le dataset
    python -m src.predict --group <groupe> --manif 2026-11-07   # baisse les prédictions d'un jour de manifestation
Entrée : data/processed/dataset_modele.parquet. Le cutoff est sa dernière heure observée (cible valide).
"""

import argparse
from datetime import datetime

import pandas as pd

from src.config import ARCS, DATA_DIR, DATASET_MODELE_PATH, OUTPUTS_DIR, SUBMISSION_END, SUBMISSION_START, TARGETS, TZ
from src.data import build_dataset, export_dataset, update
from src.models.registry import MODELS

OUTPUT_COLUMNS = {"arc": object, "datetime": object, "debit_horaire": float, "taux_occupation": float}


def to_submission(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    out["datetime"] = out["datetime"].dt.tz_convert(TZ).dt.strftime("%Y-%m-%d %H:%M")
    # Le test du brief exige le dtype object (pandas >= 3 infère `str` par défaut).
    for col in ["arc", "datetime"]:
        out[col] = out[col].astype(object)
    for col in TARGETS:
        out[col] = out[col].astype(float)
    return out[list(OUTPUT_COLUMNS)].sort_values(["arc", "datetime"]).reset_index(drop=True)


def check_submission(df: pd.DataFrame) -> None:
    """Reprise des contrôles de la fonction de test fournie dans le brief (slide 13), plus l'absence de NaN."""
    assert sorted(df.columns) == sorted(OUTPUT_COLUMNS), "Colonnes manquantes ou en trop"
    for col, col_type in OUTPUT_COLUMNS.items():
        assert df[col].dtype == col_type, f"La colonne {col} n'est pas de type {col_type}"
    df["datetime"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M"))
    assert sorted(df["arc"].unique()) == sorted(ARCS), "Valeurs inattendues dans la colonne arc"
    assert df.shape[0] == 360, f"360 lignes attendues, {df.shape[0]} obtenues"
    assert not df[TARGETS].isna().any().any(), "Des prédictions sont manquantes (NaN)"


REQUIRED_SOURCES = ["trafic/recent.parquet", "meteo/prevision.parquet", "meteo/prevision_historique.parquet",
                    "calendrier/jours_feries.parquet", "calendrier/vacances_scolaires.parquet",
                    "evenements/evenements.parquet"]


def refresh() -> None:
    """Trafic et prévision météo à jour, puis dataset reconstruit et exporté (contrôles bloquants)."""
    if not update.main(["trafic", "meteo"]):
        raise SystemExit("Échec de la mise à jour des données (voir ci-dessus) : rien n'a été reconstruit.")
    check_sources()
    df, meta = build_dataset.build_dataset()
    build_dataset.save(df, meta)
    export_meta = export_dataset.write()
    print(f"Dernière heure observée : {export_meta['derniere_heure_observee_utc']} (UTC)")
    print("Features incomplètes sur la fenêtre :", export_meta["fenetre_part_renseignee"] or "aucune")


def check_sources() -> None:
    missing = [p for p in REQUIRED_SOURCES if not (DATA_DIR / p).exists()]
    if missing:
        raise SystemExit(f"Données manquantes dans data/ : {', '.join(missing)}. "
                         "Lancer `python -m src.data.update`, puis relancer.")


# Jour de manifestation : débit et occupation prédits multipliés par ce facteur de 12 h à 23 h (heure de Paris).
# Facteur = mesuré / prédit, en moyenne sur les samedis de manifestation des 17 et 24/10/2026 (docs/4_rendu.md §1).
MANIF_FACTORS = {"Champs-Elysées": 0.65, "Convention": 0.8, "Saint-Pères": 0.65}
MANIF_HOURS = (12, 23)


def apply_manifestation(pred: pd.DataFrame, day: str) -> pd.DataFrame:
    """Le `day` d'une manifestation (AAAA-MM-JJ), multiplie les 2 prédictions par MANIF_FACTORS sur MANIF_HOURS.
    Le modèle ne peut pas apprendre seul cet effet : trop peu de samedis de manifestation dans l'historique."""
    local = pred["datetime"].dt.tz_convert(TZ)
    hit = (local.dt.strftime("%Y-%m-%d") == day) & local.dt.hour.between(*MANIF_HOURS)
    if not hit.any():
        raise SystemExit(f"--manif {day} : aucune heure de la fenêtre ce jour-là.")
    pred = pred.copy()
    for col in TARGETS:
        pred.loc[hit, col] *= pred.loc[hit, "arc"].map(MANIF_FACTORS)
    print(f"Manifestation le {day} : prédictions × {MANIF_FACTORS} de {MANIF_HOURS[0]} h à {MANIF_HOURS[1]} h")
    return pred


def predict(model_name: str, group: str, manif: str | None = None) -> pd.DataFrame:
    if not DATASET_MODELE_PATH.exists():
        raise SystemExit(f"{DATASET_MODELE_PATH} introuvable : lancer avec --refresh (ou, à la main, "
                         "`python -m src.data.build_dataset` puis `python -m src.data.export_dataset`).")
    traffic = pd.read_parquet(DATASET_MODELE_PATH)
    cutoff = traffic.loc[traffic["is_valid"], "datetime"].max()
    start = pd.Timestamp(SUBMISSION_START, tz=TZ).tz_convert("UTC")
    end = pd.Timestamp(SUBMISSION_END, tz=TZ).tz_convert("UTC")
    print(f"Cutoff : {cutoff.tz_convert(TZ):%Y-%m-%d %H:%M} (Paris), horizon max {(end - cutoff) / pd.Timedelta(hours=1):.0f} h")
    model = MODELS[model_name]().fit(traffic, cutoff)
    pred = model.predict(traffic, cutoff, start, end)
    if manif:
        pred = apply_manifestation(pred, manif)
    submission = to_submission(pred)
    try:
        check_submission(submission)
    except AssertionError as err:
        raise SystemExit(f"Contrôle du format BCG échoué : {err}. Aucun fichier écrit.")
    path = OUTPUTS_DIR / f"output_{group}.csv"
    OUTPUTS_DIR.mkdir(exist_ok=True)
    submission.to_csv(path, sep=";")
    print(f"-> {path}")
    return submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgb", choices=list(MODELS))
    parser.add_argument("--group", required=True, help="nom du groupe, utilisé dans output_<groupe>.csv")
    parser.add_argument("--refresh", action="store_true", help="met à jour trafic et météo puis reconstruit le dataset")
    parser.add_argument("--manif", metavar="AAAA-MM-JJ", help="jour de manifestation : prédictions baissées de 12 h à 23 h")
    args = parser.parse_args()
    if args.refresh:
        refresh()
    predict(args.model, args.group, args.manif)
