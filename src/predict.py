"""Point d'entrée du rendu : produit outputs/output_<groupe>.csv (07/11 00:00 -> 11/11 23:00), vérifié et écrit par la fonction de test du brief.

Usage :
    python -m src.predict --group <groupe>              # part des données déjà présentes
    python -m src.predict --group <groupe> --refresh    # met d'abord à jour trafic et prévision météo, reconstruit le dataset
    python -m src.predict --group <groupe> --manif 2026-11-07   # baisse les prédictions d'un jour de manifestation
La cérémonie du 11/11 aux Champs-Elysées est toujours appliquée (voir `apply_ceremony`).
Entrée : data/processed/dataset_modele.parquet. Le cutoff est sa dernière heure observée (cible valide).
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, DATASET_MODELE_PATH, OUTPUTS_DIR, SUBMISSION_END, SUBMISSION_START, TARGETS, TRAFIC_DIR, TZ
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


# Fonction de test et d'export fournie par BCG X : brief « 2026 - Brief - LivraisonCo.pptx », slide 13 (« Format de
# l'output », fonction de test de l'output). Recopiée mot pour mot, seuls les 3 noms d'arc ont leurs accents : ne pas la modifier.
def test_format_and_export_output(output_df, output_directory, name_of_the_group):
    output_columns = {"arc": object, "datetime": object, "debit_horaire": float, "taux_occupation": float}
    # 1. Check relevant columns are in output dataframe
    assert sorted(list(output_df.columns)) == list(
        output_columns.keys()
    ), "Some columns are missing or unnecessary columns are in output"
    # 2. Check types
    for col, col_type in output_columns.items():
        assert output_df[col].dtype == col_type, f"Column {col} does not have type {col_type}"
    # 3. Check datetime string has right format
    try:
        output_df.datetime.apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M"))
    except ValueError as e:
        raise e
    # 4. Check `arc` columns has right values
    assert sorted(list(output_df["arc"].unique())) == [
        "Champs-Elysées",
        "Convention",
        "Saint-Pères",
    ], "Output does not have expected unique values for column `arc`"
    # 5. Check dataframe has right number of rows
    assert output_df.shape[0] == 360, f"Expected number of rows is 360, output has {output_df.shape[0]}"
    # 6. Export output
    output_path = Path(output_directory) / f"output_{name_of_the_group}.csv"
    print(f"[SAVE OUTPUT] Saving output here: {output_path}")
    output_df[output_columns.keys()].to_csv(output_path, sep=";")


def check_no_nan(df: pd.DataFrame) -> None:
    """Contrôle supplémentaire, absent du brief : aucune prédiction manquante."""
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
# Effet mesuré les samedis de manifestation des 17 et 24/10/2026 : × 0,65 à 0,8 ; aucun le 31/10. Effet incertain,
# donc facteur prudent : sur les samedis 2026, × 0,9 réduit la RMSE dès que la probabilité d'un effet dépasse 30 %
# environ, contre 60 % pour l'effet plein, qui triple l'erreur un samedi ordinaire (docs/4_rendu.md §1).
MANIF_FACTORS = {"Champs-Elysées": 0.9, "Convention": 0.9, "Saint-Pères": 0.9}
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


# Cérémonie du 11/11 : le haut des Champs-Elysées est fermé le matin. L'arc 4264 est « Barré » de 9 h à 13 h
# (heure de Paris) en 2021, 2023, 2024 et 2025, avec 12 à 223 véh/h au lieu de ~700. Ces heures sont exclues de
# l'entraînement (cible invalide), donc le modèle y prédit un férié ordinaire. La fermeture est certaine et connue
# à l'avance : on remplace ces heures par la moyenne mesurée les années passées, sans toucher au reste de la journée.
CEREMONY = {"arc": "Champs-Elysées", "arc_id": "4264", "day": "11-11", "hours": (9, 13)}


def ceremony_profile() -> pd.DataFrame:
    """Débit et occupation moyens par heure de cérémonie, sur les 11/11 passés (données brutes, heures barrées incluses)."""
    raw = pd.concat(pd.read_parquet(p, columns=["arc_id", "datetime", *TARGETS])
                    for p in sorted((TRAFIC_DIR / "historique").glob("*.parquet")))
    raw = raw[raw["arc_id"] == CEREMONY["arc_id"]]
    local = raw["datetime"].dt.tz_convert(TZ)
    hit = (local.dt.strftime("%m-%d") == CEREMONY["day"]) & local.dt.hour.between(*CEREMONY["hours"])
    return raw[hit].groupby(local[hit].dt.hour)[TARGETS].mean()


def apply_ceremony(pred: pd.DataFrame) -> pd.DataFrame:
    """Remplace les prédictions des Champs-Elysées pendant la cérémonie du 11/11 par `ceremony_profile`."""
    local = pred["datetime"].dt.tz_convert(TZ)
    hit = ((pred["arc"] == CEREMONY["arc"]) & (local.dt.strftime("%m-%d") == CEREMONY["day"])
           & local.dt.hour.between(*CEREMONY["hours"]))
    if not hit.any():
        return pred
    profile = ceremony_profile()
    pred = pred.copy()
    for col in TARGETS:
        pred.loc[hit, col] = local[hit].dt.hour.map(profile[col]).to_numpy()
    print(f"Cérémonie du 11/11 : Champs-Elysées de {CEREMONY['hours'][0]} h à {CEREMONY['hours'][1]} h remplacés par "
          f"la moyenne des années passées ({profile['debit_horaire'].round().astype(int).tolist()} véh/h)")
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
    pred = apply_ceremony(pred)
    if manif:
        pred = apply_manifestation(pred, manif)
    submission = to_submission(pred)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    try:
        check_no_nan(submission)
        test_format_and_export_output(submission, OUTPUTS_DIR, group)   # contrôle le format puis écrit le CSV
    except AssertionError as err:
        raise SystemExit(f"Contrôle du format BCG échoué : {err}. Aucun fichier écrit.")
    return submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ensemble", choices=list(MODELS))
    parser.add_argument("--group", required=True, help="nom du groupe, utilisé dans output_<groupe>.csv")
    parser.add_argument("--refresh", action="store_true", help="met à jour trafic et météo puis reconstruit le dataset")
    parser.add_argument("--manif", metavar="AAAA-MM-JJ", help="jour de manifestation : prédictions baissées de 12 h à 23 h")
    args = parser.parse_args()
    if args.refresh:
        refresh()
    predict(args.model, args.group, args.manif)
