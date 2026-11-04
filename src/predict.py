"""Stream C : produit le CSV de rendu (07/11 00:00 -> 11/11 23:00) et le vérifie avec le test du brief.

Usage : python -m src.predict --model baseline --group <nom_du_groupe>
Entrée : data/processed/dataset.parquet. Le cutoff est sa dernière heure observée (cible valide).
"""

import argparse
from datetime import datetime

import pandas as pd

from src.config import ARCS, DATASET_PATH, OUTPUTS_DIR, SUBMISSION_END, SUBMISSION_START, TARGETS, TZ
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="baseline", choices=list(MODELS))
    parser.add_argument("--group", required=True, help="nom du groupe, utilisé dans output_<groupe>.csv")
    args = parser.parse_args()

    traffic = pd.read_parquet(DATASET_PATH)
    cutoff = traffic.loc[traffic["is_valid"], "datetime"].max()
    start = pd.Timestamp(SUBMISSION_START, tz=TZ).tz_convert("UTC")
    end = pd.Timestamp(SUBMISSION_END, tz=TZ).tz_convert("UTC")
    print(f"Cutoff : {cutoff.tz_convert(TZ):%Y-%m-%d %H:%M} (Paris), horizon max {(end - cutoff) / pd.Timedelta(hours=1):.0f} h")

    model = MODELS[args.model]().fit(traffic, cutoff)
    submission = to_submission(model.predict(traffic, cutoff, start, end))
    check_submission(submission)

    path = OUTPUTS_DIR / f"output_{args.group}.csv"
    submission.to_csv(path, sep=";")
    print(f"-> {path}")
