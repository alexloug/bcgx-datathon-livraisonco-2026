"""Stream 2 : réglage, validation temporelle et diagnostics, avec artefacts par expérience."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import TARGETS
from src.evaluate import backtest, parse_cutoffs, rmse_table
from src.models.preprocessing import feature_columns


def select_features(dataset, candidate):
    """Ne retire que des entrées du modèle ; les clés et cibles sont protégées."""
    prefixes = tuple(candidate.get("drop_prefixes", []))
    drop = [c for c in feature_columns(dataset) if prefixes and c.startswith(prefixes)]
    return dataset.drop(columns=drop)


def relative_score(metrics, baseline):
    """Moyenne des six ratios RMSE / baseline ; convention interne, pas score officiel."""
    ratios = metrics[TARGETS].div(baseline[TARGETS])
    if len(metrics) != 3 or not np.isfinite(ratios.to_numpy()).all():
        raise ValueError("Le score nécessite trois arcs et des RMSE baseline strictement positives")
    return float(ratios.to_numpy().mean())


def evaluate(candidate, dataset, dates, directory):
    directory.mkdir(parents=True)
    data = select_features(dataset, candidate)
    predictions = backtest(candidate["model"], data, parse_cutoffs(dates), candidate.get("params"))
    predictions.to_parquet(directory / "predictions.parquet", index=False)
    metrics = rmse_table(predictions, by=("arc",))
    metrics.to_csv(directory / "rmse_by_arc.csv")
    rmse_table(predictions, by=("cutoff", "arc")).to_csv(directory / "rmse_by_fold.csv")
    rmse_table(predictions).to_csv(directory / "rmse_by_horizon.csv")
    if "source_cible" in predictions:
        rmse_table(predictions, by=("arc", "source_cible")).to_csv(directory / "rmse_by_source.csv")
    predictions.groupby(["cutoff", "arc"]).size().rename("n_scored").to_csv(directory / "coverage.csv")
    (directory / "candidate.json").write_text(json.dumps(candidate, indent=2))
    print(f"{directory.name}: {candidate['name']} terminé", flush=True)
    return metrics


def run(config_path, output):
    config = json.loads(Path(config_path).read_text())
    candidates = config["candidates"]
    names = [c["name"] for c in candidates]
    if len(names) != len(set(names)) or any(Path(n).name != n or n in (".", "..") for n in names):
        raise ValueError("Noms d'expériences invalides ou dupliqués")
    baseline = next(c for c in candidates if c["name"] == "baseline" and c["model"] == "baseline")
    reference = next(c for c in candidates if c["name"] == "xgb_reference")
    tuning, validation = config["tuning_cutoffs"], config["validation_cutoffs"]
    if not tuning or not validation or max(tuning) >= min(validation):
        raise ValueError("La validation doit suivre strictement les cutoffs de réglage")
    # Les fenêtres font cinq jours : elles doivent aussi rester disjointes.
    if (pd.Timestamp(min(validation)) - pd.Timestamp(max(tuning))).days < 6:
        raise ValueError("Fenêtres de réglage et validation superposées")
    dataset_path = Path(config["dataset"])
    dataset = pd.read_parquet(dataset_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "git_revision": revision, "git_status": dirty, "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                "packages": {p: importlib.metadata.version(p) for p in ["pandas", "numpy", "xgboost", "scikit-learn"]}}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    # L'état Git peut contenir des fichiers non commités : conserver le code exécuté.
    shutil.copytree("src", output / "source", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    tables = {}
    for candidate in [baseline] + [c for c in candidates if c != baseline]:
        tables[candidate["name"]] = evaluate(candidate, dataset, tuning, output / "tuning" / candidate["name"])
    ranking = pd.DataFrame([{"candidate": name, "relative_rmse": relative_score(table, tables["baseline"])}
                            for name, table in tables.items()]).sort_values("relative_rmse")
    ranking.to_csv(output / "ranking_tuning.csv", index=False)
    winner = next(c for c in candidates if c["name"] == ranking.iloc[0]["candidate"])
    # Le gagnant est figé avant d'ouvrir la validation ; on ne re-classe pas tous les candidats dessus.
    (output / "selected.json").write_text(json.dumps(winner, indent=2))
    selected = {c["name"]: c for c in [baseline, reference, winner]}
    for stage, dates in [("validation", validation), ("diagnostic", config.get("diagnostic_cutoffs", []))]:
        if not dates:
            continue
        tables = {name: evaluate(c, dataset, dates, output / stage / name) for name, c in selected.items()}
        pd.DataFrame([{"candidate": name, "relative_rmse": relative_score(table, tables["baseline"])}
                      for name, table in tables.items()]).to_csv(output / f"scores_{stage}.csv", index=False)
    print(f"Configuration sélectionnée sur le réglage : {winner['name']} ; résultats : {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stream2.json")
    parser.add_argument("--output", default=f"outputs/experiments/{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}")
    args = parser.parse_args()
    run(args.config, args.output)
