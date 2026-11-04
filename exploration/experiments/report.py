"""Produit une note et un graphique à partir d'une campagne terminée, sans réentraîner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import TARGETS


def markdown_table(df):
    frame = df.copy()
    for col in frame.select_dtypes(include="number"):
        frame[col] = frame[col].map(lambda v: f"{v:.3f}")
    return "\n".join(["| " + " | ".join(frame.columns) + " |",
                       "| " + " | ".join(["---"] * len(frame.columns)) + " |"] +
                      ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None)])


def report(run, destination):
    run = Path(run)
    selected = json.loads((run / "selected.json").read_text())
    metadata = json.loads((run / "metadata.json").read_text())
    sections = ["# Résultats du stream 2", f"Campagne : `{run}`. Créée le {metadata['created_utc']}.",
                "Dataset : `dataset_modele.parquet`, empreinte SHA256 : `" + metadata['dataset_sha256'] + "`.",
                "## Réglage", markdown_table(pd.read_csv(run / "ranking_tuning.csv")),
                f"Configuration sélectionnée avant validation : **{selected['name']}**.",
                "Score interne : moyenne des six ratios RMSE / baseline ; plus petit = meilleur, 1 = baseline. "
                "Ce score n'est pas l'agrégation officielle du concours."]
    validation_tables = {}
    for stage, title in [("validation", "Validation temporelle récente"), ("diagnostic", "Diagnostic des semaines du 11 novembre")]:
        sections += [f"## {title}", markdown_table(pd.read_csv(run / f"scores_{stage}.csv"))]
        tables = []
        for folder in sorted((run / stage).iterdir()):
            df = pd.read_csv(folder / "rmse_by_arc.csv")
            df.insert(0, "candidate", folder.name)
            tables.append(df)
            if stage == "validation": validation_tables[folder.name] = df.set_index("arc")[TARGETS]
        sections += ["RMSE sur 24 h : débit en véhicules/heure ; occupation en points.", markdown_table(pd.concat(tables))]
    sections += ["## Interprétation et limites",
                 "La sélection repose seulement sur septembre 2026. La validation porte sur octobre, avec des fenêtres "
                 "distinctes. Les semaines du 11 novembre sont diagnostiques, déjà explorées par l'équipe. "
                 "Les périodes récentes ont également été étudiées dans les travaux précédents : ce n'est pas un test externe vierge.",
                 "Les cibles récentes des Champs-Elysées sont reconstruites. Les RMSE y mesurent la concordance au proxy. "
                 "Seules les heures avec une vérité valide sont notées ; voir coverage.csv pour les effectifs.",
                 "Avant adoption : vérifier les régressions par rue/cible, étendre la validation aux jeudis réguliers "
                 "2022–2024 et confirmer l'agrégation officielle. Le modèle de rendu n'est pas remplacé automatiquement."]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n\n".join(sections) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for target, ax in zip(TARGETS, axes):
        pd.DataFrame({name: table[target] for name, table in validation_tables.items()}).plot.bar(ax=ax, rot=15)
        ax.set_title("Débit" if target == TARGETS[0] else "Occupation")
        ax.set_ylabel("RMSE (véhicules/heure)" if target == TARGETS[0] else "RMSE (points)")
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Validation temporelle — RMSE sur 24 heures")
    fig.tight_layout()
    fig.savefig(run / "validation.png", dpi=160)
    plt.close(fig)
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="dossier de campagne terminée")
    parser.add_argument("--output", default="exploration/docs/resultats_stream2.md")
    args = parser.parse_args()
    report(args.run, args.output)
