"""Quelles colonnes servent vraiment ? Trois mesures, de la plus naïve à la plus fiable.

1. Matrice de corrélation (Pearson) : liens linéaires entre colonnes, et avec les cibles.
   Limite : ne voit ni les effets non linéaires ni les interactions (ex. férié x heure).
2. Importance « gain » de XGBoost : combien chaque colonne a servi à couper les arbres.
   Limite : favorise les colonnes très variées ; deux colonnes redondantes se partagent le crédit.
3. Importance par permutation, sur des semaines de test jamais vues : on mélange une colonne (ou une famille
   entière) dans le test et on mesure de combien l'erreur augmente. C'est la mesure à croire.
   Mélanger la famille entière évite le piège des colonnes redondantes (lag S-1 / S-2 / S-3).

Usage : python -m exploration.essais_victor.analyse_features
Sorties : outputs/experiments/analyse_features/*.png et *.csv (ignorés par git)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from src.config import OUTPUTS_DIR, TARGETS
from exploration.essais_victor.essai_xgb import DATASET, MODES, PARAMS, prepare, rmse
from src.evaluate import parse_cutoffs, window
from src.models.preprocessing import feature_columns

SORTIE = OUTPUTS_DIR / "experiments" / "analyse_features"
CUTOFFS = MODES["automne"][::3]       # 9 jeudis d'automne 2024-2025
N_REPET = 3                           # nombre de mélanges par colonne

# Couleurs (palette de référence) : bleu = corrélation négative, rouge = positive, gris = nulle
BLEU, GRIS, ROUGE, ENCRE, ENCRE_2, FOND = "#1c5cab", "#f0efec", "#e34948", "#0b0b0b", "#52514e", "#fcfcfb"
BARRE = "#2a78d6"


def famille(col):
    if col == "arc":
        return "arc"
    if "_lag" in col or col.endswith("_niveau28j") and not col.startswith("vacances"):
        return "historique trafic"
    if col.startswith("meteo_"):
        return "météo"
    if col.startswith("evenement"):
        return "événements"
    if col.startswith("vacances") or col.startswith("jours_"):
        return "calendrier scolaire"
    if col in ("jo_2024", "post_jo_2024", "post_30kmh"):
        return "régimes (JO, 30 km/h)"
    return "calendrier"


def score(y, y_hat, arcs):
    """RMSE / moyenne par arc, moyenné : chaque rue compte pareil."""
    return np.mean([rmse(y[arcs == a], y_hat[arcs == a]) / y[arcs == a].mean() for a in np.unique(arcs)])


def style(ax):
    ax.set_facecolor(FOND)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d6d5d0")
    ax.tick_params(colors=ENCRE_2, labelsize=9)


def matrice_correlation(df, feats):
    cols = sorted(feats, key=lambda c: (famille(c), c)) + TARGETS
    corr = df[df["is_valid"].astype(bool)][cols].astype(float).corr()
    corr.to_csv(SORTIE / "correlation.csv")

    cmap = LinearSegmentedColormap.from_list("div", [BLEU, GRIS, ROUGE])
    fig, ax = plt.subplots(figsize=(15, 13), facecolor=FOND)
    im = ax.imshow(corr.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=8, color=ENCRE_2)
    ax.set_yticks(range(len(cols)), cols, fontsize=8, color=ENCRE_2)
    # traits fins entre familles
    fams = [famille(c) if c not in TARGETS else "cibles" for c in cols]
    for i in range(1, len(cols)):
        if fams[i] != fams[i - 1]:
            ax.axhline(i - 0.5, color=FOND, lw=2)
            ax.axvline(i - 0.5, color=FOND, lw=2)
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("corrélation de Pearson", color=ENCRE_2)
    cb.outline.set_visible(False)
    ax.set_title("Corrélation entre colonnes (triées par famille ; 2 dernières = cibles)", color=ENCRE, loc="left")
    fig.tight_layout()
    fig.savefig(SORTIE / "correlation.png", dpi=110)
    plt.close(fig)
    return corr


def permutations(df, feats, preprocessor):
    rng = np.random.default_rng(0)
    groupes = {f: [c for c in feats if famille(c) == f] for f in dict.fromkeys(famille(c) for c in feats)}
    rows, gains = [], []
    for cutoff in parse_cutoffs(CUTOFFS):
        start, end = window(cutoff)
        train = df[(df["datetime"] <= cutoff) & df["is_valid"]]
        test = df[(df["datetime"] >= start) & (df["datetime"] <= end) & df["is_valid"]]
        if test.empty:
            continue
        arcs = test["arc"].to_numpy()
        for cible in TARGETS:
            model = make_pipeline(preprocessor, XGBRegressor(**PARAMS)).fit(train[feats], train[cible])
            y = test[cible].to_numpy()
            base = score(y, model.predict(test[feats]), arcs)

            booster = model[-1].get_booster()
            noms = model[0].get_feature_names_out()
            g = booster.get_score(importance_type="gain")
            gains.append(pd.Series({n.split("__", 1)[1]: g.get(f"f{i}", 0.0) for i, n in enumerate(noms)}, name=cible))

            a_tester = [("colonne", c, [c]) for c in feats] + [("famille", f, cs) for f, cs in groupes.items()]
            for niveau, nom, cs in a_tester:
                hausses = []
                for _ in range(N_REPET):
                    X = test[feats].copy()
                    perm = rng.permutation(len(X))
                    X[cs] = X[cs].to_numpy()[perm]           # même mélange pour toute la famille
                    hausses.append(score(y, model.predict(X), arcs) / base - 1)
                rows.append({"cutoff": cutoff.date(), "cible": cible, "niveau": niveau, "nom": nom,
                             "hausse_erreur_%": 100 * np.mean(hausses)})
        print(f"  jeudi {cutoff.date()} fait")
    perm = pd.DataFrame(rows)
    perm.to_csv(SORTIE / "permutation.csv", index=False)
    gain = pd.concat(gains, axis=1).T.groupby(level=0).mean().T
    gain = gain / gain.sum() * 100
    gain.to_csv(SORTIE / "importance_gain.csv")
    return perm, gain


def barres(table, titre, fichier, xlabel):
    """Une rangée de barres horizontales par cible (même axe x pour comparer)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, max(3.5, 0.28 * len(table) + 1.5)), facecolor=FOND, sharey=True)
    xmax = table.to_numpy().max() * 1.15
    for ax, cible in zip(axes, TARGETS):
        style(ax)
        v = table[cible]
        ax.barh(range(len(v)), v.to_numpy(), color=BARRE, height=0.7)
        ax.axvline(0, color="#d6d5d0", lw=1)
        ax.set_yticks(range(len(v)), v.index, color=ENCRE_2)
        ax.invert_yaxis()
        ax.set_xlim(min(0, table.to_numpy().min() * 1.15), xmax)
        ax.set_title(cible, color=ENCRE, loc="left", fontsize=11)
        ax.set_xlabel(xlabel, color=ENCRE_2)
        ax.grid(axis="x", color="#ebeae6", lw=0.8)
        ax.set_axisbelow(True)
        for i, val in enumerate(v.to_numpy()):
            ax.text(val + xmax * 0.01, i, f"{val:.1f}", va="center", fontsize=8, color=ENCRE_2)
    fig.suptitle(titre, color=ENCRE, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(SORTIE / fichier, dpi=110)
    plt.close(fig)


def main():
    SORTIE.mkdir(parents=True, exist_ok=True)
    df = prepare(pd.read_parquet(DATASET))
    cat_cols, num_cols = ["arc"], feature_columns(df)
    feats = cat_cols + num_cols
    preprocessor = make_column_transformer((OneHotEncoder(handle_unknown="ignore"), cat_cols),
                                           ("passthrough", num_cols))

    corr = matrice_correlation(df, num_cols)
    print("Corrélation avec les cibles (|r| décroissant, 15 premières) :")
    lien = corr.loc[num_cols, TARGETS]
    print(lien.reindex(lien.abs().max(axis=1).sort_values(ascending=False).index).head(15).round(2).to_string())

    perm, gain = permutations(df, feats, preprocessor)
    fam = perm[perm["niveau"] == "famille"].groupby(["nom", "cible"])["hausse_erreur_%"].mean().unstack()
    fam = fam.sort_values(TARGETS[0], ascending=False)
    col = perm[perm["niveau"] == "colonne"].groupby(["nom", "cible"])["hausse_erreur_%"].mean().unstack()
    col = col.reindex(col.max(axis=1).sort_values(ascending=False).index)

    barres(fam, "Hausse de l'erreur quand on mélange une famille de colonnes (test : 9 jeudis d'automne)",
           "permutation_familles.png", "hausse de l'erreur (%)")
    barres(col, "Hausse de l'erreur quand on mélange une seule colonne (les autres restent intactes)",
           "permutation_colonnes.png", "hausse de l'erreur (%)")
    gain_tri = gain.reindex(gain.max(axis=1).sort_values(ascending=False).index)
    barres(gain_tri, "Importance « gain » de XGBoost (part du gain total, %)", "importance_gain.png", "part du gain (%)")

    print("\nPermutation par famille (hausse de l'erreur en %, > 0 = la famille sert) :")
    print(fam.round(1).to_string())
    print("\nPermutation par colonne, 10 plus utiles :")
    print(col.head(10).round(1).to_string())
    print("\nColonnes quasi inutiles (hausse < 0,2 % sur les deux cibles) :")
    print(", ".join(col[(col < 0.2).all(axis=1)].index))
    print(f"\nGraphiques dans {SORTIE}")


if __name__ == "__main__":
    main()
