"""Recherche aléatoire des hyperparamètres de XGBoost (RandomizedSearchCV), avec un découpage temporel.

Usage (depuis la racine, venv actif) :
    python -m exploration.essais_victor.recherche_xgb --mode rapide      # ~10 min : 8 jeudis, 10 combinaisons
    python -m exploration.essais_victor.recherche_xgb --mode nuit        # 2 h et plus : 38 jeudis, 40 combinaisons
    python -m exploration.essais_victor.recherche_xgb --mode nuit --n-iter 80    # encore plus de combinaisons

Étapes :
    1. Réglage : RandomizedSearchCV sur les jeudis réguliers 2022-2024 (un modèle par cible).
       La « cv » n'est pas aléatoire : chaque pli = train jusqu'au jeudi 23h, test les 5 jours suivants.
    2. Validation : on compare la meilleure combinaison et la référence sur les jeudis d'automne
       2024-2025, jamais vus pendant le réglage.
Résultats dans outputs/experiments/recherche_<mode>/ (dossier ignoré par git).
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import make_column_transformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from src.config import OUTPUTS_DIR, TARGETS
from exploration.essais_victor.essai_xgb import DATASET, MODES, PARAMS, prepare, rmse
from src.evaluate import parse_cutoffs, window
from src.models.preprocessing import feature_columns

# Espace de recherche : on tire au hasard dans ces lois
PARAM_DISTRIBUTIONS = {
    "xgbregressor__n_estimators": randint(200, 1500),
    "xgbregressor__learning_rate": loguniform(0.01, 0.2),
    "xgbregressor__max_depth": randint(3, 11),            # profondeur des arbres : 3 à 10
    "xgbregressor__min_child_weight": loguniform(1, 50),  # taille minimale d'une feuille
    "xgbregressor__subsample": uniform(0.5, 0.5),         # part des lignes par arbre : 0,5 à 1
    "xgbregressor__colsample_bytree": uniform(0.4, 0.6),  # part des colonnes par arbre : 0,4 à 1
    "xgbregressor__reg_lambda": loguniform(0.1, 30),      # régularisation L2
}

REGLAGES = {
    #           jeudis de réglage          combinaisons   jeudis de validation
    "rapide": dict(reglage=MODES["reguliers"][::5], n_iter=10, validation=MODES["automne"][::3]),
    "nuit":   dict(reglage=MODES["reguliers"],      n_iter=40, validation=MODES["automne"]),
}


def decoupages(df, cutoffs):
    """Liste de (indices train, indices test) : un pli par jeudi, sans regarder le futur."""
    plis = []
    for cutoff in parse_cutoffs(cutoffs):
        start, end = window(cutoff)
        train = np.flatnonzero(df["datetime"] <= cutoff)
        test = np.flatnonzero((df["datetime"] >= start) & (df["datetime"] <= end))
        if len(test):
            plis.append((train, test))
    return plis


def score_rmse_relatif(estimator, X, y):
    """RMSE / moyenne de la cible, calculé par arc puis moyenné : chaque rue compte pareil.
    (sinon les Champs-Elysées, au débit 2 fois plus élevé, dicteraient le choix). Négatif : sklearn maximise."""
    y_hat = estimator.predict(X)
    scores = [rmse(y[m], y_hat[m]) / y[m].mean() for m in [(X["arc"] == a).to_numpy() for a in X["arc"].unique()]]
    return -np.mean(scores)


def valider(df, feats, preprocessor, cible, cutoffs, configs):
    """RMSE par arc sur les jeudis de validation, pour chaque configuration {nom: params}."""
    rows = []
    for train_idx, test_idx in decoupages(df, cutoffs):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        for nom, params in configs.items():
            model = make_pipeline(preprocessor, XGBRegressor(**params)).fit(train[feats], train[cible])
            y_hat = model.predict(test[feats])
            for arc in test["arc"].unique():
                m = (test["arc"] == arc).to_numpy()
                rows.append({"config": nom, "arc": arc, "rmse": rmse(test[cible].to_numpy()[m], y_hat[m])})
    return pd.DataFrame(rows).groupby(["arc", "config"])["rmse"].mean().unstack()


def main(mode, n_iter=None):
    reglage = REGLAGES[mode]
    n_iter = n_iter or reglage["n_iter"]
    sortie = OUTPUTS_DIR / "experiments" / f"recherche_{mode}"
    sortie.mkdir(parents=True, exist_ok=True)

    df = prepare(pd.read_parquet(DATASET))
    df = df[df["is_valid"] & (df["partie"] == "historique")].reset_index(drop=True)   # heures avec une vraie valeur

    cat_cols = ["arc"]
    num_cols = feature_columns(df)
    feats = cat_cols + num_cols
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("passthrough", num_cols))

    plis = decoupages(df, reglage["reglage"])
    print(f"Mode {mode} : {len(plis)} jeudis de réglage, {n_iter} combinaisons, 2 cibles "
          f"=> {len(plis) * n_iter * 2} entraînements")

    resultats = {}
    for cible in TARGETS:
        t0 = time.time()
        model = make_pipeline(preprocessor, XGBRegressor(tree_method="hist", n_jobs=-1, random_state=42))
        search = RandomizedSearchCV(model, PARAM_DISTRIBUTIONS, n_iter=n_iter, cv=plis,
                                    scoring=score_rmse_relatif, refit=False, random_state=42, verbose=1)
        search.fit(df[feats], df[cible])

        cv = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
        cv.to_csv(sortie / f"cv_results_{cible}.csv", index=False)
        best = {k.replace("xgbregressor__", ""): v for k, v in search.best_params_.items()}
        best = {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, np.floating) else v)
                for k, v in best.items()}

        print(f"\n=== {cible} ({(time.time() - t0) / 60:.1f} min)")
        print("Top 5 (score = RMSE / moyenne, moyenné sur les arcs ; plus petit = mieux) :")
        cols = [c for c in cv.columns if c.startswith("param_")]
        top = cv[cols + ["mean_test_score", "std_test_score"]].head(5).copy()
        top["mean_test_score"] = -top["mean_test_score"]
        top.columns = [c.replace("param_xgbregressor__", "") for c in top.columns]
        print(top.round(3).to_string(index=False))

        # Validation sur l'automne : meilleure combinaison contre la référence actuelle
        ref = {k: v for k, v in PARAMS.items()}
        configs = {"reference": ref, "meilleure": {**ref, **best}}
        val = valider(df, feats, preprocessor, cible, reglage["validation"], configs)
        val["gain_%"] = (1 - val["meilleure"] / val["reference"]) * 100
        print(f"\nValidation automne 2024-2025 ({len(decoupages(df, reglage['validation']))} jeudis), RMSE par arc :")
        print(val.round(2).to_string())

        resultats[cible] = {"meilleurs_params": best, "score_reglage": -search.best_score_,
                            "validation": val.round(3).reset_index().to_dict(orient="records")}

    (sortie / "resultats.json").write_text(json.dumps(resultats, indent=2, ensure_ascii=False))
    print(f"\nRésultats enregistrés dans {sortie}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(REGLAGES), default="rapide")
    parser.add_argument("--n-iter", type=int, help="nombre de combinaisons (remplace celui du mode)")
    args = parser.parse_args()
    main(args.mode, args.n_iter)
