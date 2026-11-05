"""Essai : moyenner plusieurs modèles (ensemble) et jouer sur l'aléatoire.

L'aléatoire dans XGBoost / LightGBM : `subsample` (part des lignes tirée au hasard pour chaque arbre) et
`colsample_bytree` / `colsample_bynode` (part des colonnes tirée au hasard). La graine `random_state` fixe ce tirage :
changer de graine = un autre modèle, un peu différent. C'est le même principe que la Random Forest, sauf que la
Random Forest fait des arbres profonds INDÉPENDANTS qu'elle moyenne, alors que le boosting fait des arbres
SUCCESSIFS, chacun corrigeant le précédent.

Variantes comparées (mêmes jeudis, mêmes features) :
    xgb               : XGBoost de référence, graine 0
    xgb_x5            : moyenne de 5 XGBoost de graines différentes (« seed bagging »)
    xgb_alea          : XGBoost plus aléatoire (60 % des lignes, 50 % des colonnes à chaque nœud, comme une RF)
    xgb_alea_x5       : moyenne de 5 XGBoost plus aléatoires
    lgbm              : LightGBM de référence
    xgb+lgbm          : moyenne XGBoost + LightGBM
    random_forest     : Random Forest de sklearn
    xgb+lgbm+rf       : moyenne des trois
Bonus : l'écart entre les 5 graines de xgb mesure le bruit de nos tests.

Usage : python -m exploration.essais_victor.essai_ensemble
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from src.config import TARGETS
from exploration.essais_victor.essai_poids import CUTOFFS          # automne 2024-2025 (1 sur 2) + octobre 2026
from exploration.essais_victor.essai_xgb import DATASET, PARAMS, prepare, rmse
from src.evaluate import parse_cutoffs, window
from exploration.models.gbm import PARAMS as LGBM_PARAMS
from src.models.preprocessing import feature_columns

GRAINES = [0, 1, 2, 3, 4]
ALEA = dict(subsample=0.6, colsample_bytree=1.0, colsample_bynode=0.5)
RF = dict(n_estimators=200, min_samples_leaf=5, max_features=0.33, max_samples=0.5, n_jobs=-1, random_state=0)


def main():
    df = prepare(pd.read_parquet(DATASET))
    cat_cols, num_cols = ["arc"], feature_columns(df)
    feats = cat_cols + num_cols
    preprocessor = make_column_transformer((OneHotEncoder(handle_unknown="ignore"), cat_cols),
                                           ("passthrough", num_cols))

    def predire(regresseur, train, test, cible):
        model = make_pipeline(preprocessor, regresseur).fit(train[feats], train[cible])
        return model.predict(test[feats])

    rows, graines = [], []
    for cutoff in parse_cutoffs(CUTOFFS):
        start, end = window(cutoff)
        train = df[(df["datetime"] <= cutoff) & df["is_valid"]]
        test = df[(df["datetime"] >= start) & (df["datetime"] <= end) & df["is_valid"]]
        if test.empty:
            continue
        # Random Forest de sklearn : NaN acceptés depuis sklearn 1.4, mais on les remplace par -1 par sécurité
        train_rf, test_rf = train.copy(), test.copy()
        train_rf[num_cols], test_rf[num_cols] = train_rf[num_cols].fillna(-1), test_rf[num_cols].fillna(-1)

        for cible in TARGETS:
            p = {}
            xgbs = [predire(XGBRegressor(**{**PARAMS, "random_state": g}), train, test, cible) for g in GRAINES]
            alea = [predire(XGBRegressor(**{**PARAMS, **ALEA, "random_state": g}), train, test, cible) for g in GRAINES]
            p["xgb"] = xgbs[0]
            p["xgb_x5"] = np.mean(xgbs, axis=0)
            p["xgb_alea"] = alea[0]
            p["xgb_alea_x5"] = np.mean(alea, axis=0)
            p["lgbm"] = predire(LGBMRegressor(**LGBM_PARAMS), train, test, cible)
            p["xgb+lgbm"] = (p["xgb"] + p["lgbm"]) / 2
            p["random_forest"] = predire(RandomForestRegressor(**RF), train_rf, test_rf, cible)
            p["xgb+lgbm+rf"] = (p["xgb"] + p["lgbm"] + p["random_forest"]) / 3

            for arc in test["arc"].unique():
                m = (test["arc"] == arc).to_numpy()
                y = test[cible].to_numpy()[m]
                for nom, y_hat in p.items():
                    rows.append({"cutoff": cutoff.date(), "variante": nom, "arc": arc, "cible": cible,
                                 "rmse": rmse(y, y_hat[m])})
                for g, y_hat in zip(GRAINES, xgbs):
                    graines.append({"cutoff": cutoff.date(), "graine": g, "arc": arc, "cible": cible,
                                    "rmse": rmse(y, y_hat[m])})
        print(f"  jeudi {cutoff.date()} fait")

    res = pd.DataFrame(rows)
    ordre = list(dict.fromkeys(res["variante"]))
    table = res.groupby(["cible", "arc", "variante"])["rmse"].mean().unstack()[ordre]
    print(f"\nRMSE moyen sur {res['cutoff'].nunique()} jeudis :")
    print(table.round(2).to_string())
    print("\nGain en % par rapport à xgb seul (> 0 = mieux) :")
    print(((1 - table.div(table["xgb"], axis=0)) * 100).round(1).to_string())
    piv = res.pivot_table(index=["cutoff", "arc", "cible"], columns="variante", values="rmse")
    print("\nPart des (jeudi, rue, cible) où la variante fait mieux que xgb seul :")
    print(pd.Series({v: (piv[v] < piv["xgb"]).mean() for v in ordre if v != "xgb"}).map("{:.0%}".format).to_string())

    # Bruit : même modèle, seule la graine change
    g = pd.DataFrame(graines).groupby(["cible", "arc", "graine"])["rmse"].mean().unstack()
    bruit = pd.DataFrame({"min": g.min(axis=1), "max": g.max(axis=1),
                          "ecart_max_%": (g.max(axis=1) / g.min(axis=1) - 1) * 100})
    print("\nBruit dû à la graine seule (RMSE moyen des 5 graines de xgb) :")
    print(bruit.round(2).to_string())


if __name__ == "__main__":
    main()
