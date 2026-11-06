"""Sélection des features du XGBoost par ablation de groupes, en backtest -> outputs/selection_features*.

Pour chaque groupe de features, on réentraîne `xgb` sans ce groupe sur les mêmes cutoffs et on compare la RMSE à
celle du modèle complet. Un groupe qu'on peut retirer sans perte n'apporte que du bruit (ou un biais) : il sort de
la short-list.

Pour ne pas sur-ajuster la sélection aux cutoffs qui servent à la juger :
- sélection sur `novembre` + `etales` (15 cutoffs, 2023-2026) ;
- validation de la short-list sur `recents` (8 derniers jeudis), jamais regardés pendant la sélection.

Score résumé : moyenne, sur les 6 RMSE (2 cibles x 3 arcs), de l'écart relatif au modèle complet. Chaque RMSE pèse
autant, quelle que soit son unité (la règle d'agrégation du jury n'est pas connue).

Usage :
    python -m src.selection_features ablation           # ~25 min, tous les groupes sur les 23 cutoffs
    python -m src.selection_features valider G1 G2 ...  # modèle sans ces groupes vs complet, sur les 23 cutoffs
"""

import argparse
import sys

import pandas as pd

from src.backtest_similaires import JEUX
from src.config import DATASET_MODELE_PATH, OUTPUTS_DIR, TARGETS
from src.evaluate import default_cutoffs, parse_cutoffs, window
from src.models.xgb import XGBModel

METEO_HORAIRE = ["meteo_temperature_2m", "meteo_precipitation", "meteo_rain", "meteo_snowfall", "meteo_cloud_cover",
                 "meteo_wind_speed_10m", "meteo_weather_code"]
GROUPES = {
    "lags_S1_S3": [f"{t}_lag{l}" for t in TARGETS for l in (168, 336, 504)],
    "lag_mean": [f"{t}_lag_mean" for t in TARGETS],
    "niveau28j": [f"{t}_niveau28j" for t in TARGETS],
    "lag52s": [f"{t}_lag52s" for t in TARGETS],
    "autre_cible": "own_lags_only",   # le modèle du débit ne voit pas l'historique de l'occupation, et inversement
    "meteo_horaire": METEO_HORAIRE,
    "meteo_jour": ["meteo_pluie", "meteo_neige_jour", "meteo_neige_veille", "meteo_tmin_jour"],
    "vacances_detail": ["vacances_type", "jours_avant_reprise", "jours_depuis_reprise", "jours_avant_vacances",
                        "vacances_niveau28j"],
    "vacances": ["vacances"],
    "ferie": ["ferie"],
    "autour_ferie": ["veille_ferie", "lendemain_ferie", "pont"],
    "evenements": ["evenement", "evenement_fermeture", "evenement_perturbation"],
    "regimes": ["jo_2024", "post_jo_2024", "post_30kmh"],
    "saison": ["month", "heure_ete"],
    "nb_lags_manquants": ["nb_lags_manquants"],
}
SELECTION = ["novembre", "etales"]
VALIDATION = ["recents"]


def modele(groupes: list[str]) -> XGBModel:
    exclude = [c for g in groupes if GROUPES[g] != "own_lags_only" for c in GROUPES[g]]
    return XGBModel(exclude=exclude, own_lags_only=any(GROUPES[g] == "own_lags_only" for g in groupes))


def cutoffs(data: pd.DataFrame) -> dict[str, list[pd.Timestamp]]:
    return {j: parse_cutoffs(JEUX[j]) if JEUX[j] else default_cutoffs(data, 8) for j in SELECTION + VALIDATION}


def run(data: pd.DataFrame, cuts: dict, groupes: list[str], nom: str) -> pd.DataFrame:
    rows = []
    for jeu, cs in cuts.items():
        for c in cs:
            start, end = window(c)
            pred = modele(groupes).fit(data, c).predict(data, c, start, end)
            truth = data[(data["datetime"] >= start) & (data["datetime"] <= end) & data["is_valid"]]
            m = truth[["arc", "datetime", *TARGETS]].merge(pred, on=["arc", "datetime"], suffixes=("", "_pred"))
            rows.append(m.assign(jeu=jeu, cutoff=c, variante=nom))
    return pd.concat(rows, ignore_index=True)


def resume(p: pd.DataFrame, jeux: list[str]) -> pd.DataFrame:
    """RMSE par variante x (cible, arc) sur les jeux donnés, et écart relatif moyen au modèle complet."""
    p = p[p["jeu"].isin(jeux)]
    out = {}
    for t in TARGETS:
        out[t] = ((p[t] - p[f"{t}_pred"]) ** 2).groupby([p["variante"], p["arc"]]).mean().pow(0.5).unstack("arc")
    r = pd.concat(out, axis=1)
    ecart = r.div(r.loc["complet"]).sub(1).mul(100)
    ecart["score"] = ecart.mean(axis=1)
    ecart["pire"] = ecart.drop(columns="score").max(axis=1)
    return ecart.round(2).sort_values("score")


def victoires(p: pd.DataFrame, jeux: list[str]) -> pd.Series:
    """Part des (cutoff, arc, cible) où la variante fait au moins aussi bien que le modèle complet."""
    p = p[p["jeu"].isin(jeux)]
    e = []
    for t in TARGETS:
        e.append(((p[t] - p[f"{t}_pred"]) ** 2).groupby([p["variante"], p["cutoff"], p["arc"]]).mean()
                 .pow(0.5).rename(t))
    e = pd.concat(e, axis=1).stack().unstack("variante")
    return e.drop(columns="complet").le(e["complet"], axis=0).mean().round(2)


def main(mode: str, retirer: list[str]) -> None:
    data = pd.read_parquet(DATASET_MODELE_PATH)
    cuts = cutoffs(data)
    variantes = {"complet": []}
    if mode == "ablation":
        variantes |= {f"sans_{g}": [g] for g in GROUPES}
    else:
        variantes["short_list"] = retirer
    parts = []
    for nom, gs in variantes.items():
        parts.append(run(data, cuts, gs, nom))
        print(f"{nom} fait", flush=True)
    p = pd.concat(parts, ignore_index=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    p.to_parquet(OUTPUTS_DIR / f"selection_features_{mode}.parquet", index=False)
    pd.set_option("display.width", 250)
    for label, jeux in [("SÉLECTION (novembre + étalés)", SELECTION), ("VALIDATION (8 récents)", VALIDATION)]:
        print(f"\n=== {label} : écart de RMSE au modèle complet, en % (négatif = mieux)")
        r = resume(p, jeux)
        print(r.assign(victoires=victoires(p, jeux)).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["ablation", "valider"])
    parser.add_argument("groupes", nargs="*", help="valider : groupes retirés de la short-list")
    args = parser.parse_args()
    if args.mode == "valider" and (unknown := set(args.groupes) - set(GROUPES)):
        sys.exit(f"groupes inconnus : {unknown}")
    main(args.mode, args.groupes)
