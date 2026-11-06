"""Batterie de tests de nuit, ADAPTATIVE : quand une variante gagne, elle devient la nouvelle référence des tests suivants.

Usage (depuis la racine, venv actif, Mac branché) :
    caffeinate -i python -m exploration.essais_victor.nuit                 # ~7 h maximum
    caffeinate -i python -m exploration.essais_victor.nuit --heures 3      # durée maximale différente

Protocole (identique pour toutes les variantes) :
    - 31 jeudis en 3 groupes (règle de Robin) : réguliers 2022-2024 (1 toutes les 8 semaines), 8 derniers jeudis
      (sept.-oct. 2026), semaines du 11 novembre (2023-2025). Entraînement : tout le passé jusqu'au jeudi 23 h.
    - Exploration rapide : 300 arbres, learning_rate 0,1 ; 3 graines moyennées (le bruit d'une graine est de 3-5 %).
    - Note : RMSE sur 24 h, par jeudi, arc et cible.
Règle d'adoption (une seule condition ratée suffit à refuser) :
    - gain moyen >= 1 % (moyenne, sur les jeudis, du ratio RMSE variante / RMSE référence) ;
    - au moins 65 % des jeudis gagnés (jeudis à égalité exclus) ;
    - aucun groupe dégradé de plus de 2 % ; aucun couple (arc, cible) dégradé de plus de 5 %.
Parmi plusieurs variantes d'un même test (ex. 3 fonctions de perte), on adopte la meilleure qui passe la règle.
Un test qui échoue ou dépasse son temps est noté et sauté : il n'est jamais relancé.
À la fin : confirmation avec le réglage complet (600 arbres), puis recherche d'hyperparamètres s'il reste du temps.

Sorties (ignorées par git) : outputs/experiments/nuit/
    rapport.md (mis à jour après chaque test), journal.log, resultats.json, un CSV de RMSE par variante.
"""

import argparse
import json
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from exploration.essais_victor.anomalies import EPISODES_CONNUS, derives_occupation, jours_anormaux, masque
from src.config import EVENEMENTS_DIR, OUTPUTS_DIR, ROOT, TARGETS, TZ
from exploration.essais_victor.essai_xgb import DATASET, PARAMS, jeudis, prepare, rmse
from src.evaluate import parse_cutoffs, window
from src.models.preprocessing import feature_columns

SORTIE = OUTPUTS_DIR / "experiments" / "nuit"
GRAINES = [0, 1, 2]
PARAMS_EXPLO = {**PARAMS, "n_estimators": 300, "learning_rate": 0.1, "tree_method": "hist", "n_jobs": -1}
PARAMS_COMPLET = {**PARAMS, "tree_method": "hist", "n_jobs": -1}
GROUPES = {
    "reguliers": jeudis("2022-01-06", "2024-12-26", pas_jours=56),
    "recents": jeudis("2026-09-03", "2026-10-22"),
    "novembre": ["2023-11-09", "2024-11-07", "2025-11-06"],
}
ARCS_CIBLES = ["Champs-Elysées", "Convention", "Saint-Pères"]
OCC = "taux_occupation"
EVENT_COLS = ["evenement", "evenement_fermeture", "evenement_perturbation"]
FRAIS_COLS = ["horizon"] + [f"frais_{k}_{c}" for c in TARGETS for k in ("val", "ecart")]
DELTA_HUBER = {"debit_horaire": {"petit": 30, "grand": 100}, OCC: {"petit": 1, "grand": 3}}
# Arcs des Champs-Elysées dans le même sens que 4264 (vers la Concorde), hors 4264 (option de Robin)
CHAMPS_EXTRA = ["4403", "4397", "4269", "4267", "4274", "4276", "4281", "4284", "270"]
FICHIER_EVENEMENTS_V2 = "evenements_propositions_v2.csv"

T0 = time.time()


def log(msg):
    ligne = f"[{time.strftime('%H:%M:%S')} | {(time.time() - T0) / 3600:4.1f} h] {msg}"
    print(ligne, flush=True)
    with open(SORTIE / "journal.log", "a") as f:
        f.write(ligne + "\n")


class TempsDepasse(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------------
# Features calculées à la demande
# ---------------------------------------------------------------------------------------------------------------

def features_fraiches(d: pd.DataFrame) -> pd.DataFrame:
    """Dernier jour connu selon l'horizon. Le rendu prédit samedi (h=1) -> mercredi (h=5) avec des données
    arrêtées le jeudi : le dernier jour connu est la date - (h+1) jours, toujours <= jeudi (aucune fuite).
    Les jeudis et vendredis, jamais prédits, reçoivent un horizon tiré au hasard (graine fixe)."""
    loc = d["datetime"].dt.tz_convert(TZ)
    date = loc.dt.tz_localize(None).dt.normalize()
    heure = loc.dt.hour
    h = loc.dt.dayofweek.map({5: 1, 6: 2, 0: 3, 1: 4, 2: 5})
    na = h.isna().to_numpy()
    h = h.to_numpy(dtype=float, copy=True)
    h[na] = np.random.default_rng(0).integers(1, 6, na.sum())
    connu = date - pd.to_timedelta(h + 1, unit="D")
    out = pd.DataFrame({"horizon": h}, index=d.index)
    valide = d["is_valid"].astype(bool)
    for c in TARGETS:
        val = d[c].where(valide)
        horaire = val.groupby([d["arc"], date, heure]).mean()
        out[f"frais_val_{c}"] = horaire.reindex(pd.MultiIndex.from_arrays([d["arc"], connu, heure])).to_numpy()
        jour = val.groupby([d["arc"], date]).agg(["mean", "count"])
        moy = jour["mean"].where(jour["count"] >= 18)
        moy_df = moy.unstack(0)                                     # dates x arcs
        moy_df = moy_df.reindex(pd.date_range(moy_df.index.min(), moy_df.index.max(), freq="D"))
        ref = pd.concat([moy_df.shift(k) for k in (7, 14, 21)]).groupby(level=0).mean()
        ecart = (moy_df / ref - 1).stack()
        ecart.index = ecart.index.swaplevel()
        out[f"frais_ecart_{c}"] = ecart.reindex(pd.MultiIndex.from_arrays([d["arc"], connu])).to_numpy()
    return out


def colonnes_evenements(d: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Recalcule evenement / _fermeture / _perturbation avec une autre table d'événements."""
    import src.data.build_dataset as bd
    original = bd.evenements.load
    bd.evenements.load = lambda name="evenements": table
    try:
        return bd.event_flags(d[["datetime", "arc"]]).astype(int)
    finally:
        bd.evenements.load = original


def lignes_champs(base: pd.DataFrame) -> pd.DataFrame:
    """Lignes d'entraînement pour les autres arcs des Champs : cibles et lags propres à l'arc, le reste (calendrier,
    météo, événements…) copié des lignes de l'arc 4264 à la même heure."""
    from src.data.build_dataset import arc_series
    from src.data.sources import trafic
    from src.features.build import lag_features, level_features, yearly_lag_features
    raw = trafic.load(CHAMPS_EXTRA)
    hours = pd.date_range(base["datetime"].min() - pd.Timedelta(days=371), base["datetime"].max(), freq="h", tz="UTC")
    frames = []
    for aid in CHAMPS_EXTRA:
        s = arc_series(raw, aid, hours)
        s.loc[~s["valide"], TARGETS] = np.nan
        frames.append(s.drop(columns="etat_arc").rename_axis("datetime").reset_index().assign(arc=f"Champs_{aid}"))
    t = pd.concat(frames, ignore_index=True).rename(columns={"valide": "is_valid"})
    t = t.sort_values(["arc", "datetime"]).reset_index(drop=True)
    t = pd.concat([t, lag_features(t), level_features(t), yearly_lag_features(t)], axis=1)
    propres = set(t.columns) | {"source_cible", "partie"} | set(FRAIS_COLS)   # fraîches : recalculées par arc
    partage = base[base["arc"] == "Champs-Elysées"]
    partage = partage[["datetime"] + [c for c in partage.columns if c not in propres]]
    e = t.merge(partage, on="datetime", how="inner")
    e = e[e["is_valid"].astype(bool)].assign(source_cible="mesure", partie="historique")
    return prepare(e).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------------
# Banc d'essai : données, entraînement, évaluation
# ---------------------------------------------------------------------------------------------------------------

class Banc:
    def __init__(self):
        self.df = prepare(pd.read_parquet(DATASET))
        self.base_cols = feature_columns(self.df)
        self.extra = None
        self.cutoffs = [(g, c) for g, cs in GROUPES.items() for c in parse_cutoffs(cs)]
        self.cache = {}
        self.split_idx = {}
        for _, c in self.cutoffs:
            start, end = window(c)
            valide = self.df["is_valid"].astype(bool)
            self.split_idx[c] = (np.flatnonzero((self.df["datetime"] <= c) & valide),
                                 np.flatnonzero((self.df["datetime"] >= start) & (self.df["datetime"] <= end) & valide))

    # --- préparations paresseuses (une erreur ici fait échouer le seul test qui en a besoin) ---
    def assurer_frais(self):
        if "horizon" not in self.df:
            log("  calcul des features fraîches…")
            self.df = pd.concat([self.df, features_fraiches(self.df)], axis=1)
        if self.extra is not None and "horizon" not in self.extra:
            self.extra = pd.concat([self.extra, features_fraiches(self.extra)], axis=1)

    def assurer_extra(self):
        if self.extra is None:
            log("  construction des lignes des autres arcs des Champs…")
            self.extra = lignes_champs(self.df)
            log(f"  {len(self.extra)} lignes ajoutées ({self.extra['arc'].nunique()} arcs)")
            if "horizon" in self.df:
                self.assurer_frais()

    def assurer_evenements(self, suffixe):
        if f"evenement_{suffixe}" in self.df:
            return
        import src.data.sources.evenements as ev
        if suffixe == "sg":
            table = ev.load("evenements")
            table = table[~table["nom"].str.contains("grève", case=False)]
        else:
            chemins = [EVENEMENTS_DIR / FICHIER_EVENEMENTS_V2, ROOT / FICHIER_EVENEMENTS_V2,
                       ROOT / "docs" / FICHIER_EVENEMENTS_V2]
            chemin = next((p for p in chemins if p.exists()), None)
            if chemin is None:
                raise FileNotFoundError(f"{FICHIER_EVENEMENTS_V2} introuvable (cherché dans data/evenements/, la racine, docs/)")
            table = ev.expand_manual(pd.read_csv(chemin, sep=";"), 2027)
            if suffixe == "v2sg":
                table = table[~table["nom"].str.contains("grève", case=False)]
        cols = colonnes_evenements(self.df, table)
        for c in EVENT_COLS:
            self.df[f"{c}_{suffixe}"] = cols[c].to_numpy()
        if self.extra is not None:                      # mêmes événements que l'arc 4264 à la même heure
            champs = self.df[self.df["arc"] == "Champs-Elysées"].set_index("datetime")
            for c in EVENT_COLS:
                self.extra[f"{c}_{suffixe}"] = champs[f"{c}_{suffixe}"].reindex(self.extra["datetime"]).to_numpy()

    def preparer(self, cfg):
        if cfg.get("lag_frais"):
            self.assurer_frais()
        if cfg.get("arcs_champs"):
            self.assurer_extra()
        if cfg.get("evenements"):
            self.assurer_evenements(cfg["evenements"])

    # --- données d'un jeudi ---
    def split(self, cfg, cutoff):
        i_tr, i_te = self.split_idx[cutoff]
        tr, te = self.df.iloc[i_tr], self.df.iloc[i_te]
        if cfg.get("arcs_champs"):
            tr = pd.concat([tr, self.extra[self.extra["datetime"] <= cutoff]], ignore_index=True)
        return tr, te

    def memo(self, cle, fn):
        if cle not in self.cache:
            self.cache[cle] = fn()
        return self.cache[cle]

    def masque_train(self, cfg, tr, cutoff, cible):
        m = np.ones(len(tr), bool)
        ext = bool(cfg.get("arcs_champs"))
        if cfg.get("anomalies"):
            k = cfg["anomalies"]
            def table():
                if k == "connus":
                    return pd.concat([pd.DataFrame({"arc": a, "date": pd.date_range(d, f)})
                                      for a, d, f, _ in EPISODES_CONNUS])
                return jours_anormaux(self.df[self.df["datetime"] <= cutoff], int(k) / 100)
            a = self.memo(("anom", k, cutoff), table)
            m &= ~self.memo(("masque_anom", k, cutoff, ext), lambda: masque(tr, a))
        if cfg.get("derive_occ") and cible == OCC:
            a = self.memo(("derive", cutoff), lambda: derives_occupation(self.df[self.df["datetime"] <= cutoff]))
            m &= ~self.memo(("masque_derive", cutoff, ext), lambda: masque(tr, a))
        if cfg.get("debut"):
            k, t = cfg["debut"], tr["datetime"]
            if k == "2023":
                if cutoff >= pd.Timestamp("2024-01-01", tz="UTC"):
                    m &= (t >= pd.Timestamp("2023-01-01", tz="UTC")).to_numpy()
            else:
                m &= (t > cutoff - pd.DateOffset(months=int(k.rstrip("m")))).to_numpy()
        return m

    def colonnes(self, cfg):
        drop = tuple(cfg.get("drop", []))
        cols = [c for c in self.base_cols if not (drop and c.startswith(drop))]
        return cols + (FRAIS_COLS if cfg.get("lag_frais") else [])

    def matrice(self, cfg, d):
        X = d[self.colonnes(cfg)].astype(float).reset_index(drop=True)
        if cfg.get("evenements"):
            for c in EVENT_COLS:
                if c in X:
                    X[c] = d[f"{c}_{cfg['evenements']}"].to_numpy(dtype=float)
        arcs = ARCS_CIBLES + ([f"Champs_{a}" for a in CHAMPS_EXTRA] if cfg.get("arcs_champs") else [])
        for a in arcs:
            X[f"arc_{a}"] = (d["arc"].to_numpy() == a).astype(float)
        return X

    @staticmethod
    def base_lag(tr, d, cible):
        """lag_mean (S-1..S-3), à défaut la médiane de l'entraînement par (arc, jour, heure). Sert aussi de baseline."""
        prof = tr.groupby(["arc", "dow", "hour"])[cible].median()
        repli = prof.reindex(pd.MultiIndex.from_frame(d[["arc", "dow", "hour"]])).to_numpy()
        return d[f"{cible}_lag_mean"].fillna(pd.Series(repli, index=d.index)).to_numpy(dtype=float)

    def predire(self, cfg, cutoff, cible, graine, params, entree_debit=None, te=None):
        tr, te0 = self.split(cfg, cutoff)
        te = te0 if te is None else te
        tr = tr[self.masque_train(cfg, tr, cutoff, cible)]
        X_tr, X_te = self.matrice(cfg, tr), self.matrice(cfg, te)
        if cfg.get("profil"):
            cle = ["arc", "dow", "hour", "vacances"]
            prof = tr.groupby(cle)[TARGETS].median()
            for c in TARGETS:
                X_tr[f"profil_{c}"] = prof[c].reindex(pd.MultiIndex.from_frame(tr[cle])).to_numpy()
                X_te[f"profil_{c}"] = prof[c].reindex(pd.MultiIndex.from_frame(te[cle])).to_numpy()
        if cfg.get("chaine") and cible == OCC:
            X_tr["debit_pred"] = tr["debit_horaire"].to_numpy(dtype=float)     # vrai débit à l'entraînement
            X_te["debit_pred"] = entree_debit                                  # débit prédit au test
        y = tr[cible].to_numpy(dtype=float)
        if cfg.get("cible") == "diff":
            b_tr, b_te = self.base_lag(tr, tr, cible), self.base_lag(tr, te, cible)
            y = y - b_tr
        elif cfg.get("cible") == "log":
            y = np.log1p(y)

        p = {**params, **cfg.get("params", {}), "random_state": graine}
        if cfg.get("perte"):
            p["base_score"] = float(np.median(y))
            if cfg["perte"] == "absolue":
                p["objective"] = "reg:absoluteerror"
            else:
                p["objective"] = "reg:pseudohubererror"
                p["huber_slope"] = DELTA_HUBER[cible][cfg["perte"].split("_")[1]]
        if cfg.get("monotone"):
            suivis = {f"{cible}_{s}" for s in ("lag168", "lag336", "lag504", "lag_mean", "niveau28j", "lag52s")}
            suivis.add(f"frais_val_{cible}")
            p["monotone_constraints"] = "(" + ",".join("1" if c in suivis else "0" for c in X_tr.columns) + ")"
        if cfg.get("arret_precoce"):
            limite = tr["datetime"].max() - pd.Timedelta(days=28)
            ajust = (tr["datetime"] <= limite).to_numpy()
            m1 = XGBRegressor(**{**p, "n_estimators": 1500, "early_stopping_rounds": 50})
            m1.fit(X_tr[ajust], y[ajust], eval_set=[(X_tr[~ajust], y[~ajust])], verbose=False)
            p["n_estimators"] = int(m1.best_iteration) + 1

        y_hat = XGBRegressor(**p).fit(X_tr, y).predict(X_te)
        if cfg.get("cible") == "diff":
            y_hat = y_hat + b_te
        elif cfg.get("cible") == "log":
            y_hat = np.expm1(y_hat)
        return np.clip(y_hat, 0, 100 if cible == OCC else None)

    def evaluer(self, cfg, nom, cibles=TARGETS, ref=None, params=PARAMS_EXPLO, graines=GRAINES,
                fin=None, groupes=None):
        """RMSE par (jeudi, arc, cible) et prédictions. Les cibles hors `cibles` reprennent celles de `ref`."""
        self.preparer(cfg)
        t = time.time()
        rows, preds = [], {}
        cutoffs = [(g, c) for g, c in self.cutoffs if groupes is None or g in groupes]
        for i, (groupe, cutoff) in enumerate(cutoffs, 1):
            tr, te = self.split(cfg, cutoff)
            if te.empty:
                continue
            p_jeudi = {}
            for cible in TARGETS:
                if cible not in cibles:
                    p_jeudi[cible] = ref["preds"][(cutoff, cible)]["y_hat"].to_numpy()
                    continue
                entree = p_jeudi.get("debit_horaire")
                p_jeudi[cible] = np.mean([self.predire(cfg, cutoff, cible, g, params, entree) for g in graines], axis=0)
            for cible in TARGETS:
                naif = self.base_lag(tr, te, cible)
                y_hat = p_jeudi[cible]
                alpha = (cfg.get("melange") or {}).get(cible)
                if alpha is not None and cible in cibles:
                    a = te["arc"].map(alpha).fillna(0).to_numpy()
                    y_hat = (1 - a) * y_hat + a * naif
                preds[(cutoff, cible)] = pd.DataFrame({"arc": te["arc"].to_numpy(), "datetime": te["datetime"].to_numpy(),
                                                       "y": te[cible].to_numpy(dtype=float), "y_hat": y_hat, "naif": naif})
            rows += lignes_rmse(preds, cutoff, groupe)
            if fin is not None and time.time() > fin:
                raise TempsDepasse(f"{nom} : temps maximal atteint après {i}/{len(cutoffs)} jeudis")
            if i % 8 == 0:
                log(f"    {nom} : {i}/{len(cutoffs)} jeudis ({(time.time() - t) / 60:.1f} min)")
        res = pd.DataFrame(rows)
        res.to_csv(SORTIE / f"rmse_{nom}.csv", index=False)
        log(f"    {nom} terminé en {(time.time() - t) / 60:.1f} min")
        return {"rmse": res, "preds": preds}


def lignes_rmse(preds, cutoff, groupe, exclure=None):
    rows = []
    for (c, cible), p in preds.items():
        if c != cutoff:
            continue
        for arc, g in p.groupby("arc"):
            if exclure is not None:
                g = g[~exclure(g)]
            if len(g):
                rows.append({"groupe": groupe, "cutoff": cutoff, "arc": arc, "cible": cible,
                             "rmse": rmse(g["y"].to_numpy(), g["y_hat"].to_numpy())})
    return rows


# ---------------------------------------------------------------------------------------------------------------
# Règle d'adoption
# ---------------------------------------------------------------------------------------------------------------

def comparer(ref, var, cibles=TARGETS, groupes=None):
    m = ref["rmse"].merge(var["rmse"], on=["groupe", "cutoff", "arc", "cible"], suffixes=("_ref", "_var"))
    m = m[m["cible"].isin(cibles)]
    if groupes:
        m = m[m["groupe"].isin(groupes)]
    m["ratio"] = m["rmse_var"] / m["rmse_ref"]
    jeudi = m.groupby(["groupe", "cutoff"])["ratio"].mean()
    decisifs = jeudi[(jeudi - 1).abs() > 5e-4]
    gain = (1 - jeudi.mean()) * 100
    part = (decisifs < 1).mean() if len(decisifs) else 0.0
    par_groupe = ((1 - jeudi.groupby(level=0).mean()) * 100).round(1).to_dict()
    par_couple = ((1 - m.groupby(["arc", "cible"])["ratio"].mean()) * 100).round(1)
    pire = par_couple.idxmin()
    ok = (gain >= 1 and part >= 0.65 and min(par_groupe.values()) >= -2 and par_couple.min() >= -5)
    return {"gain_%": round(gain, 2), "jeudis_gagnes": f"{part:.0%}", "par_groupe": par_groupe,
            "pire_couple": f"{pire[0]} / {pire[1]} : {par_couple.min():+.1f} %",
            "par_couple": {f"{a} / {c}": v for (a, c), v in par_couple.items()}, "adopte": bool(ok)}


# ---------------------------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------------------------

def fusion(cfg, ajout):
    out = {**cfg, **ajout}
    if "drop" in ajout:
        out["drop"] = sorted(set(cfg.get("drop", [])) | set(ajout["drop"]))
    return out


def test_variantes(banc, etat, exp, fin):
    """Teste les variantes contre la référence courante ; adopte la meilleure qui passe la règle."""
    cibles = exp.get("cibles", TARGETS)
    resultats = {}
    for nom_v, ajout in exp["variantes"].items():
        cfg_v = fusion(etat["cfg"], ajout)
        log(f"  variante {nom_v} : {json.dumps(ajout, ensure_ascii=False)}")
        try:
            var = banc.evaluer(cfg_v, f"{exp['nom']}__{nom_v}", cibles=cibles, ref=etat["ref"], fin=fin)
        except TempsDepasse as e:                      # on garde les variantes déjà terminées
            log(f"  variante {nom_v} INTERROMPUE : {e}")
            resultats[nom_v] = {"statut": f"interrompue : {e}"}
            break
        except Exception as e:
            log(f"  variante {nom_v} ÉCHOUÉE : {e!r}")
            resultats[nom_v] = {"statut": f"échouée : {e!r}"[:300]}
            continue
        comp = comparer(etat["ref"], var, cibles=cibles, groupes=exp.get("groupes_decision"))
        log(f"  -> {nom_v} : gain {comp['gain_%']:+.2f} %, jeudis gagnés {comp['jeudis_gagnes']}, "
            f"groupes {comp['par_groupe']}, pire {comp['pire_couple']} => {'ADOPTABLE' if comp['adopte'] else 'refusé'}")
        resultats[nom_v] = {**comp, "_var": var, "_cfg": cfg_v}
    candidats = {k: v for k, v in resultats.items() if v.get("adopte")}
    adopte = max(candidats, key=lambda k: candidats[k]["gain_%"]) if candidats else None
    if adopte:
        etat["cfg"] = resultats[adopte]["_cfg"]
        etat["ref"] = resultats[adopte]["_var"]
        log(f"  ADOPTÉ : {adopte} -> nouvelle référence {json.dumps(etat['cfg'], ensure_ascii=False)}")
    return {"adopte": adopte, "variantes": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                                            for k, v in resultats.items()}}


def test_melange(banc, etat, exp, fin):
    """Mélange XGBoost / baseline, poids choisi par (arc, cible) sur les jeudis réguliers, jugé sur les autres."""
    ref = etat["ref"]
    alphas, preds = {c: {} for c in TARGETS}, {}
    grille = np.round(np.arange(0, 0.55, 0.05), 2)
    regs = {c for g, c in banc.cutoffs if g == "reguliers"}
    for cible in TARGETS:
        tout = pd.concat([p for (c, ci), p in ref["preds"].items() if ci == cible and c in regs])
        for arc, g in tout.groupby("arc"):
            erreurs = [rmse(g["y"], (1 - a) * g["y_hat"] + a * g["naif"]) for a in grille]
            alphas[cible][arc] = float(grille[int(np.argmin(erreurs))])
    log(f"  poids de la baseline choisis sur les réguliers : {alphas}")
    rows = []
    for (cutoff, cible), p in ref["preds"].items():
        a = p["arc"].map(alphas[cible]).fillna(0).to_numpy()
        preds[(cutoff, cible)] = p.assign(y_hat=(1 - a) * p["y_hat"] + a * p["naif"])
    for g, c in banc.cutoffs:
        rows += lignes_rmse(preds, c, g)
    var = {"rmse": pd.DataFrame(rows), "preds": preds}
    comp = comparer(ref, var, groupes=["recents", "novembre"])
    log(f"  -> mélange : gain {comp['gain_%']:+.2f} %, jeudis gagnés {comp['jeudis_gagnes']} => "
        f"{'ADOPTÉ' if comp['adopte'] else 'refusé'}")
    if comp["adopte"]:
        etat["cfg"] = fusion(etat["cfg"], {"melange": alphas})
        etat["ref"] = var
    return {"adopte": "melange" if comp["adopte"] else None, "alphas": alphas, "variantes": {"melange": comp}}


def test_manif(banc, etat, exp, fin):
    """Samedis 17 et 24/10/2026 : modèle seul, modèle avec un événement « perturbation », modèle x facteurs."""
    facteurs = {"Champs-Elysées": 0.6, "Saint-Pères": 0.55, "Convention": 0.75}
    rows, constats = [], []
    for cutoff in parse_cutoffs(["2026-10-15", "2026-10-22"]):
        samedi = (cutoff.tz_convert(TZ).tz_localize(None).normalize() + pd.Timedelta(days=2)).date()
        tr, te = banc.split(etat["cfg"], cutoff)
        jour = te["datetime"].dt.tz_convert(TZ).dt.date == samedi
        heure = te["datetime"].dt.tz_convert(TZ).dt.hour
        # le samedi était-il vraiment perturbé ? débit du jour / moyenne des 3 samedis précédents
        for arc in ARCS_CIBLES:
            d = banc.df[(banc.df["arc"] == arc) & banc.df["is_valid"].astype(bool)]
            dj = d["datetime"].dt.tz_convert(TZ).dt.date
            avant = [samedi - pd.Timedelta(days=7 * k) for k in (1, 2, 3)]
            ratio = d.loc[dj == samedi, "debit_horaire"].mean() / d.loc[dj.isin(avant), "debit_horaire"].mean()
            constats.append(f"{samedi} {arc} : débit du samedi = {ratio:.2f} x moyenne des 3 samedis précédents")
        te_flag = te.copy()
        for c in EVENT_COLS[::2]:                         # evenement, evenement_perturbation (+ versions adoptées)
            for col in [c] + [f"{c}_{s}" for s in ("sg", "v2", "v2sg") if f"{c}_{s}" in te_flag]:
                te_flag.loc[jour, col] = 1
        options = {}
        for cible in TARGETS:
            seul = etat["ref"]["preds"][(cutoff, cible)]["y_hat"].to_numpy()
            entree = options.get(("flag", "debit_horaire"))
            flag = np.mean([banc.predire(etat["cfg"], cutoff, cible, g, PARAMS_EXPLO, entree, te=te_flag)
                            for g in GRAINES], axis=0)
            options[("flag", cible)] = flag
            f = te["arc"].map(facteurs).to_numpy() if cible == "debit_horaire" else np.ones(len(te))
            options_c = {"modele_seul": seul, "evenement_perturbation": flag,
                         "facteurs_journee": np.where(jour, seul * f, seul),
                         "facteurs_12h_20h": np.where(jour & heure.between(12, 20), seul * f, seul)}
            for nom, y_hat in options_c.items():
                for arc in ARCS_CIBLES:
                    m = (jour & (te["arc"] == arc)).to_numpy()
                    if m.any():
                        rows.append({"samedi": str(samedi), "option": nom, "arc": arc, "cible": cible,
                                     "rmse": rmse(te[cible].to_numpy()[m], y_hat[m])})
    for c in constats:
        log("  " + c)
    t = pd.DataFrame(rows).groupby(["cible", "arc", "option"])["rmse"].mean().unstack().round(2)
    log("  RMSE moyen des 2 samedis :\n" + t.to_string())
    t.to_csv(SORTIE / "samedi_manif.csv")
    return {"adopte": None, "constats": constats, "tableau": t.reset_index().to_dict(orient="records"),
            "note": "test indicatif (2 samedis) : sert à choisir l'option du samedi 7/11, rien n'est adopté"}


def test_confirmation(banc, etat, exp, fin):
    """Réglage complet (600 arbres) : référence d'origine contre configuration finale, avec et sans jours anormaux."""
    if etat["cfg"] == {}:
        return {"adopte": None, "note": "aucune variante adoptée : rien à confirmer"}
    cfg_final = {k: v for k, v in etat["cfg"].items() if k != "melange"}
    ini = banc.evaluer({}, "confirmation__origine", params=PARAMS_COMPLET, fin=fin)
    fin_ = banc.evaluer(cfg_final, "confirmation__final", params=PARAMS_COMPLET, fin=fin)
    if etat["cfg"].get("melange"):                     # le mélange est un post-traitement : on le réapplique
        for (c, ci), p in fin_["preds"].items():
            a = p["arc"].map(etat["cfg"]["melange"][ci]).fillna(0).to_numpy()
            fin_["preds"][(c, ci)] = p.assign(y_hat=(1 - a) * p["y_hat"] + a * p["naif"])
        fin_["rmse"] = pd.DataFrame([r for g, c in banc.cutoffs for r in lignes_rmse(fin_["preds"], c, g)])
    comp = comparer(ini, fin_)
    log(f"  CONFIRMATION (600 arbres) : gain {comp['gain_%']:+.2f} %, jeudis gagnés {comp['jeudis_gagnes']}, "
        f"groupes {comp['par_groupe']}, pire {comp['pire_couple']}")
    # Robin : erreur avec / sans les jours anormaux du test (on ne pourra pas les retirer au rendu)
    anormaux = jours_anormaux(banc.df, 0.30)
    cles = set(anormaux["arc"] + "|" + anormaux["date"].dt.strftime("%Y-%m-%d"))
    def exclure(g):
        return (g["arc"] + "|" + pd.to_datetime(g["datetime"], utc=True).dt.tz_convert(TZ).dt.strftime("%Y-%m-%d")).isin(cles)
    sans = {}
    for nom, r in [("origine", ini), ("final", fin_)]:
        rows = [x for g, c in banc.cutoffs for x in lignes_rmse(r["preds"], c, g, exclure=exclure)]
        sans[nom] = pd.DataFrame(rows).groupby(["arc", "cible"])["rmse"].mean().round(2)
    avec = {n: r["rmse"].groupby(["arc", "cible"])["rmse"].mean().round(2) for n, r in [("origine", ini), ("final", fin_)]}
    t = pd.DataFrame({"origine_tous_jours": avec["origine"], "final_tous_jours": avec["final"],
                      "origine_sans_anormaux": sans["origine"], "final_sans_anormaux": sans["final"]})
    log("  RMSE moyen (600 arbres), avec / sans jours anormaux dans le test :\n" + t.to_string())
    t.to_csv(SORTIE / "confirmation.csv")
    return {"adopte": None, "comparaison": comp, "tableau": t.reset_index().to_dict(orient="records")}


def test_recherche(banc, etat, exp, fin):
    reste_min = (fin - time.time()) / 60
    if reste_min < 60:
        return {"adopte": None, "note": f"pas lancée : {reste_min:.0f} min restantes (< 60)"}
    n_iter = max(5, int((reste_min - 20) / 6))
    log(f"  recherche_xgb --mode nuit --n-iter {n_iter} (limite {reste_min - 5:.0f} min)")
    with open(SORTIE / "recherche_hyperparametres.log", "w") as f:
        try:
            r = subprocess.run([sys.executable, "-m", "exploration.essais_victor.recherche_xgb", "--mode", "nuit", "--n-iter", str(n_iter)],
                               stdout=f, stderr=subprocess.STDOUT, timeout=(reste_min - 5) * 60, cwd=ROOT)
            statut = "terminée" if r.returncode == 0 else f"code {r.returncode}"
        except subprocess.TimeoutExpired:
            statut = "interrompue (temps)"
    return {"adopte": None, "note": f"recherche {statut} ; résultats dans outputs/experiments/recherche_nuit/ "
                                    "(features de base, pas la configuration adoptée)"}


EXPERIENCES = [
    {"nom": "evenements_v2", "titre": "Nouveaux événements (fichier v2), avec et sans grèves",
     "variantes": {"v2": {"evenements": "v2"}, "v2_sans_greves": {"evenements": "v2sg"}}},
    {"nom": "sans_greves", "titre": "Retirer les grèves 2023 des événements",
     "variantes": {"sans_greves": {"evenements": "sg"}}},
    {"nom": "samedi_manif", "titre": "Samedi de manifestation : 3 options sur les 17 et 24/10/2026",
     "fn": test_manif, "minutes": 15},
    {"nom": "lags_frais", "titre": "Données les plus fraîches selon le jour prédit",
     "variantes": {"frais": {"lag_frais": True}}},
    {"nom": "jours_anormaux", "titre": "Retirer de l'entraînement les jours anormaux (travaux, déviations)",
     "variantes": {"episodes_connus": {"anomalies": "connus"}, "retrait_30%": {"anomalies": "30"},
                   "retrait_20%": {"anomalies": "20"}}, "minutes": 75},
    {"nom": "derive_occupation", "titre": "Retirer les dérives du capteur d'occupation (occupation seule)",
     "variantes": {"derive_occ": {"derive_occ": True}}, "cibles": [OCC]},
    {"nom": "date_debut", "titre": "Date de début de l'entraînement",
     "variantes": {"depuis_2023": {"debut": "2023"}, "24_mois": {"debut": "24m"}, "18_mois": {"debut": "18m"},
                   "12_mois": {"debut": "12m"}}, "minutes": 75},
    {"nom": "profil_long_terme", "titre": "Profil médian arc x jour x heure x vacances en colonne",
     "variantes": {"profil": {"profil": True}}},
    {"nom": "sans_meteo", "titre": "Retirer la météo", "variantes": {"sans_meteo": {"drop": ["meteo_"]}}},
    {"nom": "occupation_depuis_debit", "titre": "Prédire l'occupation à partir du débit prédit",
     "variantes": {"chaine": {"chaine": True}}, "cibles": [OCC]},
    {"nom": "transformation_cible", "titre": "Transformer la cible : écart à lag_mean, logarithme",
     "variantes": {"ecart_lag_mean": {"cible": "diff"}, "log": {"cible": "log"}}},
    {"nom": "fonction_perte", "titre": "Fonction de perte : Huber, erreur absolue",
     "variantes": {"huber_petit": {"perte": "huber_petit"}, "huber_grand": {"perte": "huber_grand"},
                   "absolue": {"perte": "absolue"}}, "minutes": 75},
    {"nom": "contraintes_monotones", "titre": "Contraintes de sens sur les lags",
     "variantes": {"monotone": {"monotone": True}}},
    {"nom": "arret_precoce", "titre": "Arrêt précoce (4 dernières semaines en validation)",
     "variantes": {"arret_precoce": {"arret_precoce": True}}},
    {"nom": "melange_baseline", "titre": "Mélange XGBoost / baseline, poids par arc et cible", "fn": test_melange,
     "minutes": 5},
    {"nom": "arcs_champs", "titre": "Ajouter les lignes des autres arcs des Champs (option de Robin)",
     "variantes": {"arcs_champs": {"arcs_champs": True}}, "minutes": 90},
    {"nom": "confirmation_finale", "titre": "Confirmation avec le réglage complet (600 arbres)",
     "fn": test_confirmation, "minutes": 150},
    {"nom": "recherche_hyperparametres", "titre": "Recherche d'hyperparamètres (temps restant)", "fn": test_recherche,
     "minutes": 600},
]


# ---------------------------------------------------------------------------------------------------------------
# Rapport et boucle principale
# ---------------------------------------------------------------------------------------------------------------

def ecrire_rapport(etat):
    L = ["# Tests de nuit", "",
         f"Démarré le {etat['debut']}. Dernière mise à jour : {time.strftime('%Y-%m-%d %H:%M')}.", "",
         "Règle d'adoption : gain moyen ≥ 1 %, ≥ 65 % des jeudis gagnés, aucun groupe < −2 %, aucun couple "
         "(arc, cible) < −5 %. Exploration : 300 arbres, 3 graines, 31 jeudis (réguliers 2022-2024, sept.-oct. 2026, "
         "semaines du 11/11).", "",
         f"**Configuration retenue à ce stade** : `{json.dumps(etat['cfg'], ensure_ascii=False)}`", "",
         "| Test | Statut | Adopté |", "|---|---|---|"]
    for h in etat["historique"]:
        L.append(f"| {h['titre']} | {h['statut']} | {h.get('adopte') or '—'} |")
    for h in etat["historique"]:
        L += ["", f"## {h['titre']}", "", f"Statut : {h['statut']} ({h.get('duree_min', 0):.0f} min)."]
        if h.get("note"):
            L.append(f"\n{h['note']}")
        for nom_v, v in (h.get("variantes") or {}).items():
            if "gain_%" in v:
                L.append(f"- **{nom_v}** : gain {v['gain_%']:+.2f} %, jeudis gagnés {v['jeudis_gagnes']}, "
                         f"par groupe {v['par_groupe']}, pire couple {v['pire_couple']} → "
                         f"{'adoptable' if v['adopte'] else 'refusé'}")
            else:
                L.append(f"- **{nom_v}** : {v.get('statut')}")
        if h.get("comparaison"):
            c = h["comparaison"]
            L.append(f"- Configuration finale contre origine : gain {c['gain_%']:+.2f} %, jeudis gagnés "
                     f"{c['jeudis_gagnes']}, par groupe {c['par_groupe']}, pire couple {c['pire_couple']}")
        for cle in ("constats",):
            for x in h.get(cle, []):
                L.append(f"- {x}")
        if h.get("tableau"):
            L += ["", "```", pd.DataFrame(h["tableau"]).to_string(index=False), "```"]
        if h.get("erreur"):
            L += ["", "```", h["erreur"][-1500:], "```"]
    (SORTIE / "rapport.md").write_text("\n".join(L) + "\n")
    propre = [{k: v for k, v in h.items()} for h in etat["historique"]]
    (SORTIE / "resultats.json").write_text(json.dumps({"cfg": etat["cfg"], "tests": propre}, indent=2,
                                                      ensure_ascii=False, default=str))


def main(heures):
    SORTIE.mkdir(parents=True, exist_ok=True)
    fin_globale = T0 + heures * 3600 - 300
    etat = {"debut": time.strftime("%Y-%m-%d %H:%M"), "cfg": {}, "historique": []}
    log(f"=== Tests de nuit, durée maximale {heures} h ===")
    banc = Banc()
    log(f"Données chargées : {len(banc.df)} lignes, {len(banc.cutoffs)} jeudis")
    etat["ref"] = banc.evaluer({}, "reference")
    ecrire_rapport(etat)

    for exp in EXPERIENCES:
        t = time.time()
        h = {"nom": exp["nom"], "titre": exp["titre"]}
        reste = (fin_globale - t) / 60
        if reste < 5:
            h["statut"] = "non lancé (temps de la nuit écoulé)"
            etat["historique"].append(h)
            ecrire_rapport(etat)
            continue
        fin = min(fin_globale, t + exp.get("minutes", 60) * 60)
        log(f"=== {exp['titre']} (max {(fin - t) / 60:.0f} min) ===")
        try:
            r = (exp["fn"] if "fn" in exp else test_variantes)(banc, etat, exp, fin)
            h.update(r)
            h["statut"] = "terminé"
        except TempsDepasse as e:
            h["statut"] = f"interrompu : {e}"
            log(f"  INTERROMPU : {e}")
        except Exception as e:
            h["statut"] = f"échoué : {e!r}"[:300]
            h["erreur"] = traceback.format_exc()
            log(f"  ÉCHEC : {e!r}")
        h["duree_min"] = (time.time() - t) / 60
        etat["historique"].append(h)
        ecrire_rapport(etat)
    log(f"=== Fini. Configuration retenue : {json.dumps(etat['cfg'], ensure_ascii=False)} ===")
    log(f"Rapport : {SORTIE / 'rapport.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--heures", type=float, default=7.0)
    args = parser.parse_args()
    main(args.heures)
