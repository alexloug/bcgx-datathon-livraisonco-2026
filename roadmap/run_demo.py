"""Stream D : simulation du planning des livreurs sur la semaine du rendu (07/11 -> 11/11).

Usage :
    python -m roadmap.run_demo                                                   # trafic simulé
    python -m roadmap.run_demo --pred output_<groupe>.csv --calage relatif       # vraies prédictions

Quatre scénarios sont comparés pour 3 zones (une par arc) et 5 jours :
    S0  : planning actuel simulé, sans prévision (vitesse moyenne de la semaine) et 10 % de marge
    S0* : même logique, avec la plus petite marge qui évite tout retard (référence à service égal)
    S1  : planning calé heure par heure sur la vitesse prédite, B2B livré de 8 h à 18 h
    S2  : S1, en laissant l'optimiseur choisir l'heure des livraisons B2B (6 h -> 19 h)
Une analyse de sensibilité donne aussi le coût de S2 avec une marge de sécurité de 0 à 25 %,
et gain_annuel.csv extrapole le gain sur une année à partir d'une semaine type.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from roadmap.staffing import approche_min, optimiser_planning, productivite_stops_h, tarif_horaire
from roadmap.traffic_to_time import enrich, load_predictions, synthetic_predictions

# Hypothèses de demande, à remplacer par l'historique de commandes de LivraisonCo.
ZONES = {  # zone de livraison -> (arc dont le trafic sert de référence, livraisons par jour ouvré)
    "8e · Champs-Élysées": ("Champs-Elysées", 700),
    "15e · Convention": ("Convention", 600),
    "6e-7e · Saint-Pères": ("Saint-Pères", 450),
}
VOLUME_RELATIF = {"ouvre": 1.0, "samedi": 0.7, "dimanche": 0.15, "ferie": 0.25}
PART_B2B = {"ouvre": 0.40, "samedi": 0.20, "dimanche": 0.0, "ferie": 0.0}
FENETRE_B2B = (6, 19)  # commerces et bureaux

# Répartition des livraisons B2C entre 7 h et 22 h, selon les créneaux choisis par les clients.
PROFIL_B2C = np.zeros(24)
PROFIL_B2C[7:22] = [2, 5, 6, 6, 5, 5, 5, 5, 5, 5, 6, 9, 10, 9, 6]
PROFIL_B2C /= PROFIL_B2C.sum()

# Pratique actuelle supposée : B2B réparti uniformément de 8 h à 18 h.
PROFIL_B2B_ACTUEL = np.zeros(24)
PROFIL_B2B_ACTUEL[8:18] = 1.0

# Marges de sécurité testées. En backtest, le modèle final se trompe d'environ 20 %
# sur le débit (RMSE 109 véh/h). Le temps de service ne dépend pas du trafic, donc
# une erreur de x % sur la vitesse ne pèse qu'environ x / 2 % sur la productivité.
MARGES_SENSIBILITE = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
MARGE_RETENUE = 0.10

# Extrapolation annuelle. La semaine du rendu (samedi -> mercredi) contient un dimanche
# et un férié : on recompose une semaine type à partir du coût moyen par type de jour.
JOURS_SEMAINE_TYPE = {"ouvre": 5, "samedi": 1, "dimanche": 1}
SEMAINES_PAR_AN = 52


def jour_type_demande(date: pd.Timestamp) -> str:
    if date.date() == pd.Timestamp("2026-11-11").date():
        return "ferie"
    return {5: "samedi", 6: "dimanche"}.get(date.dayofweek, "ouvre")


def demande_du_jour(volume_ouvre: float, type_jour: str) -> tuple[np.ndarray, float, np.ndarray]:
    """Renvoie la demande B2C par heure, le volume B2B du jour et sa répartition actuelle."""
    total = volume_ouvre * VOLUME_RELATIF[type_jour]
    b2b_total = total * PART_B2B[type_jour]
    b2c = (total - b2b_total) * PROFIL_B2C
    b2b_actuel = b2b_total * PROFIL_B2B_ACTUEL / PROFIL_B2B_ACTUEL.sum()
    return b2c, b2b_total, b2b_actuel


def evaluer(plan, b2c, b2b_h, prod, appr) -> dict:
    """Coût, heures et livraisons hors créneau d'un plan, avec la productivité réelle."""
    demande = b2c + b2b_h
    retard = np.clip(demande - plan.capacite_reelle(prod, appr), 0, None).sum()
    heures_nuit = sum(plan.livreurs_presents()[h] for h in range(24) if tarif_horaire(h) > 10)
    return {"cout": plan.cout, "heures": plan.heures_livreur, "retard": float(retard),
            "stops": float(demande.sum()), "heures_nuit": float(heures_nuit)}


def simuler_zone_jour(zone: str, date, g: pd.DataFrame, volume_ouvre: float, v_moyenne: float):
    """Résout les 4 scénarios et la sensibilité à la marge pour une zone et un jour."""
    type_jour = jour_type_demande(pd.Timestamp(date))
    b2c, b2b_total, b2b_actuel = demande_du_jour(volume_ouvre, type_jour)
    demande = dict(b2c=b2c, b2b_total=b2b_total, fenetre_b2b=FENETRE_B2B)

    # Productivité réelle (vitesse prédite heure par heure) et vue sans prévision (vitesse moyenne).
    v = g["v_parcours_kmh"].to_numpy()
    prod, appr = productivite_stops_h(v), approche_min(v)
    prod_moy = np.full(24, productivite_stops_h(np.array([v_moyenne]))[0])
    appr_moy = np.full(24, approche_min(np.array([v_moyenne]))[0])

    p0 = optimiser_planning(prod=prod_moy, appr=appr_moy, b2b_profil_fixe=PROFIL_B2B_ACTUEL,
                            marge=0.10, **demande)
    # S0* : on augmente la marge par pas de 5 % jusqu'à ne plus avoir de retard.
    for marge_s0 in np.arange(0.10, 1.51, 0.05):
        p0_etoile = optimiser_planning(prod=prod_moy, appr=appr_moy,
                                       b2b_profil_fixe=PROFIL_B2B_ACTUEL, marge=marge_s0, **demande)
        if evaluer(p0_etoile, b2c, b2b_actuel, prod, appr)["retard"] < 0.5:
            break
    p1 = optimiser_planning(prod=prod, appr=appr, b2b_profil_fixe=PROFIL_B2B_ACTUEL, **demande)
    # Sans B2B (dimanche, férié), S2 est identique à S1.
    p2 = optimiser_planning(prod=prod, appr=appr, b2b_profil_fixe=None, **demande) if b2b_total > 0 else p1

    sensibilite = []
    for marge in MARGES_SENSIBILITE:
        profil = None if b2b_total > 0 else PROFIL_B2B_ACTUEL
        p2_marge = optimiser_planning(prod=prod, appr=appr, b2b_profil_fixe=profil, marge=marge, **demande)
        sensibilite.append({"zone": zone, "date": str(date), "type_jour": type_jour,
                            "marge": marge, "cout": p2_marge.cout})

    lignes = []
    for nom, plan, b2b_h in [("S0", p0, b2b_actuel), ("S0*", p0_etoile, b2b_actuel),
                             ("S1", p1, b2b_actuel), ("S2", p2, p2.b2b_par_heure)]:
        ligne = {"zone": zone, "date": str(date), "type_jour": type_jour, "scenario": nom,
                 **evaluer(plan, b2c, b2b_h, prod, appr)}
        if nom == "S0*":
            ligne["marge_S0*"] = round(float(marge_s0), 2)
        lignes.append(ligne)

    detail = {
        "v_parcours": v.round(2).tolist(), "prod": prod.round(2).tolist(),
        "b2c": b2c.round(1).tolist(), "b2b_S1": b2b_actuel.round(1).tolist(),
        "b2b_S2": np.asarray(p2.b2b_par_heure).round(1).tolist(),
        "livreurs_S0": p0.livreurs_presents().tolist(),
        "livreurs_S1": p1.livreurs_presents().tolist(),
        "livreurs_S2": p2.livreurs_presents().tolist(),
        "shifts_S2": [[s.debut, s.duree, n] for s, n in p2.effectifs.items()],
    }
    return lignes, detail, sensibilite


def gain_annuel(res: pd.DataFrame, sensibilite: pd.DataFrame) -> pd.DataFrame:
    """Coût d'une semaine type et d'une année, pour S0*, S2 et S2 avec la marge retenue."""
    couts = {
        "S0*": res[res.scenario == "S0*"],
        "S2": res[res.scenario == "S2"],
        f"S2 + marge {MARGE_RETENUE:.0%}": sensibilite[np.isclose(sensibilite.marge, MARGE_RETENUE)],
    }
    lignes = []
    for nom, df in couts.items():
        # Coût moyen d'un jour de chaque type, sommé sur les zones.
        par_type = df.groupby(["zone", "type_jour"])["cout"].mean().groupby("type_jour").sum()
        semaine = sum(n * par_type[t] for t, n in JOURS_SEMAINE_TYPE.items())
        lignes.append({"scenario": nom, "cout_semaine_type": semaine,
                       "cout_annuel": semaine * SEMAINES_PAR_AN})
    out = pd.DataFrame(lignes).set_index("scenario")
    out["gain_annuel_vs_S0*"] = out.loc["S0*", "cout_annuel"] - out["cout_annuel"]
    out["gain_pct"] = 100 * out["gain_annuel_vs_S0*"] / out.loc["S0*", "cout_annuel"]
    return out


def main(pred_path: str | None, out_dir: str, calage: str = "physique") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions(pred_path) if pred_path else synthetic_predictions()
    df = enrich(predictions, calage=calage)
    df.to_csv(out / "vitesses_predites.csv", sep=";", index=False)

    # Vitesse « sans prévision » : moyenne de 7 h à 21 h sur la semaine, par arc.
    v_moyenne = df[df.heure.between(7, 21)].groupby("arc")["v_parcours_kmh"].mean()

    lignes, details, sensibilite = [], {}, []
    for zone, (arc, volume_ouvre) in ZONES.items():
        for date, g in df[df.arc == arc].groupby("date"):
            l, d, s = simuler_zone_jour(zone, date, g.sort_values("heure"), volume_ouvre, v_moyenne[arc])
            lignes += l
            details[f"{zone}|{date}"] = d
            sensibilite += s

    res = pd.DataFrame(lignes)
    res.round(2).to_csv(out / "resultats_par_zone_jour.csv", sep=";", index=False)

    synth = res.groupby("scenario")[["cout", "heures", "retard", "stops", "heures_nuit"]].sum()
    synth["cout_par_stop"] = synth["cout"] / synth["stops"]
    synth["pct_retard"] = 100 * synth["retard"] / synth["stops"]
    synth["gain_vs_S0*_pct"] = 100 * (1 - synth["cout"] / synth.loc["S0*", "cout"])
    synth.round(2).to_csv(out / "synthese_scenarios.csv", sep=";")

    sensibilite = pd.DataFrame(sensibilite)
    sens = sensibilite.groupby("marge")["cout"].sum().to_frame()
    sens["gain_vs_S0*_pct"] = 100 * (1 - sens["cout"] / synth.loc["S0*", "cout"])
    sens.round(1).to_csv(out / "sensibilite_marge.csv", sep=";")

    annuel = gain_annuel(res, sensibilite)
    annuel = annuel.round({"cout_semaine_type": 0, "cout_annuel": 0, "gain_annuel_vs_S0*": 0, "gain_pct": 1})
    annuel.to_csv(out / "gain_annuel.csv", sep=";")

    (out / "details.json").write_text(json.dumps(details, ensure_ascii=False))

    print(synth.round(2).to_string())
    print("\nSensibilité de S2 à la marge de sécurité :\n", sens.round(1).to_string())
    print("\nExtrapolation sur une année (semaine type) :\n", annuel.to_string())
    print("\nMarges nécessaires pour S0* :\n",
          res[res.scenario == "S0*"][["zone", "date", "marge_S0*"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default=None,
                        help="CSV de prédictions au format du rendu (sinon trafic simulé)")
    parser.add_argument("--out", default=None,
                        help="dossier de sortie (outputs/stream_d par défaut, trafic_simule/ sans --pred)")
    parser.add_argument("--calage", choices=["physique", "relatif"], default="physique",
                        help="relatif conseillé sur les vraies prédictions (voir traffic_to_time.py)")
    args = parser.parse_args()
    # Sans --pred, les résultats sur trafic simulé ne doivent pas écraser ceux sur nos prédictions.
    out_dir = args.out or ("outputs/stream_d" if args.pred else "outputs/stream_d/trafic_simule")
    main(args.pred, out_dir, args.calage)
