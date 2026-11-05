"""
Stream D · Démo bout-en-bout : prédictions trafic S+1 → planning livreurs optimisé.

Usage :
    python -m src.roadmap.run_demo                                # prédictions synthétiques
    python -m src.roadmap.run_demo --pred outputs/output_<groupe>.csv   # vraies prédictions

Compare 4 scénarios sur la semaine du 07/11 au 11/11 :
    S0  : planning actuel, sans prévision (productivité moyenne constante, +10 % de marge)
    S0* : même logique, avec la marge minimale qui évite tout retard
    S1  : planning calé sur le trafic prédit (même timing de livraison)
    S2  : S1 + les livraisons B2B déplacées vers les heures fluides
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .staffing import approche_min, optimiser_planning, productivite_stops_h, tarif_horaire
from .traffic_to_time import enrich, load_predictions, synthetic_predictions

# --- Hypothèses de demande (à remplacer par l'historique de commandes) -----
ZONES = {  # zone de livraison -> (arc capteur de référence, stops / jour ouvré)
    "8e · Champs-Élysées": ("Champs-Elysées", 700),
    "15e · Convention": ("Convention", 600),
    "6e-7e · Saint-Pères": ("Saint-Pères", 450),
}
MULT_JOUR = {"ouvre": 1.0, "samedi": 0.7, "dimanche": 0.15, "ferie": 0.25}
PART_B2B = {"ouvre": 0.40, "samedi": 0.20, "dimanche": 0.0, "ferie": 0.0}
FENETRE_B2B = (6, 19)  # commerces / bureaux : réception possible de 6h à 19h
# Créneaux choisis par les clients B2C (7h-22h), poids relatifs
PROFIL_B2C = np.zeros(24)
PROFIL_B2C[7:22] = [2, 5, 6, 6, 5, 5, 5, 5, 5, 5, 6, 9, 10, 9, 6]
PROFIL_B2C /= PROFIL_B2C.sum()
# Pratique actuelle supposée : B2B livré uniformément de 8h à 18h
PROFIL_B2B_ACTUEL = np.zeros(24)
PROFIL_B2B_ACTUEL[8:18] = 1.0
# Marges de sécurité testées pour couvrir l'erreur de prévision du modèle.
# Backtest du modèle final (42 semaines) : RMSE débit ≈ 109 véh/h (≈ 20 %).
# Le temps de service ne dépend pas du trafic : une erreur de x % sur la vitesse
# ne pèse qu'environ x/2 % sur la productivité.
MARGES_SENSIBILITE = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]


def jour_type_demande(date: pd.Timestamp) -> str:
    if date.date() == pd.Timestamp("2026-11-11").date():
        return "ferie"
    return {5: "samedi", 6: "dimanche"}.get(date.dayofweek, "ouvre")


def evaluer(plan, b2c, b2b_h, prod, appr) -> dict:
    cap = plan.capacite_reelle(prod, appr)
    demande = b2c + b2b_h
    retard = np.clip(demande - cap, 0, None).sum()
    return {"cout": plan.cout, "heures": plan.heures_livreur, "retard": float(retard),
            "stops": float(demande.sum()),
            "heures_nuit": float(sum(plan.livreurs_presents()[h] for h in range(24) if tarif_horaire(h) > 10))}


def main(pred_path: str | None, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = enrich(load_predictions(pred_path) if pred_path else synthetic_predictions())
    df.to_csv(out / "vitesses_predites.csv", sep=";", index=False)

    # Vitesse de référence "sans prévision" : moyenne 7h-22h de la zone sur la semaine
    v_moy = df[df.heure.between(7, 21)].groupby("arc")["v_parcours_kmh"].mean()

    lignes, details, sensib = [], {}, []
    for zone, (arc, vol) in ZONES.items():
        for date, g in df[df.arc == arc].groupby("date"):
            g = g.sort_values("heure")
            ts = pd.Timestamp(date)
            jt = jour_type_demande(ts)
            total = vol * MULT_JOUR[jt]
            b2b_total = total * PART_B2B[jt]
            b2c = (total - b2b_total) * PROFIL_B2C
            b2b_fixe = b2b_total * PROFIL_B2B_ACTUEL / PROFIL_B2B_ACTUEL.sum()

            v = g["v_parcours_kmh"].to_numpy()
            prod, appr = productivite_stops_h(v), approche_min(v)
            prod_flat = np.full(24, productivite_stops_h(np.array([v_moy[arc]]))[0])
            appr_flat = np.full(24, approche_min(np.array([v_moy[arc]]))[0])

            kw = dict(b2c=b2c, b2b_total=b2b_total, fenetre_b2b=FENETRE_B2B)
            fixe = PROFIL_B2B_ACTUEL if b2b_total > 0 else None
            # S0 : sans prévision, marge 10 %
            p0 = optimiser_planning(prod=prod_flat, appr=appr_flat, b2b_profil_fixe=PROFIL_B2B_ACTUEL,
                                    marge=0.10, **kw)
            # S0* : plus petite marge (pas de 5 %) qui évite tout retard
            for m in np.arange(0.10, 1.51, 0.05):
                p0s = optimiser_planning(prod=prod_flat, appr=appr_flat,
                                         b2b_profil_fixe=PROFIL_B2B_ACTUEL, marge=m, **kw)
                if evaluer(p0s, b2c, b2b_fixe, prod, appr)["retard"] < 0.5:
                    break
            p1 = optimiser_planning(prod=prod, appr=appr, b2b_profil_fixe=PROFIL_B2B_ACTUEL, **kw)
            p2 = optimiser_planning(prod=prod, appr=appr, b2b_profil_fixe=None, **kw) if b2b_total > 0 else p1

            # Sensibilité : S2 avec une marge de sécurité couvrant l'erreur de prévision
            for m_s in MARGES_SENSIBILITE:
                p2m = (optimiser_planning(prod=prod, appr=appr, b2b_profil_fixe=None, marge=m_s, **kw)
                       if b2b_total > 0 else
                       optimiser_planning(prod=prod, appr=appr, b2b_profil_fixe=PROFIL_B2B_ACTUEL, marge=m_s, **kw))
                sensib.append({"marge": m_s, "cout": p2m.cout})

            for nom, plan, b2b_h in [("S0", p0, b2b_fixe), ("S0*", p0s, b2b_fixe),
                                     ("S1", p1, b2b_fixe), ("S2", p2, p2.b2b_par_heure)]:
                r = evaluer(plan, b2c, b2b_h, prod, appr)
                lignes.append({"zone": zone, "date": str(date), "type_jour": jt, "scenario": nom, **r,
                               **({"marge_S0*": round(float(m), 2)} if nom == "S0*" else {})})
            details[f"{zone}|{date}"] = {
                "v_parcours": v.round(2).tolist(), "prod": prod.round(2).tolist(),
                "b2c": b2c.round(1).tolist(), "b2b_S1": b2b_fixe.round(1).tolist(),
                "b2b_S2": np.asarray(p2.b2b_par_heure).round(1).tolist(),
                "livreurs_S0": p0.livreurs_presents().tolist(),
                "livreurs_S1": p1.livreurs_presents().tolist(),
                "livreurs_S2": p2.livreurs_presents().tolist(),
                "shifts_S2": [[s.debut, s.duree, n] for s, n in p2.effectifs.items()],
            }

    res = pd.DataFrame(lignes)
    res.to_csv(out / "resultats_par_zone_jour.csv", sep=";", index=False)
    synth = res.groupby("scenario")[["cout", "heures", "retard", "stops", "heures_nuit"]].sum()
    synth["cout_par_stop"] = synth["cout"] / synth["stops"]
    synth["pct_retard"] = 100 * synth["retard"] / synth["stops"]
    synth["gain_vs_S0*_pct"] = 100 * (1 - synth["cout"] / synth.loc["S0*", "cout"])
    synth.round(2).to_csv(out / "synthese_scenarios.csv", sep=";")
    (out / "details.json").write_text(json.dumps(details, ensure_ascii=False))
    sens = pd.DataFrame(sensib).groupby("marge")["cout"].sum().to_frame()
    sens["gain_vs_S0*_pct"] = 100 * (1 - sens["cout"] / synth.loc["S0*", "cout"])
    sens.round(1).to_csv(out / "sensibilite_marge.csv", sep=";")
    print(synth.round(2).to_string())
    print("\nSensibilité de S2 à la marge de sécurité :\n", sens.round(1).to_string())
    print("\nMarges S0* nécessaires :\n", res[res.scenario == "S0*"][["zone", "date", "marge_S0*"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=None, help="CSV output_<groupe>.csv (sinon prédictions synthétiques)")
    ap.add_argument("--out", default="outputs/stream_d")
    a = ap.parse_args()
    main(a.pred, a.out)
