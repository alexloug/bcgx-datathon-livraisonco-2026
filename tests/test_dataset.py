import numpy as np
import pandas as pd

from src.config import CHAMPS_PROXIES, TARGETS
from src.data import build_dataset
from src.data.build_dataset import apply_ratios, clean_targets, fit_ratios, reconstruct_champs
from src.data.sources import evenements

HOURS = pd.date_range("2025-01-06", periods=24 * 28, freq="h", tz="UTC")


def series(debit, occ, etat="Ouvert"):
    return pd.DataFrame({"debit_horaire": debit, "taux_occupation": occ, "etat_arc": etat}, index=HOURS)


def test_clean_targets_flags_stuck_spikes_and_states():
    rng = np.random.default_rng(0)
    df = series(rng.uniform(100, 500, len(HOURS)), rng.uniform(1, 10, len(HOURS)))
    df.iloc[10:20, 0] = 300.0                 # débit figé 10 h
    df.iloc[50, 0] = 50_000.0                 # pic absurde
    df.iloc[60, 1] = np.nan                   # occupation manquante
    df.iloc[70, 2] = "Invalide"
    ok = clean_targets(df)
    assert not ok.iloc[10:20].any()
    assert not ok.iloc[[50, 60, 70]].any()
    assert ok.drop(ok.index[[*range(10, 20), 50, 60, 70]]).all()


def test_ratios_recover_a_proportional_target():
    proxy = pd.Series(np.random.default_rng(1).uniform(100, 500, len(HOURS)), index=HOURS)
    target = 1.5 * proxy
    rebuilt = apply_ratios(proxy, fit_ratios(target, proxy))
    assert np.allclose(rebuilt, target)


def test_reconstruct_champs_fills_only_missing_hours():
    rng = np.random.default_rng(2)
    proxies = {pid: series(rng.uniform(100, 500, len(HOURS)), rng.uniform(1, 10, len(HOURS))).assign(valide=True)
               for spec in CHAMPS_PROXIES.values() for pid in spec["arcs"]}
    first = CHAMPS_PROXIES["debit_horaire"]["arcs"][0]
    target = (proxies[first][TARGETS] * 2).assign(etat_arc="Ouvert", valide=True)
    target.iloc[-48:, :2] = np.nan
    target.iloc[-48:, -1] = False
    out, _ = reconstruct_champs(target, proxies)
    assert (out["source_cible"].iloc[:-48] == "mesure").all()
    assert (out["source_cible"].iloc[-48:] == "proxy").all()
    assert np.allclose(out["debit_horaire"].iloc[-48:], 2 * proxies[first]["debit_horaire"].iloc[-48:])


def test_expand_manual_defaults_when_columns_missing():
    df = pd.DataFrame({"nom": ["A"], "date_debut": ["2024-11-11"], "date_fin": ["2024-11-11"],
                       "recurrence": ["ponctuelle"], "rues": ["Champs_Elysees"], "source": [""], "a_verifier": ["non"]})
    out = evenements.expand_manual(df, 2025)
    assert out.loc[0, ["type", "heure_debut", "heure_fin"]].tolist() == ["fermeture", 0, 23]


def test_event_flags_respect_hours_and_type(monkeypatch):
    ev = pd.DataFrame({"nom": ["matin", "nuit", "greve"],
                       "debut": pd.to_datetime(["2024-11-11", "2024-12-31", "2024-11-12"]).date,
                       "fin": pd.to_datetime(["2024-11-11", "2025-01-01", "2024-11-12"]).date,
                       "heure_debut": [7, 18, 0], "heure_fin": [14, 3, 23],
                       "type": ["fermeture", "fermeture", "perturbation"],
                       "rues": ["Champs_Elysees", "Champs_Elysees", "Convention"], "a_verifier": [False] * 3})
    monkeypatch.setattr(evenements, "load", lambda name="evenements": ev)
    t = pd.date_range("2024-11-10", "2025-01-02", freq="h", tz="Europe/Paris").tz_convert("UTC")
    df = pd.concat([pd.DataFrame({"arc": a, "datetime": t}) for a in ["Champs-Elysées", "Convention"]], ignore_index=True)
    out = build_dataset.event_flags(df)
    paris = df["datetime"].dt.tz_convert("Europe/Paris").dt.tz_localize(None)
    champs, conv = df["arc"] == "Champs-Elysées", df["arc"] == "Convention"
    ferm = paris[out["evenement_fermeture"] & champs]
    assert ferm.min() == pd.Timestamp("2024-11-11 07:00") and pd.Timestamp("2024-11-11 14:00") in set(ferm)
    assert pd.Timestamp("2024-11-11 15:00") not in set(ferm)
    assert {pd.Timestamp("2024-12-31 18:00"), pd.Timestamp("2025-01-01 03:00")} <= set(ferm)
    assert pd.Timestamp("2025-01-01 04:00") not in set(ferm)
    assert (out["evenement_perturbation"] & conv).sum() == 24 and not (out["evenement_perturbation"] & champs).any()
    assert (out["evenement"] == (out["evenement_fermeture"] | out["evenement_perturbation"])).all()
