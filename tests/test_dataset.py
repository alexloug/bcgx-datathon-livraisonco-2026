import numpy as np
import pandas as pd

from src.config import CHAMPS_PROXIES, TARGETS
from src.data.build_dataset import apply_ratios, clean_targets, fit_ratios, reconstruct_champs

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
