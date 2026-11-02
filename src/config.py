"""Constantes partagées par tous les streams (chemins, arcs, horizon de prédiction)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR                      # instantané versionné des exports (repli si data/recent est vide)
RECENT_DIR = DATA_DIR / "recent"        # exports frais de fetch_recent.py (non versionnés)
ARCHIVES_DIR = DATA_DIR / "archives"    # zips annuels 2010-2025 (non versionnés)
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT / "outputs"

TRAFFIC_PATH = PROCESSED_DIR / "traffic_hourly.parquet"
EXTERNAL_PATH = PROCESSED_DIR / "external.parquet"

TZ = "Europe/Paris"
TARGETS = ["debit_horaire", "taux_occupation"]

# Arcs du brief : nom attendu dans le CSV de rendu -> fichier brut + nœuds amont/aval.
# Champs-Elysées : l'arc 4264 est muet depuis le 19/11/2025. Proxys possibles
# sur la même avenue : 4401 et 4402 (à trancher au tutorat du 05/11).
ARCS = {
    "Champs-Elysées": {
        "raw_file": "champs_elysees.csv",
        "nd_amont": "Av_Champs_Elysees-Washington",
        "nd_aval": "Av_Champs_Elysees-Berri",
    },
    "Convention": {
        "raw_file": "convention.csv",
        "nd_amont": "Lecourbe-Convention",
        "nd_aval": "Convention-Blomet",
    },
    "Saint-Pères": {
        "raw_file": "saints_peres.csv",
        "nd_amont": "Sts_Peres-Voltaire",
        "nd_aval": "Sts_Peres-Universite",
    },
}

# Horizon de la version notée : données publiées en J-1, téléchargées le 06/11,
# donc dernière heure connue le 05/11 23:00 ; on prédit le 07/11 00:00 -> 11/11 23:00.
GAP_DAYS = 1        # jour non prédit entre le cutoff et le début de la fenêtre
HORIZON_DAYS = 5
MIN_LAG_HOURS = 24 * (GAP_DAYS + HORIZON_DAYS)   # 144 h : tout lag plus court fuit

SUBMISSION_START = "2026-11-07 00:00"
SUBMISSION_END = "2026-11-11 23:00"
