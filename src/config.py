"""Constantes partagées par tous les streams (chemins, arcs, horizon de prédiction)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
# Un dossier par source, rempli par src/data/update.py (connecteurs dans src/data/sources/).
TRAFIC_DIR = DATA_DIR / "trafic"
METEO_DIR = DATA_DIR / "meteo"
CALENDRIER_DIR = DATA_DIR / "calendrier"
EVENEMENTS_DIR = DATA_DIR / "evenements"
CHANTIERS_DIR = DATA_DIR / "chantiers"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT / "outputs"

TRAFFIC_PATH = PROCESSED_DIR / "traffic_hourly.parquet"
EXTERNAL_PATH = PROCESSED_DIR / "external.parquet"

TZ = "Europe/Paris"
TARGETS = ["debit_horaire", "taux_occupation"]

# Arcs du brief : nom attendu dans le CSV de rendu -> nœuds amont/aval.
# Champs-Elysées : l'arc 4264 est muet depuis le 19/11/2025. Proxys possibles
# sur la même avenue : 4401 et 4402 (à trancher au tutorat du 05/11).
ARCS = {
    "Champs-Elysées": {
        "nd_amont": "Av_Champs_Elysees-Washington",
        "nd_aval": "Av_Champs_Elysees-Berri",
    },
    "Convention": {
        "nd_amont": "Lecourbe-Convention",
        "nd_aval": "Convention-Blomet",
    },
    "Saint-Pères": {
        "nd_amont": "Sts_Peres-Voltaire",
        "nd_aval": "Sts_Peres-Universite",
    },
}

# On télécharge toutes les rues des 3 arcs (proxys, arcs voisins) : sous-chaînes cherchées dans `libelle`.
# Les libellés varient selon les années (ex. AV_Champs_Elysees / Av_des_Champs_Elysees).
STREETS = ["Champs_Elysees", "Sts_Peres", "Convention"]

# Point météo : centre de Paris (les 3 arcs sont à moins de 5 km les uns des autres).
PARIS_LAT, PARIS_LON = 48.8566, 2.3522

# Horizon de la version notée : données publiées en J-1, téléchargées le 06/11,
# donc dernière heure connue le 05/11 23:00 ; on prédit le 07/11 00:00 -> 11/11 23:00.
GAP_DAYS = 1        # jour non prédit entre le cutoff et le début de la fenêtre
HORIZON_DAYS = 5
MIN_LAG_HOURS = 24 * (GAP_DAYS + HORIZON_DAYS)   # 144 h : tout lag plus court fuit

SUBMISSION_START = "2026-11-07 00:00"
SUBMISSION_END = "2026-11-11 23:00"
