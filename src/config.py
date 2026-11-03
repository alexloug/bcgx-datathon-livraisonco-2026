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
DATASET_PATH = PROCESSED_DIR / "dataset.parquet"   # src/data/build_dataset.py, doc : docs/dataset_entrainement.md

TZ = "Europe/Paris"
TARGETS = ["debit_horaire", "taux_occupation"]

# Arcs du brief : nom attendu dans le CSV de rendu -> identifiant, nœuds amont/aval, rue (cf. STREETS).
# Champs-Elysées : l'arc 4264 est muet depuis le 19/11/2025, reconstruit par CHAMPS_PROXIES.
ARCS = {
    "Champs-Elysées": {
        "arc_id": "4264",
        "rue": "Champs_Elysees",
        "nd_amont": "Av_Champs_Elysees-Washington",
        "nd_aval": "Av_Champs_Elysees-Berri",
    },
    "Convention": {
        "arc_id": "5671",
        "rue": "Convention",
        "nd_amont": "Lecourbe-Convention",
        "nd_aval": "Convention-Blomet",
    },
    "Saint-Pères": {
        "arc_id": "191",
        "rue": "Sts_Peres",
        "nd_amont": "Sts_Peres-Voltaire",
        "nd_aval": "Sts_Peres-Universite",
    },
}

# On télécharge toutes les rues des 3 arcs (proxys, arcs voisins) : sous-chaînes cherchées dans `libelle`.
# Les libellés varient selon les années (ex. AV_Champs_Elysees / Av_des_Champs_Elysees).
STREETS = ["Champs_Elysees", "Sts_Peres", "Convention"]

# Proxys de l'arc 4264, même sens de circulation, en aval (notebook 03) :
# 4274 Berri -> La Boétie (tronçon suivant), 4281 Colisée -> Rond-Point. Par cible, ordre de préférence,
# et « moyenne » = moyenne des reconstructions disponibles. 4267 (amont) est muet de 2021 à 2025 et depuis 08/2026.
CHAMPS_PROXIES = {
    "debit_horaire": {"arcs": ["4274", "4281"], "methode": "premier"},
    "taux_occupation": {"arcs": ["4274", "4281"], "methode": "moyenne"},
}

# Début du dataset d'entraînement : fin du couvre-feu Covid (20/06/2021). Les prévisions météo archivées
# (backtest) commencent en 2021 ; avant, ruptures de régime sur Champs-Elysées (2019) et Convention (2020).
TRAIN_START = "2021-06-21 00:00"   # heure de Paris

# Point météo : centre de Paris (les 3 arcs sont à moins de 5 km les uns des autres).
PARIS_LAT, PARIS_LON = 48.8566, 2.3522

# Horizon de la version notée : données publiées en J-1, téléchargées le 06/11,
# donc dernière heure connue le 05/11 23:00 ; on prédit le 07/11 00:00 -> 11/11 23:00.
GAP_DAYS = 1        # jour non prédit entre le cutoff et le début de la fenêtre
HORIZON_DAYS = 5
MIN_LAG_HOURS = 24 * (GAP_DAYS + HORIZON_DAYS)   # 144 h : tout lag plus court fuit

SUBMISSION_START = "2026-11-07 00:00"
SUBMISSION_END = "2026-11-11 23:00"
