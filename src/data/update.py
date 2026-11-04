"""Met à jour toutes les sources de data/ depuis l'open data. Une source en échec n'arrête pas les autres.

Usage :
    python -m src.data.update                               # toutes les sources, trafic sur 13 mois
    python -m src.data.update --sources trafic meteo        # seulement certaines sources
    python -m src.data.update --history 2010 2025           # + archives trafic manquantes (~50 s et 200 Mo par an)
    python -m src.data.update --history 2024 2025 --force   # retélécharge ces archives
"""

import argparse
import time
import traceback

from src.data.sources import calendrier, chantiers, evenements, meteo, trafic

SOURCES = {
    "trafic": trafic,
    "meteo": meteo,
    "calendrier": calendrier,
    "evenements": evenements,
    "chantiers": chantiers,
}


def main(sources=tuple(SOURCES), history=None, force=False) -> bool:
    failed = []
    for name in sources:
        start = time.time()
        try:
            if name == "trafic":
                trafic.update(history, force)
            else:
                SOURCES[name].update()
        except Exception:
            traceback.print_exc()
            failed.append(name)
        print(f"         ({name} : {time.time() - start:.0f} s)")
    if failed:
        print(f"ÉCHEC : {', '.join(failed)}")
    return not failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mise à jour des données open data (un dossier par source dans data/)")
    parser.add_argument("--sources", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument("--history", nargs=2, type=int, metavar=("DEBUT", "FIN"),
                        help="années d'archives trafic à récupérer, bornes incluses (2010-2025)")
    parser.add_argument("--force", action="store_true", help="retélécharger les archives déjà présentes")
    args = parser.parse_args()
    years = list(range(args.history[0], args.history[1] + 1)) if args.history else None
    raise SystemExit(0 if main(args.sources, years, args.force) else 1)
