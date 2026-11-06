"""Met à jour toutes les sources de data/ depuis l'open data. Une source en échec n'arrête pas les autres.

Usage :
    python -m src.data.update                               # sources du modèle, trafic sur 13 mois
    python -m src.data.update --all                         # + chantiers et « Que faire à Paris » (optionnels)
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
# Sources que le modèle n'utilise pas : mises à jour seulement avec --all (ou --sources chantiers).
OPTIONAL = ("chantiers",)


def main(sources=tuple(s for s in SOURCES if s not in OPTIONAL), history=None, force=False, que_faire=False) -> bool:
    failed = []
    for name in sources:
        start = time.time()
        try:
            if name == "trafic":
                trafic.update(history, force)
            elif name == "evenements":
                evenements.update(que_faire)
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
    parser.add_argument("--sources", nargs="+", choices=list(SOURCES), help="par défaut : toutes sauf chantiers")
    parser.add_argument("--all", action="store_true", help="ajoute les sources optionnelles : chantiers et « Que faire à Paris »")
    parser.add_argument("--history", nargs=2, type=int, metavar=("DEBUT", "FIN"),
                        help="années d'archives trafic à récupérer, bornes incluses (2010-2025)")
    parser.add_argument("--force", action="store_true", help="retélécharger les archives déjà présentes")
    args = parser.parse_args()
    years = list(range(args.history[0], args.history[1] + 1)) if args.history else None
    sources = args.sources or [s for s in SOURCES if args.all or s not in OPTIONAL]
    raise SystemExit(0 if main(sources, years, args.force, que_faire=args.all) else 1)
