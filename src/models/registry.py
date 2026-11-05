"""Modèles disponibles, par nom : `--model <nom>` dans evaluate.py et predict.py."""
from src.models.baseline import SeasonalBaseline
from src.models.ensemble import EnsembleModel
from src.models.xgb import XGBModel

MODELS = {"baseline": SeasonalBaseline, "xgb": XGBModel, "ensemble": EnsembleModel}
