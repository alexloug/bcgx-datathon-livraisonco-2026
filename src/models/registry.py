"""Noms de modèles utilisables en ligne de commande (evaluate.py, predict.py)."""

from src.models.baseline import SeasonalBaseline
from src.models.gbm import GBMModel

MODELS = {
    "baseline": SeasonalBaseline,
    "gbm": GBMModel,
}
