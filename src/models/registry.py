"""Noms de modèles utilisables en ligne de commande (evaluate.py, predict.py)."""

from src.models.baseline import SeasonalBaseline
from src.models.gbm import GBMModel
from src.models.mlp import MLPModel
from src.models.xgb import XGBModel

MODELS = {
    "baseline": SeasonalBaseline,
    "gbm": GBMModel,
    "mlp": MLPModel,
    "xgb": XGBModel,
}
