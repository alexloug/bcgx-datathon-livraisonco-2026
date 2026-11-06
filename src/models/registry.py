"""Noms de modèles utilisables en ligne de commande (evaluate.py, predict.py)."""

from src.models.baseline import SeasonalBaseline
from src.models.gbm import GBMModel
from src.models.mlp import MLPModel
from src.models.xgb import XGBModel
from src.models.xgb_similaires import XGBSimModel

MODELS = {
    "baseline": SeasonalBaseline,
    "gbm": GBMModel,
    "mlp": MLPModel,
    "xgb": XGBModel,
    # Arcs similaires (dataset_similaires.parquet, docs/arcs_similaires_backtest.md)
    "xgb_rel": lambda: XGBSimModel(relative=True),
    "xgb_groupe": lambda: XGBSimModel(relative=False, groupe=True),
    "xgb_rel_groupe": lambda: XGBSimModel(relative=True, groupe=True),
    "xgb_pool": lambda: XGBSimModel(relative=True, pool=True),
    "xgb_pool_w5": lambda: XGBSimModel(relative=True, pool=True, brief_weight=5.0),
    "xgb_pool_groupe": lambda: XGBSimModel(relative=True, groupe=True, pool=True, brief_weight=5.0),
}
