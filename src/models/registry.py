"""Modèles chargés seulement à leur utilisation (dépendances natives facultatives)."""
from importlib import import_module


def _factory(module, name):
    def create(**params):
        return getattr(import_module(module), name)(**params)
    return create


MODELS = {
    "baseline": _factory("src.models.baseline", "SeasonalBaseline"),
    "gbm": _factory("src.models.gbm", "GBMModel"),
    "mlp": _factory("src.models.mlp", "MLPModel"),
    "xgb": _factory("src.models.xgb", "XGBModel"),
}
