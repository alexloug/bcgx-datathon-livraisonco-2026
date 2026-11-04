"""Modèles disponibles ; chargés à l'utilisation. Gradient boosting alternatif et MLP : voir exploration/."""
from importlib import import_module


def _factory(module, name):
    def create(**params):
        return getattr(import_module(module), name)(**params)
    return create


MODELS = {
    "baseline": _factory("src.models.baseline", "SeasonalBaseline"),
    "xgb": _factory("src.models.xgb", "XGBModel"),
}
