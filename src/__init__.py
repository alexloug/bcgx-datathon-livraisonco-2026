"""LivraisonCo : prévision du débit et de l'occupation de 3 arcs parisiens."""

# LightGBM doit être chargé AVANT pandas : sous Windows, si pandas (donc pyarrow) est chargé en premier, l'entraînement
# de LightGBM plante (« access violation » dans LGBM_DatasetSetField). Ce paquet est toujours importé en premier.
try:
    import lightgbm  # noqa: F401
except ImportError:
    pass
