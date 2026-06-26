class ModelNotLoadedError(Exception):
    """Levée quand une requête arrive alors que le modèle n'est pas chargé."""


class ModelOutputError(Exception):
    """Levée quand la sortie du modèle ne correspond pas à ce qui est attendu

    (ex: nombre de classes renvoyées différent de settings.CLASS_LABELS).
    Indique un problème de configuration/déploiement, pas une erreur côté appelant.
    """
