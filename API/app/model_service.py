"""Service d'inférence.

Le modèle candidat final est un pipeline scikit-learn (ColumnTransformer +
XGBClassifier) sérialisé via joblib (models/XGBoost.pkl). Le pipeline embarque
tout le preprocessing (imputation, scaling, one-hot encoding) : l'API lui
transmet directement les features brutes sous forme de DataFrame, sans
reproduire la transformation manuellement (évite tout risque de divergence
train/serve).
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import settings
from .exceptions import ModelNotLoadedError, ModelOutputError

logger = logging.getLogger(__name__)


class FailureTypeModelService:
    """Wrapper autour du pipeline scikit-learn de classification du type de défaillance."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or settings.MODEL_PATH
        self.pipeline = None
        self.loaded_at: datetime | None = None

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def load(self) -> None:
        if not os.path.exists(self.model_path):
            logger.warning(
                "Modèle introuvable à '%s'. L'API démarre en mode dégradé "
                "(/health renverra model_loaded=false). Déposez votre modèle "
                "entraîné (joblib) à cet emplacement pour activer /predict.",
                self.model_path,
            )
            self.pipeline = None
            self.loaded_at = None
            return

        self.pipeline = joblib.load(self.model_path)
        self.loaded_at = datetime.now(timezone.utc)
        logger.info("Pipeline chargé depuis '%s'.", self.model_path)

    def get_info(self) -> dict[str, Any]:
        if not self.is_loaded:
            return {"model_type": None, "n_features_after_preprocessing": None}

        clf = self.pipeline.named_steps.get("clf")
        return {
            "model_type": f"{type(self.pipeline).__name__}({' -> '.join(self.pipeline.named_steps)})",
            "n_features_after_preprocessing": getattr(clf, "n_features_in_", None),
        }

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        if not self.is_loaded:
            raise ModelNotLoadedError(
                "Le modèle n'est pas chargé. Vérifiez MODEL_PATH et /health."
            )

        # Étape 1 — Construction du DataFrame d'entrée.
        # Les features arrivent déjà validées/typées par Pydantic (SensorData) ;
        # on les remet ici dans un DataFrame 1 ligne avec les noms de colonnes
        # exacts attendus par le ColumnTransformer ("prep"), dans l'ordre de
        # settings.RAW_FEATURES. Aucune transformation manuelle n'est faite ici :
        # imputation, scaling (numérique) et one-hot encoding (catégoriel) sont
        # entièrement délégués au pipeline pour rester identiques à l'entraînement.
        row = {col: features[col] for col in settings.RAW_FEATURES}
        df = pd.DataFrame([row], columns=settings.RAW_FEATURES)

        # Étape 2 — Inférence. predict_proba() fait passer le DataFrame dans
        # "prep" (preprocessing) puis "clf" (XGBClassifier) et renvoie un
        # vecteur de probabilités, une valeur par classe, dans l'ordre de
        # clf.classes_ (= [0, 1, 2, 3, 4], donc aligné sur settings.CLASS_LABELS).
        probabilities = self.pipeline.predict_proba(df)[0]

        # Garde-fou : si le modèle a été ré-entraîné avec un nombre de classes
        # différent sans mettre à jour CLASS_LABELS, on échoue explicitement
        # plutôt que de renvoyer un label erroné.
        if len(probabilities) != len(settings.CLASS_LABELS):
            raise ModelOutputError(
                f"Le modèle renvoie {len(probabilities)} probabilités mais "
                f"{len(settings.CLASS_LABELS)} classes sont configurées dans "
                "settings.CLASS_LABELS. Mettez à jour cette liste pour qu'elle "
                "corresponde exactement à l'ordre des classes utilisé à l'entraînement."
            )

        # Étape 3 — Décision : on retient la classe de plus haute probabilité
        # (argmax), conformément à la consigne ("faire ressortir la plus haute
        # probabilité de failure, 'none' étant une classe comme une autre).
        best_index = int(np.argmax(probabilities))
        predicted_class = settings.CLASS_LABELS[best_index]

        # Étape 4 — Mise en forme de la réponse métier.
        return {
            "predicted_class": predicted_class,
            "is_failure": predicted_class != settings.NO_FAILURE_LABEL,
            "probability": float(probabilities[best_index]),
            "class_probabilities": dict(
                zip(settings.CLASS_LABELS, (float(p) for p in probabilities))
            ),
            "model_version": settings.API_VERSION,
        }


model_service = FailureTypeModelService()
