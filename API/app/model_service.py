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

import joblib
import numpy as np
import pandas as pd

from .config import settings
from .exceptions import ModelNotLoadedError, ModelOutputError
from .schemas import ModelInfo, PredictResponse, SensorData

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

    def get_info(self) -> ModelInfo:
        if not self.is_loaded:
            return ModelInfo()

        clf = self.pipeline.named_steps.get("clf")
        return ModelInfo(
            model_type=f"{type(self.pipeline).__name__}({' -> '.join(self.pipeline.named_steps)})",
            n_features_after_preprocessing=getattr(clf, "n_features_in_", None),
        )

    def predict(self, data: SensorData) -> PredictResponse:
        if not self.is_loaded:
            raise ModelNotLoadedError(
                "Le modèle n'est pas chargé. Vérifiez MODEL_PATH et /health."
            )

        df = pd.DataFrame([data.model_dump()], columns=settings.RAW_FEATURES)

        # predict_proba() fait passer le DataFrame dans "prep" (preprocessing)
        # puis "clf" (XGBClassifier). Renvoie un vecteur de probabilités dans
        # l'ordre de clf.classes_ (= [0,1,2,3,4], aligné sur CLASS_LABELS).
        probabilities = self.pipeline.predict_proba(df)[0]

        # Garde-fou : nombre de classes incohérent entre modèle et config.
        if len(probabilities) != len(settings.CLASS_LABELS):
            raise ModelOutputError(
                f"Le modèle renvoie {len(probabilities)} probabilités mais "
                f"{len(settings.CLASS_LABELS)} classes sont configurées dans "
                "settings.CLASS_LABELS. Mettez à jour cette liste pour qu'elle "
                "corresponde exactement à l'ordre des classes utilisé à l'entraînement."
            )

        best_index = int(np.argmax(probabilities))
        predicted_class = settings.CLASS_LABELS[best_index]

        return PredictResponse(
            predicted_class=predicted_class,
            is_failure=predicted_class != settings.NO_FAILURE_LABEL,
            probability=float(probabilities[best_index]),
            class_probabilities=dict(
                zip(settings.CLASS_LABELS, (float(p) for p in probabilities))
            ),
            model_version=settings.API_VERSION,
        )


model_service = FailureTypeModelService()
