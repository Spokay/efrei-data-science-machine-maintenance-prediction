from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class SensorData(BaseModel):
    """Données capteur brutes d'une machine à un instant donné.

    Ces 9 champs correspondent exactement aux colonnes attendues par le
    pipeline de preprocessing (ColumnTransformer) du modèle entraîné. Le
    pipeline se charge lui-même de l'imputation, du scaling et de
    l'encodage : l'API transmet les valeurs brutes sans transformation.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "vibration_rms": 2.35,
                "temperature_motor": 68.4,
                "current_phase_avg": 9.1,
                "pressure_level": 58.2,
                "rpm": 1450.0,
                "hours_since_maintenance": 120.0,
                "ambient_temp": 13.5,
                "machine_type": "Pump",
                "operating_mode": "normal",
            }
        },
    )

    # --- Features numériques : imputées (médiane) puis standardisées (StandardScaler)
    # par l'étape "prep" du pipeline (cf. app/config.py::NUMERIC_FEATURES).
    vibration_rms: float = Field(..., ge=0, description="Vibration RMS du capteur")
    temperature_motor: float = Field(..., description="Température du moteur (°C)")
    current_phase_avg: float = Field(..., ge=0, description="Courant de phase moyen (A)")
    pressure_level: float = Field(..., ge=0, description="Niveau de pression")
    rpm: float = Field(..., ge=0, description="Vitesse de rotation (tours/minute)")
    hours_since_maintenance: float = Field(..., ge=0, description="Heures depuis la dernière maintenance")
    ambient_temp: float = Field(..., description="Température ambiante (°C)")

    # --- Features catégorielles : encodées en one-hot par le pipeline.
    # Les valeurs autorisées (Literal) reprennent exactement les catégories
    # vues à l'entraînement (OneHotEncoder.categories_) : toute autre valeur
    # est rejetée ici en 422 plutôt que silencieusement ignorée par l'encodeur
    # (qui est configuré en handle_unknown="ignore" côté modèle).
    machine_type: Literal["CNC", "Compressor", "Pump", "Robotic Arm"] = Field(
        ..., description="Type de machine"
    )
    operating_mode: Literal["idle", "normal", "peak"] = Field(..., description="Mode opératoire")


class PredictResponse(BaseModel):
    predicted_class: str = Field(
        ..., description="Type de défaillance prédit (classe de plus haute probabilité, ex: 'bearing', 'none'...)"
    )
    is_failure: bool = Field(..., description="False si la classe prédite est la classe 'none' (pas de défaillance)")
    probability: float = Field(..., ge=0, le=1, description="Probabilité de la classe prédite")
    class_probabilities: dict[str, float] = Field(
        ..., description="Probabilité prédite pour chaque type de défaillance"
    )
    model_version: str = Field(..., description="Version du modèle ayant servi à la prédiction")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predicted_class": "bearing",
                "is_failure": True,
                "probability": 0.74,
                "class_probabilities": {
                    "bearing": 0.74,
                    "electrical": 0.10,
                    "hydraulic": 0.05,
                    "motor_overheat": 0.06,
                    "none": 0.05,
                },
                "model_version": "0.1.0",
            }
        }
    )


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' si le service et le modèle sont opérationnels, 'degraded' sinon")
    model_loaded: bool
    model_path: str
    timestamp: datetime


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: Optional[str] = Field(None, description="Type du pipeline chargé (étapes scikit-learn)")
    task: str
    target_variable: str
    numeric_features: list[str]
    categorical_features: list[str]
    categorical_values: dict[str, list[str]] = Field(
        ..., description="Valeurs valides pour chaque feature catégorielle"
    )
    n_features_after_preprocessing: Optional[int] = Field(
        None, description="Nombre de features après preprocessing (one-hot inclus)"
    )
    class_labels: list[str] = Field(..., description="Ordre des classes attendu en sortie du modèle")
    no_failure_label: str
    version: str
    loaded_at: Optional[datetime] = None
