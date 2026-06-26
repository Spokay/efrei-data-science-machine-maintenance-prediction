import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings:
    """Configuration centralisée de l'API, surchargeable via variables d'environnement."""

    API_TITLE: str = "Predictive Maintenance API"
    API_DESCRIPTION: str = (
        "API d'inférence pour la maintenance prédictive industrielle. "
        "Tâche : identification du type de défaillance (classification multi-classe)."
    )
    API_VERSION: str = "0.1.0"

    # Modèle candidat final : pipeline scikit-learn (ColumnTransformer + XGBClassifier)
    # sérialisé via joblib. Le pipeline embarque tout le preprocessing
    # (imputation, scaling, one-hot encoding).
    MODEL_PATH: str = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "XGBoost.pkl"))

    TASK: str = "classification_multiclasse"
    TARGET_VARIABLE: str = "failure_type"

    # Classe représentant "pas de défaillance" parmi les classes prédites.
    NO_FAILURE_LABEL: str = "none"

    # Ordre des classes en sortie du modèle (= clf.classes_ = [0, 1, 2, 3, 4]).
    # index 0 → bearing, 1 → electrical, 2 → hydraulic, 3 → motor_overheat, 4 → none
    CLASS_LABELS: list[str] = [
        "bearing",
        "electrical",
        "hydraulic",
        "motor_overheat",
        "none",
    ]

    # Features brutes attendues par le pipeline (cf. ColumnTransformer "prep"),
    # AVANT preprocessing. L'API les transmet telles quelles ; le pipeline se
    # charge de l'imputation/scaling/encodage.
    NUMERIC_FEATURES: list[str] = [
        "vibration_rms",
        "temperature_motor",
        "current_phase_avg",
        "pressure_level",
        "rpm",
        "hours_since_maintenance",
        "ambient_temp",
    ]
    CATEGORICAL_FEATURES: list[str] = ["machine_type", "operating_mode"]
    RAW_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    # Catégories valides (cf. OneHotEncoder.categories_ entraîné).
    MACHINE_TYPES: list[str] = ["CNC", "Compressor", "Pump", "Robotic Arm"]
    OPERATING_MODES: list[str] = ["idle", "normal", "peak"]

    ALLOWED_ORIGINS: list[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")


settings = Settings()
