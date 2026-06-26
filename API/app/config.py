from pathlib import Path
from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    API_TITLE: str = "Predictive Maintenance API"
    API_DESCRIPTION: str = (
        "API d'inférence pour la maintenance prédictive industrielle. "
        "Tâche : identification du type de défaillance (classification multi-classe)."
    )
    API_VERSION: str = "0.1.0"

    MODEL_PATH: str = str(BASE_DIR / "models" / "XGBoost.pkl")

    TASK: str = "classification_multiclasse"
    TARGET_VARIABLE: str = "failure_type"
    NO_FAILURE_LABEL: str = "none"

    CLASS_LABELS: list[str] = [
        "bearing",
        "electrical",
        "hydraulic",
        "motor_overheat",
        "none",
    ]

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

    MACHINE_TYPES: list[str] = ["CNC", "Compressor", "Pump", "Robotic Arm"]
    OPERATING_MODES: list[str] = ["idle", "normal", "peak"]

    # Chaîne CSV : "http://a.com,http://b.com" ou "*". Splitée dans main.py.
    ALLOWED_ORIGINS: str = "*"

    @computed_field
    @property
    def RAW_FEATURES(self) -> list[str]:
        return self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES


settings = Settings()
