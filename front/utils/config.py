import os
from pathlib import Path

# front/utils/config.py -> front/utils -> front -> racine du dépôt
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "training" / "data" / "industrial_machine_maintenance.csv"
METRICS_DIR = PROJECT_ROOT / "training" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "training" / "figures"
TEST_VALUES_PATH = PROJECT_ROOT / "API" / "tests" / "test_values.json"

DEFAULT_API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

NUMERIC_FEATURES = [
    "vibration_rms",
    "temperature_motor",
    "current_phase_avg",
    "pressure_level",
    "rpm",
    "hours_since_maintenance",
    "ambient_temp",
]
CATEGORICAL_FEATURES = ["machine_type", "operating_mode"]
RAW_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Valeurs par défaut si l'API n'est pas joignable (reprises de app/config.py côté API).
DEFAULT_MACHINE_TYPES = ["CNC", "Compressor", "Pump", "Robotic Arm"]
DEFAULT_OPERATING_MODES = ["idle", "normal", "peak"]

# Recommandations métier affichées en fonction de la classe prédite par le modèle.
FAILURE_ADVICE = {
    "bearing": {
        "label": "Défaillance roulement",
        "action": "Inspecter les roulements et planifier un remplacement préventif dans les 24 à 48h.",
    },
    "electrical": {
        "label": "Défaillance électrique",
        "action": "Contrôler le circuit électrique et le courant de phase ; faire intervenir un électricien.",
    },
    "hydraulic": {
        "label": "Défaillance hydraulique",
        "action": "Vérifier le niveau et la pression du circuit hydraulique ; rechercher une fuite.",
    },
    "motor_overheat": {
        "label": "Surchauffe moteur",
        "action": "Réduire la charge ou le mode opératoire et vérifier le refroidissement moteur en urgence.",
    },
    "none": {
        "label": "Aucune défaillance",
        "action": "Aucune action requise — poursuivre le monitoring standard.",
    },
}
