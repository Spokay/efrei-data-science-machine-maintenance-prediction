from pathlib import Path

ROOT = Path(__file__).parent.parent

DATA_PATH   = ROOT / "data" / "industrial_machine_maintenance.csv"
MODELS_DIR      = ROOT / "models"
API_MODELS_DIR  = ROOT.parent / "API" / "models"
FIGURES_DIR = ROOT / "figures"
METRICS_DIR = ROOT / "metrics"

RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5

TARGET = "failure_type"

FEATURES_NUM = [
    "vibration_rms",
    "temperature_motor",
    "current_phase_avg",
    "pressure_level",
    "rpm",
    "hours_since_maintenance",
    "ambient_temp",
]

FEATURES_CAT = ["machine_type", "operating_mode"]

COLS_TO_DROP = [
    "timestamp",
    "machine_id",
    "failure_within_24h",
    "rul_hours",
    "estimated_repair_cost",
]

CLASS_ORDER = ["bearing", "electrical", "hydraulic", "motor_overheat", "none"]
