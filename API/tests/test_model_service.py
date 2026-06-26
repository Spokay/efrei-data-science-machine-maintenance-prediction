import numpy as np
import pytest

from app.exceptions import ModelOutputError
from app.model_service import FailureTypeModelService
from app.schemas import SensorData

VALID_FEATURES = SensorData(
    vibration_rms=2.35,
    temperature_motor=68.4,
    current_phase_avg=9.1,
    pressure_level=58.2,
    rpm=1450.0,
    hours_since_maintenance=120.0,
    ambient_temp=13.5,
    machine_type="Pump",
    operating_mode="normal",
)


class _FakeClassifier:
    def __init__(self, n_features_in_):
        self.n_features_in_ = n_features_in_


class _FakePipeline:
    """Simule le pipeline scikit-learn (ColumnTransformer + XGBClassifier)
    sans dépendre d'un vrai fichier .pkl."""

    def __init__(self, proba_row, n_features_in_=14):
        self._proba_row = np.asarray(proba_row, dtype=np.float64)
        self.named_steps = {"prep": object(), "clf": _FakeClassifier(n_features_in_)}

    def predict_proba(self, df):
        assert list(df.columns) == [
            "vibration_rms",
            "temperature_motor",
            "current_phase_avg",
            "pressure_level",
            "rpm",
            "hours_since_maintenance",
            "ambient_temp",
            "machine_type",
            "operating_mode",
        ]
        return np.array([self._proba_row])


def _make_service(proba_row, n_features_in_=14):
    service = FailureTypeModelService(model_path="unused.pkl")
    service.pipeline = _FakePipeline(proba_row, n_features_in_)
    return service


def test_predict_picks_highest_probability_class():
    # ordre CLASS_LABELS: bearing, electrical, hydraulic, motor_overheat, none
    service = _make_service([0.74, 0.10, 0.05, 0.06, 0.05])

    result = service.predict(VALID_FEATURES)

    assert result.predicted_class == "bearing"
    assert result.is_failure is True
    assert result.probability == pytest.approx(0.74, rel=1e-3)
    assert result.class_probabilities["bearing"] == pytest.approx(0.74, rel=1e-3)


def test_predict_returns_none_when_no_failure_class_dominant():
    service = _make_service([0.05, 0.05, 0.05, 0.05, 0.80])

    result = service.predict(VALID_FEATURES)

    assert result.predicted_class == "none"
    assert result.is_failure is False


def test_predict_raises_on_class_count_mismatch():
    service = _make_service([0.5, 0.5])  # 2 classes au lieu de 5

    with pytest.raises(ModelOutputError):
        service.predict(VALID_FEATURES)


def test_get_info_reports_pipeline_metadata():
    service = _make_service([0.2, 0.2, 0.2, 0.2, 0.2], n_features_in_=14)

    info = service.get_info()

    assert info.n_features_after_preprocessing == 14
    assert "prep" in info.model_type
    assert "clf" in info.model_type
