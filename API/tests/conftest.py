import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model_service import model_service


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def loaded_model(monkeypatch):
    """Simule un pipeline chargé, sans dépendre d'un vrai fichier .pkl."""
    monkeypatch.setattr(model_service, "pipeline", object())
    monkeypatch.setattr(
        model_service,
        "predict",
        lambda features: {
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
        },
    )
    yield model_service


@pytest.fixture
def unloaded_model(monkeypatch):
    monkeypatch.setattr(model_service, "pipeline", None)
    yield model_service
