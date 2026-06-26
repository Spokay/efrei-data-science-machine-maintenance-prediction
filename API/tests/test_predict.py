VALID_PAYLOAD = {
    "vibration_rms": 2.35,
    "temperature_motor": 68.4,
    "current_phase_avg": 9.1,
    "pressure_level": 58.2,
    "rpm": 1450,
    "hours_since_maintenance": 120.0,
    "ambient_temp": 13.5,
    "machine_type": "Pump",
    "operating_mode": "normal",
}


def test_predict_returns_503_when_model_not_loaded(client, unloaded_model):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 503
    assert response.json()["error"] == "model_not_loaded"


def test_predict_returns_prediction_when_model_loaded(client, loaded_model):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == "bearing"
    assert body["is_failure"] is True
    assert 0.0 <= body["probability"] <= 1.0
    assert set(body["class_probabilities"]) == {
        "bearing",
        "electrical",
        "hydraulic",
        "motor_overheat",
        "none",
    }


def test_predict_missing_field_returns_422(client, loaded_model):
    payload = dict(VALID_PAYLOAD)
    del payload["vibration_rms"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_type_returns_422(client, loaded_model):
    payload = dict(VALID_PAYLOAD)
    payload["rpm"] = "not_a_number"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_value_returns_422(client, loaded_model):
    payload = dict(VALID_PAYLOAD)
    payload["vibration_rms"] = -5
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_machine_type_returns_422(client, loaded_model):
    payload = dict(VALID_PAYLOAD)
    payload["machine_type"] = "Toaster"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_operating_mode_returns_422(client, loaded_model):
    payload = dict(VALID_PAYLOAD)
    payload["operating_mode"] = "turbo"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
