def test_model_info_when_not_loaded(client, unloaded_model):
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_type"] is None
    assert body["n_features_after_preprocessing"] is None
    assert body["task"] == "classification_multiclasse"
    assert body["target_variable"] == "failure_type"
    assert body["class_labels"] == [
        "bearing",
        "electrical",
        "hydraulic",
        "motor_overheat",
        "none",
    ]
    assert body["no_failure_label"] == "none"
    assert body["numeric_features"] == [
        "vibration_rms",
        "temperature_motor",
        "current_phase_avg",
        "pressure_level",
        "rpm",
        "hours_since_maintenance",
        "ambient_temp",
    ]
    assert body["categorical_features"] == ["machine_type", "operating_mode"]
    assert body["categorical_values"]["machine_type"] == ["CNC", "Compressor", "Pump", "Robotic Arm"]
    assert body["categorical_values"]["operating_mode"] == ["idle", "normal", "peak"]
