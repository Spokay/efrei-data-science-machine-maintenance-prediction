# Machine Maintenance Prediction

Multi-class classification of industrial machine failure types from sensor data. Given real-time measurements (vibration, temperature, pressure, etc.), the system predicts which type of failure is occurring — or that none is.

**Classes:** `bearing` · `electrical` · `hydraulic` · `motor_overheat` · `none`

## Architecture

```
training/          → model training (scikit-learn, XGBoost, TensorFlow)
API/               → FastAPI inference service
```

The trained XGBoost pipeline (`training/models/XGBoost.pkl`) is copied to `API/models/XGBoost.pkl` to be served. The pipeline embeds all preprocessing (imputation, scaling, one-hot encoding), so the API passes raw sensor features directly without any transformation logic.

## Training

```bash
cd training
pip install -r requirements.txt
python src/train.py
```

Trains 4 models (Logistic Regression baseline, Random Forest, XGBoost, TensorFlow MLP). Outputs saved to:

| Path | Content |
|---|---|
| `training/models/` | Serialized models (`.pkl`, `.keras`, `.onnx`) + `label_encoder.pkl` |
| `training/figures/` | Confusion matrices, ROC curves, feature importance, SHAP |
| `training/metrics/` | `comparison_table.csv`, `cv_results.csv` |

Primary metric: **macro recall** (imbalanced dataset — missing a failure matters more than a false alarm).

## API

```bash
cd API
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Swagger UI: `http://127.0.0.1:8000/docs`

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service + model load status |
| POST | `/predict` | Predict failure type from sensor data |
| GET | `/model-info` | Pipeline metadata, expected features, valid categories |

### Predict request

```json
{
  "vibration_rms": 2.35,
  "temperature_motor": 68.4,
  "current_phase_avg": 9.1,
  "pressure_level": 58.2,
  "rpm": 1450,
  "hours_since_maintenance": 120,
  "ambient_temp": 13.5,
  "machine_type": "Pump",
  "operating_mode": "normal"
}
```

`machine_type` ∈ `CNC | Compressor | Pump | Robotic Arm`  
`operating_mode` ∈ `idle | normal | peak`

### Tests

```bash
pytest API/tests -v
```

## Input features

| Feature | Type |
|---|---|
| `vibration_rms` | numeric |
| `temperature_motor` | numeric |
| `current_phase_avg` | numeric |
| `pressure_level` | numeric |
| `rpm` | numeric |
| `hours_since_maintenance` | numeric |
| `ambient_temp` | numeric |
| `machine_type` | categorical |
| `operating_mode` | categorical |

## Notes

- `xgboost` is pinned to `==2.1.4` in `API/requirements.txt` — the serialized pipeline is incompatible with xgboost ≥ 3. Lift the pin only after re-training and re-serializing.
- `MODEL_PATH` env var overrides the default model location in the API.
- If `API/models/XGBoost.pkl` is missing, the API starts in degraded mode: `/health` returns `model_loaded: false`, `/predict` returns 503.