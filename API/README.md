# API — Maintenance Prédictive (FastAPI + scikit-learn/XGBoost)

Service d'inférence REST pour la tâche **classification multi-classe :
identification du type de défaillance** (`failure_type`). Le modèle candidat
final est un **pipeline scikit-learn** (`ColumnTransformer` + `XGBClassifier`)
sérialisé via **joblib**, qui embarque tout le preprocessing (imputation,
scaling, one-hot encoding) : l'API lui transmet directement les features
brutes, sans dupliquer la transformation côté serveur (évite tout risque de
divergence train/serve). Le pipeline renvoie un vecteur de probabilités (une
par type de panne, y compris la classe `none` = pas de panne) ; l'API renvoie
la classe de plus haute probabilité ainsi que le détail des probabilités par
classe. L'API ne contient aucune logique d'entraînement, uniquement
l'inférence (séparation modèle / interface, cf. cahier des charges).

Classes (ordre exact = `clf.classes_` = `[0, 1, 2, 3, 4]`, cf. `app/config.py::CLASS_LABELS`) :

| Index | Classe |
|---|---|
| 0 | `bearing` |
| 1 | `electrical` |
| 2 | `hydraulic` |
| 3 | `motor_overheat` |
| 4 | `none` (pas de défaillance) |

## Structure

```
API/
├── app/
│   ├── main.py            # App FastAPI, CORS, exception handlers
│   ├── config.py          # Settings (chemin modèle, classes, features attendues)
│   ├── schemas.py          # Modèles Pydantic (requêtes/réponses)
│   ├── model_service.py    # Chargement du pipeline (joblib) + inférence
│   ├── exceptions.py       # Exceptions métier (modèle non chargé, sortie incohérente)
│   └── routers/
│       ├── health.py       # GET /health
│       ├── predict.py      # POST /predict
│       └── model_info.py   # GET /model-info
├── models/
│   └── XGBoost.pkl          # Pipeline scikit-learn (preprocessing + XGBClassifier)
├── tests/                  # Tests pytest indépendants du dashboard
├── requirements.txt
└── README.md
```

## Installation

```bash
cd API
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

⚠️ `xgboost` est **pinné à la version 2.1.4** dans `requirements.txt` : le
pickle fourni a été entraîné avec une version antérieure à 3.x, et
`xgboost>=3` échoue à désérialiser le `Booster` (`input stream corrupted`).
Si vous ré-entraînez/ré-exportez le modèle avec une version plus récente
d'xgboost, vous pourrez lever ce pin.

## Déposer le modèle

Le pipeline entraîné (preprocessing + `XGBClassifier`) doit être sérialisé via
`joblib.dump(pipeline, "XGBoost.pkl")` et placé ici :

```
API/models/XGBoost.pkl
```

Le chemin est configurable via la variable d'environnement `MODEL_PATH`.

Le pipeline doit exposer deux étapes nommées `prep` (un `ColumnTransformer`)
et `clf` (le classifieur), et accepter en entrée un DataFrame avec exactement
les 9 colonnes brutes suivantes (`app/config.py::RAW_FEATURES`) :

| Colonne | Type | Détail |
|---|---|---|
| `vibration_rms` | numérique | imputation médiane + `StandardScaler` |
| `temperature_motor` | numérique | idem |
| `current_phase_avg` | numérique | idem |
| `pressure_level` | numérique | idem |
| `rpm` | numérique | idem |
| `hours_since_maintenance` | numérique | idem |
| `ambient_temp` | numérique | idem |
| `machine_type` | catégorielle | one-hot — valeurs : `CNC`, `Compressor`, `Pump`, `Robotic Arm` |
| `operating_mode` | catégorielle | one-hot — valeurs : `idle`, `normal`, `peak` |

⚠️ Si le nombre de classes renvoyées par `predict_proba` ne correspond pas à
`settings.CLASS_LABELS` (5 classes), `/predict` renvoie une erreur `500
model_output_mismatch` explicite — mettez à jour `CLASS_LABELS` dans
`app/config.py` si l'ordre/le nombre de classes change après un nouvel
entraînement.

Si aucun modèle n'est présent, l'API démarre tout de même en **mode dégradé** :
`/health` répond avec `model_loaded: false` et `/predict` répond `503`. Cela
permet de valider toute la structure de l'API avant même d'avoir un modèle.

## Lancer l'API

```bash
uvicorn app.main:app --reload --app-dir API
# ou, depuis le dossier API/ :
uvicorn app.main:app --reload
```

Documentation interactive (Swagger) : http://127.0.0.1:8000/docs

## Endpoints

### `GET /health`
Vérifie que le service tourne et que le modèle est chargé.

```bash
curl http://127.0.0.1:8000/health
```
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "models/XGBoost.pkl",
  "timestamp": "2026-06-26T10:00:00Z"
}
```

### `POST /predict`
Reçoit les données capteurs brutes et renvoie le type de défaillance le plus probable.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "vibration_rms": 2.35,
        "temperature_motor": 68.4,
        "current_phase_avg": 9.1,
        "pressure_level": 58.2,
        "rpm": 1450,
        "hours_since_maintenance": 120,
        "ambient_temp": 13.5,
        "machine_type": "Pump",
        "operating_mode": "normal"
      }'
```
```json
{
  "predicted_class": "bearing",
  "is_failure": true,
  "probability": 0.74,
  "class_probabilities": {
    "bearing": 0.74,
    "electrical": 0.10,
    "hydraulic": 0.05,
    "motor_overheat": 0.06,
    "none": 0.05
  },
  "model_version": "0.1.0"
}
```

`predicted_class` est la classe de plus haute probabilité (argmax sur le
vecteur renvoyé par `predict_proba`). Si c'est `none`, `is_failure` vaut
`false` : aucune défaillance n'est prédite.

Codes d'erreur :
- `422` : champ manquant, type incorrect, ou valeur hors des catégories autorisées (`machine_type`/`operating_mode`) — validation Pydantic automatique.
- `503` : modèle non chargé (`model_not_loaded`).
- `500` : le modèle renvoie un nombre de classes différent de `CLASS_LABELS` (`model_output_mismatch`).

### `GET /model-info`
Informations sur le modèle chargé (type de pipeline, features attendues, catégories valides, classes).

```bash
curl http://127.0.0.1:8000/model-info
```

## Tests

```bash
pytest API/tests -v
```

Les tests de `/predict` mockent `model_service` (pas besoin d'un vrai fichier
`.pkl` pour valider la structure de l'API), et `test_model_service.py` simule
un pipeline scikit-learn (`predict_proba`) pour valider la logique d'argmax et
de gestion d'erreurs — conformément à la recommandation du cahier des charges
de tester l'API indépendamment du dashboard/modèle.

## Architecture cible

```
Dashboard (Streamlit) ──HTTP──> API (FastAPI) ──joblib──> XGBoost.pkl (pipeline sklearn)
```

Le dashboard doit appeler cette API plutôt que de charger le modèle
directement, afin de reproduire une architecture Front / API / Modèle réaliste.
