from fastapi import APIRouter

from ..model_service import model_service
from ..schemas import PredictResponse, SensorData

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(data: SensorData) -> PredictResponse:
    """Reçoit des données capteurs brutes et renvoie le type de défaillance le plus probable.

    `data` est déjà validée par Pydantic (types, bornes, catégories autorisées) ;
    le DataFrame de features brutes est construit et transformé par
    `model_service.predict` (preprocessing + inférence + argmax). Une erreur
    503 est levée automatiquement si le modèle n'est pas chargé (gérée par
    l'exception handler global, voir main.py).
    """
    return model_service.predict(data)
