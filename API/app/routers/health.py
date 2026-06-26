from datetime import datetime, timezone

from fastapi import APIRouter

from ..config import settings
from ..model_service import model_service
from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Vérifie que le service est actif et indique si le modèle est chargé."""
    return HealthResponse(
        status="ok" if model_service.is_loaded else "degraded",
        model_loaded=model_service.is_loaded,
        model_path=settings.MODEL_PATH,
        timestamp=datetime.now(timezone.utc),
    )
