from fastapi import APIRouter

from ..config import settings
from ..model_service import model_service
from ..schemas import ModelInfoResponse

router = APIRouter(tags=["model-info"])


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Fournit des informations sur le modèle de classification de type de défaillance actuellement chargé."""
    info = model_service.get_info()
    return ModelInfoResponse(
        model_name="predictive-maintenance-failure-type-classifier",
        model_type=info["model_type"],
        task=settings.TASK,
        target_variable=settings.TARGET_VARIABLE,
        numeric_features=settings.NUMERIC_FEATURES,
        categorical_features=settings.CATEGORICAL_FEATURES,
        categorical_values={
            "machine_type": settings.MACHINE_TYPES,
            "operating_mode": settings.OPERATING_MODES,
        },
        n_features_after_preprocessing=info["n_features_after_preprocessing"],
        class_labels=settings.CLASS_LABELS,
        no_failure_label=settings.NO_FAILURE_LABEL,
        version=settings.API_VERSION,
        loaded_at=model_service.loaded_at,
    )
