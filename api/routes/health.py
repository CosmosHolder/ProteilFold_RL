"""
api/routes/health.py
GET /health — liveness + readiness probe.

Returns model load status, supported proteins, and version.
Used by HuggingFace Spaces health check and the frontend.

Author : ProteinFold-RL team
"""

from fastapi import APIRouter

from api.model_manager import get_model_manager, SUPPORTED_PROTEINS
from api.schemas import HealthResponse

router = APIRouter(tags=["Health"])

VERSION = "2.0.0"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and readiness probe",
    description=(
        "Returns 200 with `status='ok'` once the model checkpoint has been "
        "loaded. Returns 200 with `model_loaded=false` while loading is in "
        "progress (HuggingFace Spaces boot can take ~30s)."
    ),
)
def health_check() -> HealthResponse:
    """Check whether the model is loaded and the server is ready."""
    mm = get_model_manager()
    return HealthResponse(
        status="ok",
        model_loaded=mm.is_loaded,
        checkpoint_path=mm.checkpoint_path,
        supported_proteins=SUPPORTED_PROTEINS,
        version=VERSION,
    )