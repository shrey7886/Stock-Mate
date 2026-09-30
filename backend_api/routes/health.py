from fastapi import APIRouter

from backend_api.core.config import settings
from backend_api.models.schemas import HealthResponse

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    checks = {
        "zerodha": {
            "configured": bool(settings.zerodha_api_key and settings.zerodha_api_secret),
            "provider": "zerodha",
            "status": "ready" if settings.zerodha_api_key and settings.zerodha_api_secret else "missing_credentials",
        },
        "upstox": {
            "configured": bool(settings.upstox_api_key and settings.upstox_api_secret),
            "provider": "upstox",
            "status": "ready" if settings.upstox_api_key and settings.upstox_api_secret else "missing_credentials",
        },
    }
    is_healthy = all(check["configured"] for check in checks.values())
    return HealthResponse(
        status="ok" if is_healthy else "degraded",
        checks=checks,
        message=(
            "Broker credentials are configured and the broker sign-in flow is ready."
            if is_healthy
            else "One or more broker credentials are missing; broker login flows will fail until configured."
        ),
    )
