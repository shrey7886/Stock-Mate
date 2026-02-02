"""
Health check endpoint.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime

from app.api.schemas.response import HealthResponse
from app.db.session import get_db
from app.core.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint to verify API and services are running.
    """
    services = {
        "database": "unknown",
        "ml_model": "unknown"
    }
    
    # Check database connection
    try:
        db.execute("SELECT 1")
        services["database"] = "connected"
    except Exception:
        services["database"] = "disconnected"
    
    # Check ML model (stub for now)
    # TODO: Add actual model check in Phase 5
    services["ml_model"] = "not_loaded" if settings.ML_MODE == "import" else "service_mode"
    
    return HealthResponse(
        status="healthy" if services["database"] == "connected" else "degraded",
        timestamp=datetime.utcnow(),
        version=settings.VERSION,
        services=services
    )
