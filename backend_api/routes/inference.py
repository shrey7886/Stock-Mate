from __future__ import annotations

from fastapi import APIRouter, Depends

from backend_api.core.security import get_current_user

router = APIRouter(prefix="/api/inference", tags=["inference"])


@router.get("/status")
def inference_status(current_user: dict = Depends(get_current_user)) -> dict:
    return {
        "user_id": current_user.get("sub"),
        "model": "TemporalFusionTransformer",
        "horizon_days": 7,
        "status": "ready",
    }
