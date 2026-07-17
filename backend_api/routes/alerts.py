from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend_api.core.security import get_current_user
from backend_api.database.token_store import (
    create_price_alert,
    delete_price_alert,
    dismiss_alert,
    list_price_alerts,
    reset_alert,
)
from backend_api.models.schemas import PriceAlertCreate, PriceAlertResponse

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.post("")
def create_alert(
    payload: PriceAlertCreate,
    current_user: dict = Depends(get_current_user),
) -> PriceAlertResponse:
    user_id = str(current_user.get("sub"))
    alert_id = create_price_alert(
        user_id=user_id,
        symbol=payload.symbol.upper(),
        target_price=payload.target_price,
        direction=payload.direction,
    )
    alert = next((a for a in list_price_alerts(user_id=user_id) if a["id"] == alert_id), None)
    if alert is None:
        raise HTTPException(status_code=500, detail="Failed to create alert.")
    return PriceAlertResponse(**alert)


@router.get("")
def get_alerts(current_user: dict = Depends(get_current_user)) -> list[PriceAlertResponse]:
    user_id = str(current_user.get("sub"))
    return [PriceAlertResponse(**a) for a in list_price_alerts(user_id=user_id)]


@router.delete("/{alert_id}")
def remove_alert(alert_id: int, current_user: dict = Depends(get_current_user)) -> dict:
    user_id = str(current_user.get("sub"))
    if not delete_price_alert(user_id=user_id, alert_id=alert_id):
        raise HTTPException(status_code=404, detail="Alert not found.")
    return {"success": True}


@router.post("/{alert_id}/dismiss")
def dismiss_alert_route(alert_id: int, current_user: dict = Depends(get_current_user)) -> dict:
    user_id = str(current_user.get("sub"))
    if not dismiss_alert(user_id=user_id, alert_id=alert_id):
        raise HTTPException(status_code=404, detail="Alert not found.")
    return {"success": True}


@router.post("/{alert_id}/reset")
def reset_alert_route(alert_id: int, current_user: dict = Depends(get_current_user)) -> dict:
    user_id = str(current_user.get("sub"))
    if not reset_alert(user_id=user_id, alert_id=alert_id):
        raise HTTPException(status_code=404, detail="Alert not found.")
    return {"success": True}
