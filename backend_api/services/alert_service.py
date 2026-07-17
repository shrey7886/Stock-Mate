from __future__ import annotations

import logging

from backend_api.database.token_store import list_all_active_alerts, mark_alert_triggered
from backend_api.services.market_data_service import market_data_service

logger = logging.getLogger(__name__)


def check_alerts() -> None:
    """Fetch current prices for every distinct symbol with an active alert and
    trigger any alert whose target condition is met. Never raises — a failed
    price fetch for one symbol is skipped and the rest of the batch continues."""
    try:
        active_alerts = list_all_active_alerts()
    except Exception as exc:
        logger.warning("failed to load active price alerts: %s", exc)
        return

    if not active_alerts:
        return

    symbols = {alert["symbol"] for alert in active_alerts}
    prices: dict[str, float] = {}
    for symbol in symbols:
        try:
            price = market_data_service.get_current_price(symbol)
        except Exception as exc:
            logger.warning("price check failed for %s: %s", symbol, exc)
            price = None
        if price is not None:
            prices[symbol] = price

    for alert in active_alerts:
        price = prices.get(alert["symbol"])
        if price is None:
            continue
        try:
            hit = (
                price >= alert["target_price"]
                if alert["direction"] == "above"
                else price <= alert["target_price"]
            )
            if hit:
                mark_alert_triggered(alert_id=alert["id"])
        except Exception as exc:
            logger.warning("failed to evaluate/trigger alert %s: %s", alert.get("id"), exc)
