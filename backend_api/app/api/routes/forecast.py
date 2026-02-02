"""
Forecast/prediction endpoints (STUB implementation).
"""
from fastapi import APIRouter, HTTPException, status
from datetime import datetime, timedelta

from app.api.schemas.request import ForecastRequest
from app.api.schemas.response import ForecastResponse, PredictionPoint, ConfidenceIntervals

router = APIRouter(prefix="/forecast", tags=["Forecast"])


@router.post("", response_model=ForecastResponse)
async def get_forecast(request: ForecastRequest):
    """
    Get stock price forecast for a given symbol and horizon.
    
    **STUB IMPLEMENTATION** - Returns dummy data.
    Real ML model integration happens in Phase 5.
    """
    # Validate symbol (basic check)
    if not request.symbol.isalpha() or len(request.symbol) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid stock symbol: {request.symbol}"
        )
    
    # Generate dummy predictions
    base_price = 180.0  # Dummy base price
    predictions = []
    p10_values = []
    p50_values = []
    p90_values = []
    
    for day in range(1, request.horizon + 1):
        # Simulate price movement (random walk)
        price = base_price + (day * 0.5)
        date_str = (datetime.utcnow() + timedelta(days=day)).strftime("%Y-%m-%d")
        
        predictions.append(
            PredictionPoint(
                day=day,
                date=date_str,
                price=round(price, 2)
            )
        )
        
        # Generate confidence intervals
        p10_values.append(round(price - 2.5, 2))  # Pessimistic
        p50_values.append(round(price, 2))        # Median
        p90_values.append(round(price + 2.5, 2))  # Optimistic
    
    return ForecastResponse(
        symbol=request.symbol.upper(),
        forecast_date=datetime.utcnow(),
        predictions=predictions,
        confidence=ConfidenceIntervals(
            p10=p10_values,
            p50=p50_values,
            p90=p90_values
        ),
        model_version="stub_v0.1"
    )
