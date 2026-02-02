"""
Response schemas for API endpoints.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from uuid import UUID


class PredictionPoint(BaseModel):
    """Single prediction point."""
    day: int
    date: str
    price: float


class ConfidenceIntervals(BaseModel):
    """Confidence intervals for predictions."""
    p10: List[float] = Field(..., description="10th percentile (pessimistic)")
    p50: List[float] = Field(..., description="50th percentile (median)")
    p90: List[float] = Field(..., description="90th percentile (optimistic)")


class ForecastResponse(BaseModel):
    """Response schema for forecast endpoint."""
    symbol: str
    forecast_date: datetime
    predictions: List[PredictionPoint]
    confidence: ConfidenceIntervals
    model_version: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "forecast_date": "2026-01-15T10:30:00Z",
                "predictions": [
                    {"day": 1, "date": "2026-01-16", "price": 182.4},
                    {"day": 2, "date": "2026-01-17", "price": 184.1}
                ],
                "confidence": {
                    "p10": [180.1, 181.5],
                    "p50": [182.4, 184.1],
                    "p90": [184.9, 186.8]
                },
                "model_version": "tft_v1.0"
            }
        }


class UserResponse(BaseModel):
    """Response schema for user data."""
    id: UUID
    email: str
    full_name: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Response schema for authentication token."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Token expiration in seconds")


class PortfolioResponse(BaseModel):
    """Response schema for portfolio data."""
    id: UUID
    name: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class HoldingResponse(BaseModel):
    """Response schema for holding data."""
    id: UUID
    symbol: str
    quantity: float
    avg_price: float
    added_at: datetime
    
    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    timestamp: datetime
    version: str
    services: dict
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2026-01-15T10:30:00Z",
                "version": "1.0.0",
                "services": {
                    "database": "connected",
                    "ml_model": "loaded"
                }
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Resource not found"
            }
        }
