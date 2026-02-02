"""
Request schemas for API endpoints.
"""
from pydantic import BaseModel, Field, EmailStr
from typing import Optional


class ForecastRequest(BaseModel):
    """Request schema for forecast endpoint."""
    symbol: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")
    horizon: int = Field(7, ge=1, le=30, description="Number of days to forecast (1-30)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "horizon": 7
            }
        }


class UserRegisterRequest(BaseModel):
    """Request schema for user registration."""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="Password (minimum 8 characters)")
    full_name: Optional[str] = Field(None, description="User's full name")
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "SecurePass123!",
                "full_name": "John Doe"
            }
        }


class UserLoginRequest(BaseModel):
    """Request schema for user login."""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "SecurePass123!"
            }
        }


class PortfolioCreateRequest(BaseModel):
    """Request schema for creating a portfolio."""
    name: str = Field(..., min_length=1, max_length=100, description="Portfolio name")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "My Tech Portfolio"
            }
        }


class HoldingAddRequest(BaseModel):
    """Request schema for adding a stock to portfolio."""
    symbol: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")
    quantity: float = Field(..., gt=0, description="Number of shares")
    avg_price: float = Field(..., gt=0, description="Average purchase price")
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "quantity": 10.0,
                "avg_price": 180.50
            }
        }
