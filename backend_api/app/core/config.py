"""
Application configuration using Pydantic Settings.
All settings can be overridden via environment variables.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Settings
    PROJECT_NAME: str = "StockMate API"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    
    # Database Settings
    DATABASE_URL: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/stockmate",
        description="PostgreSQL connection string"
    )
    TIMESCALE_ENABLED: bool = True
    
    # Security Settings
    JWT_SECRET_KEY: str = Field(
        default="CHANGE_THIS_SECRET_KEY_IN_PRODUCTION",
        description="Secret key for JWT token generation"
    )
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # CORS Settings
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
    ]
    
    # ML Model Settings
    ML_MODEL_PATH: str = "../ml_service/models/tft_final.ckpt"
    ML_SERVICE_URL: Optional[str] = None  # For microservice mode
    ML_MODE: str = "import"  # "import" or "service"
    
    # Redis/Caching (Phase 6)
    REDIS_URL: Optional[str] = None
    CACHE_PREDICTIONS_TTL: int = 900  # 15 minutes
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
