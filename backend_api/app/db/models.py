"""
SQLAlchemy models for database tables.
"""
from sqlalchemy import Column, String, Numeric, Integer, DateTime, ForeignKey, Text, BigInteger, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.session import Base


class User(Base):
    """User account model."""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    portfolios = relationship("Portfolio", back_populates="user", cascade="all, delete-orphan")


class Portfolio(Base):
    """Portfolio model for user stock portfolios."""
    __tablename__ = "portfolios"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="portfolios")
    holdings = relationship("Holding", back_populates="portfolio", cascade="all, delete-orphan")


class Holding(Base):
    """Stock holding in a portfolio."""
    __tablename__ = "holdings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    symbol = Column(String, nullable=False)
    quantity = Column(Numeric, nullable=False)
    avg_price = Column(Numeric, nullable=False)
    added_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="holdings")


class StockPrice(Base):
    """
    Historical stock price data (TimescaleDB hypertable).
    Note: Hypertable creation happens via SQL migration, not SQLAlchemy.
    """
    __tablename__ = "stock_prices"
    
    timestamp = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    symbol = Column(String, primary_key=True, nullable=False)
    open = Column(Numeric)
    high = Column(Numeric)
    low = Column(Numeric)
    close = Column(Numeric, nullable=False)
    volume = Column(BigInteger)


class Prediction(Base):
    """
    Cached forecast predictions (TimescaleDB hypertable).
    Note: Hypertable creation happens via SQL migration, not SQLAlchemy.
    """
    __tablename__ = "predictions"
    
    created_at = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    symbol = Column(String, primary_key=True, nullable=False)
    horizon = Column(Integer, nullable=False)
    predictions = Column(JSONB, nullable=False)  # Array of {day, date, price}
    confidence = Column(JSONB, nullable=False)   # {p10: [], p50: [], p90: []}
    model_version = Column(String, nullable=True)
