"""
Database Configuration
SQLAlchemy setup for portfolio and user data
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Database URL (placeholder - replace with actual database)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./stockmate.db")

# Create engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

class Portfolio(Base):
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text)
    balance = Column(Float)
    user_id = Column(Integer, index=True)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

class StockPosition(Base):
    __tablename__ = "stock_positions"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, index=True)
    symbol = Column(String, index=True)
    quantity = Column(Integer)
    purchase_price = Column(Float)
    current_price = Column(Float)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
