"""
FastAPI Main Application
Entry point for the StockMate backend API service
Handles portfolio management and routing to other microservices
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import portfolio, users
from .database import engine, Base

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="StockMate Backend API",
    description="Portfolio Management & Microservice Routing",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["portfolio"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])

@app.get("/")
async def root():
    return {"message": "StockMate Backend API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "backend_api"}
