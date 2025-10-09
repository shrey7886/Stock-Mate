"""
LLM Orchestrator Tools
Functions that the LLM agent can call to interact with backend services
"""

import asyncio
import httpx
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Configuration
BACKEND_API_URL = "http://localhost:8000"
ML_SERVICE_URL = "http://localhost:8001"

async def make_api_request(method: str, url: str, data: Dict = None, params: Dict = None) -> Dict[str, Any]:
    """Make HTTP request to backend services"""
    async with httpx.AsyncClient() as client:
        try:
            if method.upper() == "GET":
                response = await client.get(url, params=params)
            elif method.upper() == "POST":
                response = await client.post(url, json=data)
            elif method.upper() == "PUT":
                response = await client.put(url, json=data)
            elif method.upper() == "DELETE":
                response = await client.delete(url)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            return {"error": f"API request failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

async def get_portfolio_summary(user_id: str) -> Dict[str, Any]:
    """
    Get portfolio summary for a user
    
    Args:
        user_id: User identifier
        
    Returns:
        Portfolio summary data
    """
    url = f"{BACKEND_API_URL}/api/v1/portfolio/"
    response = await make_api_request("GET", url)
    
    if "error" in response:
        return response
    
    # Mock portfolio data for demonstration
    if not response:  # Empty response
        return {
            "message": "No portfolios found. Would you like to create one?",
            "portfolios": [],
            "total_value": 0.0
        }
    
    # Calculate total value
    total_value = sum(portfolio.get("balance", 0) for portfolio in response)
    
    return {
        "portfolios": response,
        "total_value": total_value,
        "portfolio_count": len(response),
        "last_updated": datetime.now().isoformat()
    }

async def get_stock_analysis(symbol: str) -> Dict[str, Any]:
    """
    Get stock analysis and predictions
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        
    Returns:
        Stock analysis data
    """
    # This would call the ML service for predictions
    ml_url = f"{ML_SERVICE_URL}/api/v1/analyze/{symbol}"
    ml_response = await make_api_request("GET", ml_url)
    
    # Mock analysis data
    analysis = {
        "symbol": symbol.upper(),
        "current_price": 150.25,  # Mock data
        "prediction": {
            "next_30_days": {
                "predicted_price": 155.30,
                "confidence": 0.75,
                "trend": "bullish"
            }
        },
        "technical_indicators": {
            "rsi": 65.2,
            "macd": "positive",
            "bollinger_position": "upper_band"
        },
        "fundamental_analysis": {
            "pe_ratio": 25.4,
            "market_cap": "2.5T",
            "dividend_yield": 0.44
        },
        "recommendation": "HOLD",
        "risk_level": "Medium",
        "last_updated": datetime.now().isoformat()
    }
    
    return analysis

async def create_portfolio(name: str, description: str = "", initial_balance: float = 10000.0) -> Dict[str, Any]:
    """
    Create a new portfolio
    
    Args:
        name: Portfolio name
        description: Portfolio description
        initial_balance: Initial balance
        
    Returns:
        Created portfolio data
    """
    url = f"{BACKEND_API_URL}/api/v1/portfolio/"
    data = {
        "name": name,
        "description": description,
        "initial_balance": initial_balance
    }
    
    response = await make_api_request("POST", url, data=data)
    
    if "error" in response:
        return response
    
    return {
        "message": f"Portfolio '{name}' created successfully",
        "portfolio": response,
        "initial_balance": initial_balance
    }

async def add_stock_position(portfolio_id: int, symbol: str, quantity: int, purchase_price: float) -> Dict[str, Any]:
    """
    Add a stock position to a portfolio
    
    Args:
        portfolio_id: Portfolio ID
        symbol: Stock symbol
        quantity: Number of shares
        purchase_price: Purchase price per share
        
    Returns:
        Position addition result
    """
    url = f"{BACKEND_API_URL}/api/v1/portfolio/{portfolio_id}/positions"
    data = {
        "symbol": symbol.upper(),
        "quantity": quantity,
        "purchase_price": purchase_price
    }
    
    response = await make_api_request("POST", url, data=data)
    
    if "error" in response:
        return response
    
    return {
        "message": f"Added {quantity} shares of {symbol.upper()} to portfolio",
        "position": data,
        "total_value": quantity * purchase_price
    }

async def get_market_news(limit: int = 5) -> Dict[str, Any]:
    """
    Get recent market news
    
    Args:
        limit: Number of news items to return
        
    Returns:
        Market news data
    """
    # Mock news data - in production, this would call a news API
    news_items = [
        {
            "title": "Tech Stocks Rally on Strong Earnings Reports",
            "summary": "Major technology companies report better-than-expected Q4 earnings",
            "source": "Financial Times",
            "published_at": (datetime.now() - timedelta(hours=2)).isoformat(),
            "sentiment": "positive"
        },
        {
            "title": "Federal Reserve Hints at Interest Rate Stability",
            "summary": "Fed officials suggest rates may remain steady in the near term",
            "source": "Wall Street Journal",
            "published_at": (datetime.now() - timedelta(hours=4)).isoformat(),
            "sentiment": "neutral"
        },
        {
            "title": "Energy Sector Sees Volatility Amid Supply Concerns",
            "summary": "Oil prices fluctuate as supply chain issues persist",
            "source": "Reuters",
            "published_at": (datetime.now() - timedelta(hours=6)).isoformat(),
            "sentiment": "negative"
        }
    ]
    
    return {
        "news_items": news_items[:limit],
        "total_items": len(news_items),
        "last_updated": datetime.now().isoformat()
    }

async def calculate_portfolio_metrics(portfolio_id: int) -> Dict[str, Any]:
    """
    Calculate portfolio performance metrics
    
    Args:
        portfolio_id: Portfolio ID
        
    Returns:
        Portfolio metrics
    """
    # Mock metrics calculation
    metrics = {
        "total_return": 12.5,  # Percentage
        "annualized_return": 8.7,
        "volatility": 15.2,
        "sharpe_ratio": 1.8,
        "max_drawdown": -8.3,
        "beta": 1.1,
        "alpha": 2.3,
        "risk_score": "Medium",
        "diversification_score": 7.5,
        "performance_rating": "Good",
        "calculated_at": datetime.now().isoformat()
    }
    
    return metrics

async def get_stock_price(symbol: str) -> Dict[str, Any]:
    """
    Get current stock price
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Current price data
    """
    # Mock price data - in production, this would call a real API
    price_data = {
        "symbol": symbol.upper(),
        "current_price": 150.25,
        "change": 2.15,
        "change_percent": 1.45,
        "volume": 1250000,
        "market_cap": "2.5T",
        "last_updated": datetime.now().isoformat()
    }
    
    return price_data

async def get_sector_analysis(sector: str) -> Dict[str, Any]:
    """
    Get analysis for a specific sector
    
    Args:
        sector: Sector name (e.g., 'Technology', 'Healthcare')
        
    Returns:
        Sector analysis data
    """
    # Mock sector analysis
    analysis = {
        "sector": sector,
        "performance": {
            "ytd_return": 15.2,
            "monthly_return": 3.1,
            "volatility": 18.5
        },
        "top_performers": ["AAPL", "MSFT", "GOOGL"],
        "outlook": "Positive",
        "key_drivers": [
            "Strong earnings growth",
            "Innovation in AI and cloud computing",
            "Favorable regulatory environment"
        ],
        "risks": [
            "Market volatility",
            "Regulatory changes",
            "Competition"
        ],
        "last_updated": datetime.now().isoformat()
    }
    
    return analysis

# Tool functions for LangChain agent
def get_portfolio_summary_sync(user_id: str) -> str:
    """Synchronous wrapper for get_portfolio_summary"""
    return asyncio.run(get_portfolio_summary(user_id))

def get_stock_analysis_sync(symbol: str) -> str:
    """Synchronous wrapper for get_stock_analysis"""
    return asyncio.run(get_stock_analysis(symbol))

def create_portfolio_sync(name: str, description: str = "", initial_balance: float = 10000.0) -> str:
    """Synchronous wrapper for create_portfolio"""
    return asyncio.run(create_portfolio(name, description, initial_balance))

def add_stock_position_sync(portfolio_id: int, symbol: str, quantity: int, purchase_price: float) -> str:
    """Synchronous wrapper for add_stock_position"""
    return asyncio.run(add_stock_position(portfolio_id, symbol, quantity, purchase_price))

def get_market_news_sync(limit: int = 5) -> str:
    """Synchronous wrapper for get_market_news"""
    return asyncio.run(get_market_news(limit))

def calculate_portfolio_metrics_sync(portfolio_id: int) -> str:
    """Synchronous wrapper for calculate_portfolio_metrics"""
    return asyncio.run(calculate_portfolio_metrics(portfolio_id))
