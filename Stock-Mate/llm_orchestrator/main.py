"""
LLM Orchestrator Main Module
LangChain-based conversational AI for StockMate
Handles user queries and coordinates with backend services
"""

import os
import asyncio
from typing import Dict, Any, List, Optional
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
import httpx
import json

from tools import (
    get_portfolio_summary,
    get_stock_analysis,
    create_portfolio,
    add_stock_position,
    get_market_news,
    calculate_portfolio_metrics
)

class StockMateLLMOrchestrator:
    """Main LLM orchestrator for StockMate conversational AI"""
    
    def __init__(self, 
                 openai_api_key: str,
                 backend_api_url: str = "http://localhost:8000",
                 ml_service_url: str = "http://localhost:8001"):
        self.backend_api_url = backend_api_url
        self.ml_service_url = ml_service_url
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-3.5-turbo",
            temperature=0.1
        )
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Keep last 10 exchanges
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize tools
        self.tools = self._setup_tools()
        
        # Create agent
        self.agent = self._create_agent()
    
    def _setup_tools(self) -> List[Tool]:
        """Setup available tools for the agent"""
        tools = [
            Tool(
                name="get_portfolio_summary",
                description="Get a summary of user's portfolio including positions and performance",
                func=get_portfolio_summary
            ),
            Tool(
                name="get_stock_analysis",
                description="Get detailed analysis and predictions for a specific stock symbol",
                func=get_stock_analysis
            ),
            Tool(
                name="create_portfolio",
                description="Create a new portfolio with initial balance",
                func=create_portfolio
            ),
            Tool(
                name="add_stock_position",
                description="Add a stock position to an existing portfolio",
                func=add_stock_position
            ),
            Tool(
                name="get_market_news",
                description="Get recent market news and updates",
                func=get_market_news
            ),
            Tool(
                name="calculate_portfolio_metrics",
                description="Calculate portfolio performance metrics and risk analysis",
                func=calculate_portfolio_metrics
            )
        ]
        return tools
    
    def _create_agent(self) -> AgentExecutor:
        """Create the ReAct agent with tools"""
        prompt = PromptTemplate.from_template("""
You are StockMate, an AI financial advisor and portfolio management assistant. 
You help users manage their investment portfolios, analyze stocks, and make informed financial decisions.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Previous conversation history:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}
""")
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        
        return agent_executor
    
    async def process_query(self, query: str, user_id: str = "default") -> str:
        """
        Process user query and return response
        
        Args:
            query: User's question or request
            user_id: User identifier for context
            
        Returns:
            AI response string
        """
        try:
            # Add user context to query
            contextual_query = f"User ID: {user_id}\nQuery: {query}"
            
            # Process with agent
            response = await self.agent.ainvoke({
                "input": contextual_query
            })
            
            return response["output"]
            
        except Exception as e:
            return f"I apologize, but I encountered an error processing your request: {str(e)}. Please try again or rephrase your question."
    
    async def get_portfolio_insights(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive portfolio insights for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with portfolio insights
        """
        try:
            # Get portfolio summary
            portfolio_summary = await get_portfolio_summary(user_id)
            
            # Get market analysis
            market_news = await get_market_news()
            
            # Calculate metrics
            metrics = await calculate_portfolio_metrics(user_id)
            
            return {
                "portfolio": portfolio_summary,
                "market_news": market_news,
                "metrics": metrics,
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            return {"error": f"Failed to get portfolio insights: {str(e)}"}
    
    def reset_conversation(self):
        """Reset conversation memory"""
        self.memory.clear()

# Global orchestrator instance
orchestrator = None

def initialize_orchestrator(openai_api_key: str, 
                          backend_api_url: str = "http://localhost:8000",
                          ml_service_url: str = "http://localhost:8001"):
    """Initialize the global orchestrator instance"""
    global orchestrator
    orchestrator = StockMateLLMOrchestrator(
        openai_api_key=openai_api_key,
        backend_api_url=backend_api_url,
        ml_service_url=ml_service_url
    )

async def process_user_query(query: str, user_id: str = "default") -> str:
    """Process a user query using the global orchestrator"""
    if orchestrator is None:
        raise RuntimeError("Orchestrator not initialized. Call initialize_orchestrator() first.")
    
    return await orchestrator.process_query(query, user_id)

# Example usage
if __name__ == "__main__":
    # Initialize with API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    initialize_orchestrator(api_key)
    
    # Example conversation
    async def example_conversation():
        queries = [
            "Hello, I'm new to investing. Can you help me create a portfolio?",
            "What's the current performance of my portfolio?",
            "Should I buy Apple stock?",
            "What are the latest market trends?"
        ]
        
        for query in queries:
            print(f"\nUser: {query}")
            response = await process_user_query(query)
            print(f"StockMate: {response}")
    
    # Run example
    asyncio.run(example_conversation())
