"""
Example of using REST API tools with LangChain and a mock API server.
This example demonstrates:
1. Running a mock API server for weather, news, and currency data
2. Using these APIs with LangChain tools
3. Providing random but realistic responses
"""

import os
import sys
import logging
import random
from pathlib import Path
import yaml
import json
from datetime import datetime
from typing import Any, Dict, List
from json import JSONEncoder
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading
import requests
from pydantic import BaseModel, Field
import time
import fnmatch

# Constants
API_PORT = 8866
API_BASE_URL = f"http://localhost:{API_PORT}"

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.tools.yaml_tool_source import YAMLToolSource
from src.services.tool_registry import ToolRegistry
from src.services.langchain_tool_loader import LangChainToolLoader
from src.config.config_loader import ConfigLoader
from langchain_ollama import OllamaLLM
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import HumanMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.schema.agent import AgentAction, AgentFinish

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Mock data generators
class DirectorySearchParams(BaseModel):
    """Parameters for directory search"""
    path: str = Field(..., description="Directory path to search in")
    include_pattern: str | None = Field(None, description="Glob pattern for files to include (e.g., '*.py')")
    exclude_pattern: str | None = Field(None, description="Glob pattern for files to exclude (e.g., '*.log')")

class MockDataGenerator:
    """Generates random but realistic mock data for APIs."""
    
    @staticmethod
    def get_weather(location: str) -> dict:
        """Generate mock weather data"""
        conditions = [
            "sunny", "partly cloudy", "cloudy", "rainy", "snowy",
            "foggy", "windy", "stormy", "clear", "overcast"
        ]
        temps = list(range(-10, 40))  # -10°C to 40°C
        humidities = list(range(30, 95))  # 30% to 95%
        
        return {
            "location": location,
            "temperature": random.choice(temps),
            "condition": random.choice(conditions),
            "humidity": random.choice(humidities),
            "wind_speed": random.randint(0, 30),
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def get_news(query: str) -> List[Dict[str, Any]]:
        """Generate random news articles."""
        topics = [
            "technology", "business", "sports", "entertainment",
            "health", "science", "politics", "environment"
        ]
        headlines = [
            f"Breaking: {query} makes major breakthrough",
            f"New developments in {query} sector",
            f"Experts weigh in on {query}",
            f"{query} trends show significant changes",
            f"Global impact of {query} continues to grow"
        ]
        
        return [
            {
                "title": random.choice(headlines),
                "source": f"{random.choice(['Tech', 'Global', 'Daily', 'World'])} News",
                "category": random.choice(topics),
                "published_at": datetime.now().isoformat(),
                "url": f"https://example.com/news/{random.randint(1000, 9999)}"
            }
            for _ in range(random.randint(3, 7))
        ]
    
    @staticmethod
    def get_currency(base: str) -> dict:
        """Generate mock currency exchange data"""
        currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "CNY"]
        rates = {
            currency: round(random.uniform(0.5, 2.0), 4)
            for currency in currencies
            if currency != base
        }
        rates[base] = 1.0
        return rates
    
    @staticmethod
    def get_directory_listing(params: DirectorySearchParams) -> List[Dict[str, Any]]:
        """Generate mock directory listing data"""
        # Simulate different file types and sizes
        file_types = ['.py', '.txt', '.log', '.yaml', '.json', '.md']
        file_sizes = [1024, 2048, 4096, 8192, 16384, 32768]  # 1KB to 32KB
        
        # Generate a random number of files and directories
        num_items = random.randint(5, 15)
        items = []
        
        for _ in range(num_items):
            is_dir = random.random() < 0.3  # 30% chance of being a directory
            if is_dir:
                name = f"dir_{random.randint(1, 100)}"
                items.append({
                    "name": name,
                    "type": "directory",
                    "size": 0,
                    "modified": datetime.now().isoformat()
                })
            else:
                ext = random.choice(file_types)
                name = f"file_{random.randint(1, 100)}{ext}"
                # Skip if file doesn't match include pattern
                if params.include_pattern and not fnmatch.fnmatch(name, params.include_pattern):
                    continue
                # Skip if file matches exclude pattern
                if params.exclude_pattern and fnmatch.fnmatch(name, params.exclude_pattern):
                    continue
                items.append({
                    "name": name,
                    "type": "file",
                    "size": random.choice(file_sizes),
                    "modified": datetime.now().isoformat()
                })
        
        return {
            "path": params.path,
            "total_items": len(items),
            "items": items,
            "timestamp": datetime.now().isoformat()
        }

# FastAPI server
app = FastAPI(title="Mock API Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint that lists all available API methods."""
    endpoints = [
        {
            "path": "/api/weather/{location}",
            "method": "GET",
            "description": "Get weather information for a location",
            "parameters": {
                "location": "string (path parameter) - The location to get weather for"
            }
        },
        {
            "path": "/api/news",
            "method": "GET",
            "description": "Get news articles",
            "parameters": {
                "query": "string (query parameter) - Optional search query for news"
            }
        },
        {
            "path": "/api/exchange/{base}",
            "method": "GET",
            "description": "Get currency exchange rates",
            "parameters": {
                "base": "string (path parameter) - Base currency code (e.g., USD, EUR)"
            }
        },
        {
            "path": "/api/dir_list",
            "method": "POST",
            "description": "List directory contents with filtering",
            "parameters": {
                "body": {
                    "path": "string (required) - Directory path to search in",
                    "include_pattern": "string (optional) - Glob pattern for files to include (e.g., '*.py')",
                    "exclude_pattern": "string (optional) - Glob pattern for files to exclude (e.g., '*.log')"
                }
            }
        }
    ]
    
    return {
        "name": "Mock API Server",
        "version": "1.0.0",
        "description": "A mock API server providing weather, news, currency exchange, and directory listing data",
        "endpoints": endpoints,
        "base_url": API_BASE_URL
    }

@app.get("/api/weather/{location}")
async def get_weather(location: str):
    """Mock weather API endpoint."""
    try:
        # Log the raw API request
        logger.info(f"Weather API Request - Location: {location}")
        
        # Generate mock data
        data = MockDataGenerator.get_weather(location)
        
        # Log the raw API response
        logger.info(f"Weather API Response - Data: {data}")
        
        return data
    except Exception as e:
        error_msg = f"Weather API error: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/news")
async def get_news(query: str = None):
    """Mock news API endpoint."""
    try:
        # Log the raw API request
        logger.info(f"News API Request - Query: {query}")
        
        # Generate mock data
        data = MockDataGenerator.get_news(query or "general")
        
        # Log the raw API response
        logger.info(f"News API Response - Data: {data}")
        
        return data
    except Exception as e:
        logger.error(f"News API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/exchange/{base}")
async def get_exchange_rate(base: str):
    """Mock currency API endpoint."""
    try:
        # Log the raw API request
        logger.info(f"Currency API Request - Base Currency: {base}")
        
        # Generate mock data
        data = MockDataGenerator.get_currency(base)
        
        # Log the raw API response
        logger.info(f"Currency API Response - Data: {data}")
        
        return data
    except Exception as e:
        error_msg = f"Currency API error: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/dir_list")
async def list_directory(params: DirectorySearchParams):
    """Mock directory listing API endpoint."""
    try:
        # Log the raw API request
        logger.info(f"Directory API Request - Params: {params.model_dump()}")
        
        # Generate mock data
        data = MockDataGenerator.get_directory_listing(params)
        
        # Log the raw API response
        logger.info(f"Directory API Response - Data: {data}")
        
        return data
    except Exception as e:
        error_msg = f"Directory API error: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=str(e))

# LangChain components
class AgentJSONEncoder(JSONEncoder):
    """Custom JSON encoder for agent-related objects."""
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, AgentAction):
            return {
                "type": "AgentAction",
                "tool": obj.tool,
                "tool_input": obj.tool_input,
                "log": obj.log
            }
        if isinstance(obj, AgentFinish):
            return {
                "type": "AgentFinish",
                "return_values": obj.return_values,
                "log": obj.log
            }
        if isinstance(obj, (HumanMessage, AIMessage)):
            return {
                "type": obj.__class__.__name__,
                "content": obj.content
            }
        return super().default(obj)

class LoggingCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for logging agent events."""
    
    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        """Log chain errors."""
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_type": "chain_error",
            "error_message": str(error),
            "kwargs": kwargs
        }
        logger.error(f"Chain error: {json.dumps(error_info, indent=2)}")
    
    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        """Log tool errors."""
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_type": "tool_error",
            "error_message": str(error),
            "tool_name": kwargs.get("tool_name", "unknown"),
            "tool_input": kwargs.get("tool_input", "unknown"),
            "error_details": {
                "type": type(error).__name__,
                "args": error.args,
                "traceback": str(error.__traceback__) if hasattr(error, "__traceback__") else None
            }
        }
        logger.error(f"Tool error: {json.dumps(error_info, indent=2)}")
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Log LLM errors."""
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_type": "llm_error",
            "error_message": str(error),
            "kwargs": kwargs
        }
        logger.error(f"LLM error: {json.dumps(error_info, indent=2)}")
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """Log when a tool starts."""
        tool_info = {
            "timestamp": datetime.now().isoformat(),
            "event": "tool_start",
            "tool_name": serialized.get("name", "unknown"),
            "input": input_str
        }
        logger.info(f"Tool execution: {json.dumps(tool_info, indent=2)}")
    
    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Log when a tool ends."""
        tool_info = {
            "timestamp": datetime.now().isoformat(),
            "event": "tool_end",
            "output": output
        }
        logger.info(f"Tool execution: {json.dumps(tool_info, indent=2)}")
    
    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        """Log agent actions."""
        action_info = {
            "timestamp": datetime.now().isoformat(),
            "event": "agent_action",
            "tool": action.tool,
            "tool_input": action.tool_input,
            "log": action.log
        }
        logger.info(f"Agent action: {json.dumps(action_info, indent=2)}")
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Log agent finishes."""
        finish_info = {
            "timestamp": datetime.now().isoformat(),
            "event": "agent_finish",
            "output": finish.return_values.get("output", ""),
            "log": finish.log
        }
        logger.info(f"Agent finish: {json.dumps(finish_info, indent=2)}")

def setup_environment(config):
    """Set up environment variables and validate configuration"""
    tool_source = YAMLToolSource(config.paths.tool_config)
    tool_registry = ToolRegistry()
    tool_loader = LangChainToolLoader(tool_source, tool_registry)
    
    # Load tools
    tools = tool_loader.load_tools()
    if not tools:
        raise ValueError("No tools were loaded successfully. Please check your tool configuration.")
    
    return tools

def initialize_components(config):
    """Initialize all components with configuration"""
    try:
        # Initialize Ollama LLM
        logger.info("Initializing Ollama LLM...")
        llm = OllamaLLM(
            model=config.ollama.model,
            temperature=config.ollama.temperature,
            base_url=config.ollama.base_url
        )
        logger.info("Ollama LLM initialized successfully")
        
        # Set up tools
        logger.info("Setting up tools...")
        tool_source = YAMLToolSource(config.paths.tool_config)
        tool_registry = ToolRegistry()
        tool_loader = LangChainToolLoader(tool_source, tool_registry)
        
        # Load tools
        logger.info("Loading tools...")
        tools = tool_loader.load_tools()
        if not tools:
            raise ValueError("No tools were loaded successfully. Please check your tool configuration.")
        logger.info(f"Loaded {len(tools)} tools successfully")
        
        # Load tool configurations for prompts
        logger.info("Loading tool configurations...")
        with open(config.paths.tool_config, 'r') as f:
            tool_configs = yaml.safe_load(f)
        
        # Build tool-specific prompts
        logger.info("Building tool prompts...")
        tool_prompts = []
        for tool_id, tool_config in tool_configs['tools'].items():
            if 'prompt' in tool_config:
                tool_prompts.append(f"{tool_config['name']}:\n{tool_config['prompt']}")
            if 'example' in tool_config:
                tool_prompts.append(f"Example for {tool_config['name']}:\n{tool_config['example']}")
        
        # Create the ReAct prompt template with variables
        logger.info("Creating ReAct prompt template...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", config.agent.system_prompt),
            ("human", "{input}"),
            ("ai", "{agent_scratchpad}")
        ]).partial(
            tools="\n".join(f"- {tool.name}: {tool.description}" for tool in tools),
            tool_names=", ".join(tool.name for tool in tools),
            tool_prompts="\n".join(tool_prompts)
        )
        logger.info("Prompt template created successfully")
        
        # Create the agent
        logger.info("Creating ReAct agent...")
        agent = create_react_agent(llm, tools, prompt)
        logger.info("ReAct agent created successfully")
        
        # Create the agent executor with callbacks
        logger.info("Creating agent executor...")
        callback_manager = CallbackManager([LoggingCallbackHandler()])
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=config.agent.verbose,
            max_iterations=config.agent.max_iterations,
            handle_parsing_errors=config.agent.handle_parsing_errors,
            callback_manager=callback_manager,
            return_intermediate_steps=True
        )
        logger.info("Agent executor created successfully")
        
        return agent_executor
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}", exc_info=True)
        raise

def start_api_server():
    """Start the FastAPI server in a separate thread."""
    try:
        logger.info("Starting FastAPI server...")
        uvicorn.run(
            app,
            host="localhost",
            port=API_PORT,
            log_level="info",
            timeout_keep_alive=30,
            timeout_graceful_shutdown=30
        )
    except Exception as e:
        logger.error(f"Failed to start API server: {str(e)}", exc_info=True)
        raise

def wait_for_server(max_retries=5, delay=1):
    """Wait for the API server to become available."""
    server_url = f"{API_BASE_URL}/api/weather/test"
    for attempt in range(max_retries):
        try:
            # Add timeout to the request
            response = requests.get(server_url, timeout=5)
            if response.status_code == 200:
                logger.info("API server is ready and responding")
                print("✓ API server is ready and responding")
                return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            logger.info(f"Waiting for API server to start... (attempt {attempt + 1}/{max_retries})")
            print(f"Waiting for API server to start... (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
    logger.error("API server failed to start within the timeout period")
    print("✗ API server failed to start within the timeout period")
    return False

def main():
    """Main function to run the example"""
    try:
        logger.info("Starting mock API server...")
        print("Starting mock API server...")
        
        # Start the API server in a separate thread
        api_thread = threading.Thread(target=start_api_server, daemon=True)
        api_thread.start()
        
        # Wait for the server to start and verify it's responding
        if not wait_for_server():
            logger.error("Failed to start API server. Exiting...")
            print("Failed to start API server. Exiting...")
            return
        
        # Additional delay to ensure server is fully ready
        logger.info("Waiting additional 3 seconds for server to stabilize...")
        print("Waiting additional 3 seconds for server to stabilize...")
        time.sleep(3)
        
        # Load configuration
        logger.info("Loading configuration...")
        print("Loading configuration...")
        config_path = "examples/config/app_config.yaml"
        config = ConfigLoader(config_path).get_config()
        
        # Initialize components
        logger.info("Initializing LangChain components...")
        print("Initializing LangChain components...")
        try:
            agent_executor = initialize_components(config)
            logger.info("LangChain components initialized successfully")
            print("✓ LangChain components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LangChain components: {str(e)}", exc_info=True)
            print(f"✗ Failed to initialize LangChain components: {str(e)}")
            return
        
        # Initialize chat history
        chat_history = []
        
        print("\nWelcome! Type 'exit' to quit.")
        print(f"The mock API server is running at {API_BASE_URL}")
        print("Available endpoints:")
        print("- GET /api/weather/{location}")
        print("- GET /api/news?query={query}")
        print("- GET /api/exchange/{base}")
        print("- POST /api/dir_list")
        
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                if user_input.lower() == 'exit':
                    break
                
                # Log the user input
                logger.info(f"User input: {user_input}")
                
                # Process the input
                response = agent_executor.invoke({
                    "input": user_input,
                    "chat_history": chat_history
                })
                
                # Log the response using our custom encoder
                logger.info(f"Agent response: {json.dumps(response, indent=2, cls=AgentJSONEncoder)}")
                
                # Update chat history
                chat_history.append(("user", user_input))
                chat_history.append(("assistant", response["output"]))
                
                # Print the response
                print(f"\nAssistant: {response['output']}")
                
            except Exception as e:
                logger.error(f"Error processing user input: {str(e)}", exc_info=True)
                print("\nAn error occurred. Please try again.")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        print(f"\nFatal error: {str(e)}")
    finally:
        # Cleanup
        logger.info("Shutting down...")
        print("\nShutting down...")

if __name__ == "__main__":
    main()
