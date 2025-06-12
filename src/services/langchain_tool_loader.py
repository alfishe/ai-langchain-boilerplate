from typing import Dict, Any, Type, Optional, List
from langchain.tools import BaseTool
import requests
from pydantic import BaseModel, Field
import os
from src.tools.tool_source import ToolSource
from src.services.tool_registry import ToolRegistry
import logging

class ToolConfig(BaseModel):
    """Pydantic model for tool configuration validation"""
    name: str = Field(description="Name of the tool")
    description: str = Field(description="Description of what the tool does")
    base_url: str = Field(description="Base URL for the API")
    endpoint: str = Field(description="API endpoint path")
    method: str = Field(default="GET", description="HTTP method to use")
    headers: Dict[str, str] = Field(default_factory=dict, description="HTTP headers")
    params: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")

class APITool(BaseTool):
    """LangChain tool for making API requests"""
    name: str
    description: str
    base_url: str
    endpoint: str
    method: str
    headers: Dict[str, str]
    params: Dict[str, Any]

    def _run(self, query: str) -> str:
        """Execute the API request with the given query"""
        try:
            # Prepare the request
            headers = self.headers.copy()
            params = self.params.copy()
            
            # Construct the full URL
            full_url = f"{self.base_url.rstrip('/')}/{self.endpoint.lstrip('/')}"
            
            # Replace path parameters
            if '{location}' in full_url:
                full_url = full_url.replace('{location}', query)
            elif '{base}' in full_url:
                full_url = full_url.replace('{base}', query)
            
            # Add query to params if it exists in the template
            for key, value in params.items():
                if isinstance(value, str) and "${query}" in value:
                    params[key] = value.replace("${query}", query)
            
            # Log the request
            logging.info(f"REST Tool Call - Tool: {self.name}")
            logging.info(f"REST Tool Call - URL: {full_url}")
            logging.info(f"REST Tool Call - Method: {self.method}")
            logging.info(f"REST Tool Call - Headers: {headers}")
            logging.info(f"REST Tool Call - Params: {params}")
            
            # Make the API request
            if self.method.upper() == 'POST':
                # For POST requests, try to parse the query as JSON
                try:
                    import json
                    data = json.loads(query)
                    response = requests.request(
                        method=self.method,
                        url=full_url,
                        headers=headers,
                        json=data
                    )
                except json.JSONDecodeError:
                    # If query is not valid JSON, send it as form data
                    response = requests.request(
                        method=self.method,
                        url=full_url,
                        headers=headers,
                        data=query
                    )
            else:
                # For GET requests, use params
                response = requests.request(
                    method=self.method,
                    url=full_url,
                    headers=headers,
                    params=params
                )
            
            response.raise_for_status()
            
            # Log the response
            logging.info(f"REST Tool Response - Tool: {self.name}")
            logging.info(f"REST Tool Response - Status: {response.status_code}")
            logging.info(f"REST Tool Response - Body: {response.text}")
            
            return response.text
        except requests.RequestException as e:
            logging.error(f"REST Tool Error - Tool: {self.name} - Error: {str(e)}")
            return f"Error making API request: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Execute the API request asynchronously"""
        return self._run(query)

class LangChainToolLoader:
    """Service for loading and converting tool configurations into LangChain tools"""
    
    def __init__(self, tool_source: ToolSource, tool_registry: ToolRegistry):
        """Initialize the loader with a source and registry"""
        self.tool_source = tool_source
        self.tool_registry = tool_registry
        self.tools: Dict[str, BaseTool] = {}
    
    def load_tools(self) -> List[BaseTool]:
        """Load tools from the source and convert them to LangChain tools"""
        tool_configs = self.tool_source.get_tool_configs()
        tools = []
        
        for tool_id, config in tool_configs.get('tools', {}).items():
            try:
                # Register the tool configuration
                self.tool_registry.register_tool(tool_id, config)
                
                # Create and store the LangChain tool
                tool = self.create_tool_from_config(tool_id, config)
                tools.append(tool)
                self.tools[tool_id] = tool
            except Exception as e:
                print(f"Error loading tool {tool_id}: {str(e)}")
                continue
        
        return tools
    
    def create_tool_from_config(self, tool_id: str, config: Dict[str, Any]) -> BaseTool:
        """Create a LangChain tool from a configuration dictionary"""
        # Validate and parse the configuration
        tool_config = ToolConfig(**config)
        
        # Create the tool instance
        return APITool(
            name=tool_config.name,
            description=tool_config.description,
            base_url=tool_config.base_url,
            endpoint=tool_config.endpoint,
            method=tool_config.method,
            headers=tool_config.headers,
            params=tool_config.params
        )
    
    def get_tool(self, tool_id: str) -> Optional[BaseTool]:
        """Get a LangChain tool by its ID"""
        return self.tools.get(tool_id)
    
    def get_all_tools(self) -> List[BaseTool]:
        """Get all loaded LangChain tools"""
        return list(self.tools.values())
    
    def get_tool_config(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get a tool's configuration by ID"""
        return self.tool_registry.get_tool(tool_id)
    
    def get_all_tool_configs(self) -> Dict[str, Any]:
        """Get all tool configurations"""
        return self.tool_registry.get_all_tools()

    def create_tool_from_config_old(self, tool_id: str, config: Dict[str, Any]) -> BaseTool:
        """Create a LangChain tool from a configuration"""
        api_config = APIToolConfig(**config)
        
        # Create a custom tool class for this API
        class APITool(BaseTool):
            name = api_config.name
            description = api_config.description
            
            def _run(self, **kwargs) -> str:
                try:
                    # Get the first endpoint (in a real implementation, you'd want to handle multiple endpoints)
                    endpoint = next(iter(api_config.endpoints.values()))
                    url = f"{api_config.base_url}{endpoint['path']}"
                    
                    # Add API key if it's in the parameters
                    params = kwargs.copy()
                    for param in endpoint.get('parameters', []):
                        if param.get('name') in ['key', 'apiKey']:
                            # Try to get API key from environment variable
                            env_var = f"{tool_id.upper()}_API_KEY"
                            if os.getenv(env_var):
                                params[param['name']] = os.getenv(env_var)
                    
                    # Make the request
                    response = requests.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    # Format the response based on the schema
                    return self._format_response(data, api_config.response_schema)
                except Exception as e:
                    return f"Error using {api_config.name}: {str(e)}"
            
            def _arun(self, **kwargs) -> str:
                raise NotImplementedError("Async not implemented")
            
            def _format_response(self, data: Dict[str, Any], schema: Dict[str, Any]) -> str:
                """Format the response based on the schema"""
                if schema['type'] == 'object':
                    result = []
                    for prop_name, prop_schema in schema.get('properties', {}).items():
                        if prop_name in data:
                            if prop_schema['type'] == 'object':
                                result.append(f"{prop_name}: {self._format_response(data[prop_name], prop_schema)}")
                            elif prop_schema['type'] == 'array':
                                items = data[prop_name]
                                if isinstance(items, list):
                                    result.append(f"{prop_name}: {', '.join(str(item) for item in items)}")
                            else:
                                result.append(f"{prop_name}: {data[prop_name]}")
                    return ' | '.join(result)
                return str(data)
        
        return APITool() 