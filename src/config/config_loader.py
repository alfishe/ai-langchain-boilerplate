import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class PathConfig(BaseModel):
    """Path configuration settings"""
    tool_config: str = "examples/config/api_tools.yaml"
    log_file: str = "app.log"

class OllamaConfig(BaseModel):
    """Ollama configuration settings"""
    model: str = "mistral"
    temperature: float = 0.7
    base_url: str = "http://localhost:11434"
    timeout: int = 30
    max_tokens: int = 2000

class AgentConfig(BaseModel):
    """Agent configuration settings"""
    type: str = "zero-shot-react-description"
    verbose: bool = True
    max_iterations: int = 5
    handle_parsing_errors: bool = True
    system_prompt: str = "You are a helpful AI assistant that can use various tools to help users. You have access to REST API tools that can make HTTP requests. When using these tools: 1. Always check the response status and handle errors appropriately 2. Format the response in a user-friendly way 3. Explain what you're doing before using a tool 4. If a tool call fails, try to understand why and suggest alternatives 5. Keep responses concise but informative"

class ToolConfig(BaseModel):
    """Tool configuration settings"""
    default_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: int = 1

class LoggingConfig(BaseModel):
    """Logging configuration settings"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

class RateLimitingConfig(BaseModel):
    """Rate limiting configuration settings"""
    enabled: bool = True
    requests_per_minute: int = 60
    burst_limit: int = 10

class ResponseFormattingConfig(BaseModel):
    """Response formatting configuration settings"""
    date_format: str = "%Y-%m-%d %H:%M:%S"
    number_format: str = "{:.2f}"
    max_items_in_list: int = 5

class AppConfig(BaseModel):
    """Application configuration settings"""
    paths: PathConfig = Field(default_factory=PathConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    rate_limiting: RateLimitingConfig = Field(default_factory=RateLimitingConfig)
    response_formatting: ResponseFormattingConfig = Field(default_factory=ResponseFormattingConfig)

class ConfigLoader:
    """Configuration loader for application settings"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config: Optional[AppConfig] = None
    
    def load_config(self) -> AppConfig:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            return AppConfig()  # Return default configuration
        
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        self.config = AppConfig(**config_data)
        return self.config
    
    def get_config(self) -> AppConfig:
        """Get current configuration, loading if necessary"""
        if self.config is None:
            self.config = self.load_config()
        return self.config
    
    def reload_config(self) -> AppConfig:
        """Reload configuration from file"""
        self.config = None
        return self.get_config() 