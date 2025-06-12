from abc import ABC, abstractmethod
from typing import Dict, Any

class ToolRegistry(ABC):
    """Interface for tool registry service"""
    
    @abstractmethod
    def register_tool(self, tool_id: str, config: Dict[str, Any]) -> None:
        """Register a new tool configuration"""
        pass
    
    @abstractmethod
    def update_tool(self, tool_id: str, config: Dict[str, Any]) -> None:
        """Update an existing tool configuration"""
        pass
    
    @abstractmethod
    def remove_tool(self, tool_id: str) -> None:
        """Remove a tool configuration"""
        pass
    
    @abstractmethod
    def get_tool_config(self, tool_id: str) -> Dict[str, Any]:
        """Get configuration for a specific tool"""
        pass
    
    @abstractmethod
    def get_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered tool configurations"""
        pass 