from typing import Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class ToolRegistry:
    """Registry for managing tool configurations and instances"""
    
    _tools: Dict[str, Any] = field(default_factory=dict)
    
    def register_tool(self, tool_id: str, tool_config: Dict[str, Any]) -> None:
        """Register a tool with its configuration"""
        self._tools[tool_id] = tool_config
    
    def get_tool(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get a tool's configuration by ID"""
        return self._tools.get(tool_id)
    
    def get_all_tools(self) -> Dict[str, Any]:
        """Get all registered tools"""
        return self._tools.copy()
    
    def unregister_tool(self, tool_id: str) -> None:
        """Remove a tool from the registry"""
        self._tools.pop(tool_id, None)
    
    def clear(self) -> None:
        """Clear all registered tools"""
        self._tools.clear()
    
    def has_tool(self, tool_id: str) -> bool:
        """Check if a tool is registered"""
        return tool_id in self._tools 