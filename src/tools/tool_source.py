from abc import ABC, abstractmethod
from typing import Dict, List, Optional

class ToolSource(ABC):
    """Base interface for tool sources that provide tool configurations."""
    
    @abstractmethod
    def get_tools(self) -> List[Dict]:
        """Get all available tools from the source.
        
        Returns:
            List[Dict]: List of tool configurations, where each tool is a dictionary
                       containing the tool's configuration.
        """
        pass
    
    @abstractmethod
    def get_tool(self, tool_id: str) -> Optional[Dict]:
        """Get a specific tool by its ID.
        
        Args:
            tool_id (str): The unique identifier of the tool.
            
        Returns:
            Optional[Dict]: The tool configuration if found, None otherwise.
        """
        pass
    
    @abstractmethod
    def add_tool(self, tool_config: Dict) -> bool:
        """Add a new tool to the source.
        
        Args:
            tool_config (Dict): The tool configuration to add.
            
        Returns:
            bool: True if the tool was added successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    def update_tool(self, tool_id: str, tool_config: Dict) -> bool:
        """Update an existing tool in the source.
        
        Args:
            tool_id (str): The unique identifier of the tool to update.
            tool_config (Dict): The new tool configuration.
            
        Returns:
            bool: True if the tool was updated successfully, False otherwise.
        """
        pass
    
    @abstractmethod
    def remove_tool(self, tool_id: str) -> bool:
        """Remove a tool from the source.
        
        Args:
            tool_id (str): The unique identifier of the tool to remove.
            
        Returns:
            bool: True if the tool was removed successfully, False otherwise.
        """
        pass 