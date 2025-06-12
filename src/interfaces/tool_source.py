from abc import ABC, abstractmethod
from typing import Dict, Any

class ToolSource(ABC):
    """Interface for external tool sources (File, DB, API)"""
    
    @abstractmethod
    def get_tool_configs(self) -> Dict[str, Any]:
        """Retrieve tool configurations from the source"""
        pass
    
    @abstractmethod
    def watch_for_changes(self) -> None:
        """Watch for changes in the tool configurations"""
        pass 