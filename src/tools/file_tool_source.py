import json
import time
from pathlib import Path
from typing import Dict, Any
from src.interfaces.tool_source import ToolSource

class FileToolSource(ToolSource):
    """Example implementation of a file-based tool source"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.last_modified = 0
    
    def get_tool_configs(self) -> Dict[str, Any]:
        """Read tool configurations from a JSON file"""
        if not self.config_path.exists():
            return {}
        
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def watch_for_changes(self) -> None:
        """Watch for changes in the configuration file"""
        while True:
            if self.config_path.exists():
                current_modified = self.config_path.stat().st_mtime
                if current_modified > self.last_modified:
                    self.last_modified = current_modified
                    # In a real implementation, this would notify the registry
                    # about the changes
            time.sleep(1)  # Check every second 