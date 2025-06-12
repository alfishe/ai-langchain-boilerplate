import yaml
from pathlib import Path
from typing import Dict, Any
from src.interfaces.tool_source import ToolSource
import time

class YAMLToolSource(ToolSource):
    """Implementation of ToolSource that reads from YAML files"""
    
    def __init__(self, yaml_path: str):
        self.yaml_path = Path(yaml_path)
        self.last_modified = 0
    
    def get_tool_configs(self) -> Dict[str, Any]:
        """Read tool configurations from a YAML file"""
        if not self.yaml_path.exists():
            return {}
        
        with open(self.yaml_path, 'r') as f:
            return yaml.safe_load(f)
    
    def watch_for_changes(self) -> None:
        """Watch for changes in the YAML file"""
        while True:
            if self.yaml_path.exists():
                current_modified = self.yaml_path.stat().st_mtime
                if current_modified > self.last_modified:
                    self.last_modified = current_modified
                    # In a real implementation, this would notify the registry
                    # about the changes
            time.sleep(1)  # Check every second 