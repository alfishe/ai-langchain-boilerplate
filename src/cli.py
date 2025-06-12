import argparse
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

from src.tools.yaml_tool_source import YAMLToolSource
from src.services.tool_registry import ToolRegistry
from src.services.langchain_tool_loader import LangChainToolLoader
from src.config.config_loader import ConfigLoader
from src.config.app_config import AppConfig

def setup_logging(verbose: bool = False, config: Optional[AppConfig] = None):
    """Set up logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear any existing handlers
    root_logger.handlers = []
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if config is provided
    if config and config.paths.log_file:
        file_handler = RotatingFileHandler(
            config.paths.log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Log the start of the application
        logging.info("Application started")
        if config.agent.system_prompt:
            logging.info(f"System Prompt:\n{config.agent.system_prompt}")

def list_tools(config_path: str):
    """List all available tools"""
    config = ConfigLoader(config_path).get_config()
    tool_source = YAMLToolSource(config.paths.tool_config)
    tool_registry = ToolRegistry()
    tool_loader = LangChainToolLoader(tool_source, tool_registry)
    
    tools = tool_loader.load_tools()
    if not tools:
        print("No tools found.")
        return
    
    print("\nAvailable Tools:")
    print("-" * 50)
    for tool_id, tool in tools.items():
        print(f"\nTool ID: {tool_id}")
        print(f"Name: {tool.name}")
        print(f"Description: {tool.description}")
        print("-" * 50)

def validate_config(config_path: str):
    """Validate tool configurations"""
    config = ConfigLoader(config_path).get_config()
    tool_source = YAMLToolSource(config.paths.tool_config)
    tool_registry = ToolRegistry()
    tool_loader = LangChainToolLoader(tool_source, tool_registry)
    
    try:
        tools = tool_loader.load_tools()
        print(f"Configuration is valid. Found {len(tools)} tools.")
    except Exception as e:
        print(f"Configuration validation failed: {str(e)}")
        sys.exit(1)

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="AI LangChain Tools CLI")
    parser.add_argument(
        "--config",
        type=str,
        default="examples/config/app_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List tools command
    list_parser = subparsers.add_parser("list", help="List available tools")
    
    # Validate config command
    validate_parser = subparsers.add_parser("validate", help="Validate tool configurations")
    
    args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.get_config()
    
    # Set up logging with config
    setup_logging(args.verbose, config)
    
    # Execute command
    if args.command == "list":
        list_tools(args.config)
    elif args.command == "validate":
        validate_config(args.config)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 