# AI LangChain Tools

A flexible and extensible framework for managing and using tools with LangChain agents. This project provides a structured way to define, load, and use external tools (like API integrations) with LangChain agents.

## Features

- YAML-based tool configuration
- Tool registry for managing tool configurations
- Automatic conversion of tool configurations to LangChain tools
- Support for REST API tools with automatic request handling
- Ollama integration for local LLM support
- Extensible architecture for adding new tool types

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-langchain-tools.git
cd ai-langchain-tools

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

1. Configure your tools in YAML:
```yaml
tools:
  weather:
    name: "Weather API"
    description: "Get current weather information for a location"
    endpoint: "https://api.weatherapi.com/v1/current.json"
    method: "GET"
    headers:
      Authorization: "Bearer ${WEATHER_API_KEY}"
    params:
      key: "${WEATHER_API_KEY}"
      q: "${query}"
```

2. Set up your environment variables:
```bash
export WEATHER_API_KEY="your_api_key"
```

3. Run the example:
```bash
python examples/rest_api_tools.py
```

## Project Structure

```
ai-langchain-tools/
├── src/
│   ├── config/         # Configuration management
│   ├── services/       # Core services
│   └── tools/          # Tool implementations
├── examples/           # Example scripts
│   └── config/         # Example configurations
├── tests/             # Test suite
├── requirements.txt   # Project dependencies
└── setup.py          # Package setup
```

## Components

### Tool Registry
Manages tool configurations and provides a central registry for tool management.

### LangChain Tool Loader
Converts tool configurations into LangChain tools and handles the integration with LangChain agents.

### YAML Tool Source
Loads tool configurations from YAML files with support for environment variable substitution.

## Configuration

The project uses YAML for configuration. See `examples/config/` for example configurations:

- `app_config.yaml`: Application settings
- `api_tools.yaml`: Tool definitions

## Development

### Adding New Tools

1. Define your tool configuration in YAML
2. Implement any necessary tool-specific logic
3. Register the tool in the registry

### Running Tests

```bash
pytest tests/
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 