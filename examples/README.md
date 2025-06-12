# REST API Tools Example

This example demonstrates how to use REST API tools with LangChain. It shows how to:
1. Run a mock API server for weather, news, and currency data
2. Load API tool configurations from YAML
3. Convert them to LangChain tools
4. Use them with a LangChain agent powered by Ollama

## Architecture

```
┌──────────────────────┐
│ Mock API Server      │
│ - Weather API        │
│ - News API           │
│ - Currency API       │
└──────────────────────┘
           │
           ▼
┌───────────────────────┐
│ YAML Configuration    │
│ - API endpoints       │
│ - Parameters          │
│ - Response schemas    │
└───────────────────────┘
           │
           ▼
┌───────────────────────┐
│ LangChain Tools       │
│ - Weather Tool        │
│ - News Tool           │
│ - Currency Tool       │
└───────────────────────┘
           │
           ▼
┌───────────────────────┐
│ Ollama LLM            │
│ - Local Model         │
│ - Zero-shot ReAct     │
└───────────────────────┘
```

## Available Tools

### 1. Weather API Tool
- **Endpoint**: `/api/weather/{location}`
- **Method**: GET
- **Functionality**: Get current weather for any location
- **Example Query**: "What's the weather like in London?"
- **Response Format**: 
  ```json
  {
    "location": "London",
    "temperature": 15,
    "condition": "partly cloudy",
    "humidity": 65,
    "wind_speed": 12,
    "timestamp": "2025-06-12T13:43:52.300101"
  }
  ```

### 2. News API Tool
- **Endpoint**: `/api/news?query={query}`
- **Method**: GET
- **Functionality**: Get latest news headlines
- **Example Query**: "What are the latest news headlines?"
- **Response Format**: List of news articles with titles, sources, and categories

### 3. Currency Exchange Tool
- **Endpoint**: `/api/exchange/{base}`
- **Method**: GET
- **Functionality**: Get currency exchange rates
- **Example Query**: "What's the exchange rate between USD and EUR?"
- **Response Format**: Exchange rates for various currencies relative to the base currency

## Prerequisites

- Python 3.8+
- Ollama installed and running locally

## Setup

1. Install Ollama:
   - Follow instructions at https://ollama.ai/download
   - Pull the Mistral model:
     ```bash
     ollama pull mistral
     ```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Example

1. Ensure Ollama is running:
```bash
ollama serve
```

2. Run the example script:
```bash
python examples/example-embedded-api-server.py
```

The script will:
1. Start a mock API server on port 8866
2. Load API tool configurations from `examples/config/mock_api_tools.yaml`
3. Create LangChain tools from these configurations
4. Initialize a LangChain agent with Ollama
5. Start an interactive session where you can ask questions

## API Documentation

The mock API server provides automatic documentation at the root endpoint:
```bash
curl http://localhost:8866/
```

This will return a JSON response listing all available endpoints, their methods, parameters, and descriptions.

## Expected Behavior

When running the example, you should see:
1. Mock API server startup messages
2. Tool loading and initialization
3. Agent initialization with Ollama
4. Interactive prompt where you can ask questions

Example interaction:
```
Starting mock API server...
✓ API server is ready and responding
Loading configuration...
Initializing LangChain components...
✓ LangChain components initialized successfully

Welcome! Type 'exit' to quit.
The mock API server is running at http://localhost:8866

You: What's the weather in San Francisco?
Assistant: The current weather in San Francisco (SFO) is snowy with a temperature of 10°C, 48% humidity, and wind speed of 13 km/h.
```

## Troubleshooting

### Common Issues and Solutions

1. **Ollama Issues**
   - Symptom: "Connection refused" or "Model not found"
   - Solution: 
     - Ensure Ollama is running: `ollama serve`
     - Verify model is pulled: `ollama list`
     - Check Ollama URL: `http://localhost:11434`

2. **API Server Issues**
   - Symptom: "Connection refused" on port 8866
   - Solution: 
     - Check if another process is using port 8866
     - Verify the server started successfully
     - Check server logs for errors

3. **Network Issues**
   - Symptom: Connection timeouts or refused connections
   - Solution: Check your internet connection
   - Verify: `ping localhost`

4. **Invalid Responses**
   - Symptom: Unexpected or malformed responses
   - Solution: Check the API documentation at `http://localhost:8866/`
   - Verify: Your query format matches the API requirements

### Debug Mode

To enable debug logging, set the environment variable:
```bash
export LOG_LEVEL=DEBUG
```

## Customization

You can:
1. Add more API endpoints to the mock server
2. Modify the mock data generation in `MockDataGenerator` class
3. Add more API tools to `examples/config/mock_api_tools.yaml`
4. Create custom tools by implementing the `BaseTool` interface
5. Modify the agent configuration in the script
6. Try different Ollama models by changing the model name in the script

### Adding New Endpoints

1. Add a new endpoint to the FastAPI server:
```python
@app.get("/api/new-endpoint/{param}")
async def new_endpoint(param: str):
    """New endpoint description."""
    try:
        data = MockDataGenerator.get_new_data(param)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

2. Add corresponding tool configuration to `examples/config/mock_api_tools.yaml`
3. Update the example script to include the new tool
4. Test with example queries 