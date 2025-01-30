# Crop Wizard Market Data Tool

A robust tool for fetching and combining agricultural market data from USDA ARMS and Quandl APIs. This tool is designed to work with LLMs in a two-pass workflow:

1. First pass - LLM generates parameters for data fetching
2. Second pass - LLM interprets the fetched data and composes a response

## Features

- Dynamic API selection (ARMS, Quandl, or both)
- Robust error handling and validation
- Structured response format
- Minimal dependencies
- Production-ready logging

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```python
from main import fetch_data_tool

# Example parameters
params = {
    "sources": ["ARMS", "QUANDL"],
    "arms": {
        "years": [2015, 2016],
        "state": "all",
        "report": "income statement",
        "variable": "igcfi"
    },
    "quandl": {
        "commodity_name": "CORN",
        "start_date": "2024-01-01",
        "end_date": "2025-01-27"
    }
}

# Fetch data
result = fetch_data_tool(params)
```

## API Keys

The tool requires API keys for both USDA ARMS and Quandl:

- USDA ARMS API key: Set in `config.py`
- Quandl API key: Set in `config.py`

## Response Format

The tool returns a dictionary with the following structure:

```python
{
    "timestamp": "2024-01-27T12:00:00",
    "success": true,
    "errors": [],
    "data": {
        "arms": { ... },  # ARMS API response
        "quandl": { ... }  # Quandl API response
    }
}
```

## Error Handling

The tool includes comprehensive error handling:

- Parameter validation
- API-specific errors
- Network errors
- Unexpected errors

All errors are logged and returned in a structured format.

## Available Commodities

Currently supported commodities in Quandl:

- CORN (CME Corn Futures)
- WHEAT (CME Wheat Futures)
- SOYBEANS (CME Soybean Futures)

## Contributing

Feel free to submit issues and pull requests. 