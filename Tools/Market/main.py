"""Main module for the Crop Wizard Market Data tool."""

import logging
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

from aggregator import get_combined_data, validate_parameters, AggregatorError
from arms_api import ARMSAPIError
from quandl_api import QuandlAPIError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fetch_data_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for the Crop Wizard Market Data tool.
    
    Args:
        params: Dictionary containing parameters for data fetching
        Example:
        {
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
    
    Returns:
        Dictionary containing the fetched data and metadata
    """
    try:
        # Validate parameters
        validation_errors = validate_parameters(params)
        if validation_errors:
            return {
                'success': False,
                'errors': validation_errors,
                'data': None
            }
        
        # Extract parameters for each API based on requested sources
        sources = params.get('sources', [])
        arms_params = params.get('arms') if 'ARMS' in sources else None
        quandl_params = params.get('quandl') if 'QUANDL' in sources else None
        
        # If no sources specified, return error
        if not sources:
            return {
                'success': False,
                'errors': ['No data sources specified'],
                'data': None
            }
        
        # Fetch and combine data
        result = get_combined_data(arms_params, quandl_params)
        
        return result
        
    except (ARMSAPIError, QuandlAPIError, AggregatorError) as e:
        logger.error(f"Error in fetch_data_tool: {str(e)}")
        return {
            'success': False,
            'errors': [str(e)],
            'data': None
        }
    except Exception as e:
        logger.error(f"Unexpected error in fetch_data_tool: {str(e)}")
        return {
            'success': False,
            'errors': ['An unexpected error occurred'],
            'data': None
        }

if __name__ == '__main__':
    # Example usage
    example_params = {
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
    
    result = fetch_data_tool(example_params)
    print(json.dumps(result, indent=2)) 