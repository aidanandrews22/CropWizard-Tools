"""Module for interacting with the Quandl API."""

import logging
from typing import Dict, Any, Optional
import requests
from datetime import datetime

from config import QUANDL_API_KEY, QUANDL_BASE_URL, REQUEST_TIMEOUT, COMMODITY_CODES

logger = logging.getLogger(__name__)

class QuandlAPIError(Exception):
    """Custom exception for Quandl API errors."""
    pass

def _make_request(dataset_code: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Make a request to the Quandl API with error handling."""
    if params is None:
        params = {}
    params['api_key'] = QUANDL_API_KEY
    
    url = f"{QUANDL_BASE_URL}/{dataset_code}.json"
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Quandl API request failed: {str(e)}")
        raise QuandlAPIError(f"Failed to fetch data from Quandl API: {str(e)}")

def get_commodity_data(
    dataset_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetch commodity data from Quandl.
    
    Args:
        dataset_code: The Quandl dataset code (e.g., 'CHRIS/CME_C1' for corn futures)
        start_date: Optional start date in YYYY-MM-DD format
        end_date: Optional end date in YYYY-MM-DD format
    
    Returns:
        Dictionary containing the commodity data
    """
    params = {}
    if start_date:
        params['start_date'] = start_date
    if end_date:
        params['end_date'] = end_date
        
    return _make_request(dataset_code, params)

def get_commodity_by_name(
    commodity_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetch commodity data using a friendly name.
    
    Args:
        commodity_name: Name of the commodity (e.g., 'CORN', 'WHEAT')
        start_date: Optional start date in YYYY-MM-DD format
        end_date: Optional end date in YYYY-MM-DD format
    
    Returns:
        Dictionary containing the commodity data
    
    Raises:
        QuandlAPIError: If the commodity name is not recognized
    """
    commodity_name = commodity_name.upper()
    if commodity_name not in COMMODITY_CODES:
        raise QuandlAPIError(f"Unknown commodity: {commodity_name}. Available commodities: {', '.join(COMMODITY_CODES.keys())}")
    
    return get_commodity_data(
        COMMODITY_CODES[commodity_name],
        start_date,
        end_date
    ) 