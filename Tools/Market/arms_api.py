"""Module for interacting with the USDA ARMS API."""

import logging
import requests
from typing import List, Optional, Dict, Any
from datetime import datetime

from config import USDA_API_KEY, USDA_BASE_URL, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

class ARMSAPIError(Exception):
    """Custom exception for ARMS API errors."""
    pass

def _make_request(endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Make a request to the ARMS API with error handling."""
    if params is None:
        params = {}
    params['api_key'] = USDA_API_KEY
    
    url = f"{USDA_BASE_URL}/{endpoint}"
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"ARMS API request failed: {str(e)}")
        raise ARMSAPIError(f"Failed to fetch data from ARMS API: {str(e)}")

def get_available_states() -> List[str]:
    """Get list of available states from the ARMS API."""
    return _make_request('state')

def get_available_years() -> List[int]:
    """Get list of available years from the ARMS API."""
    return _make_request('year')

def get_survey_data(
    years: List[int],
    state: Optional[str] = None,
    report: Optional[str] = None,
    variable: Optional[str] = None,
    farmtype: Optional[str] = None,
    category: Optional[str] = None,
    category_value: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetch survey data from the ARMS API.
    
    Args:
        years: List of years to fetch data for
        state: Optional state filter
        report: Optional report type
        variable: Optional variable filter
        farmtype: Optional farm type filter
        category: Optional category filter
        category_value: Optional category value filter
    
    Returns:
        Dictionary containing the survey data
    """
    params = {
        'year': ','.join(map(str, years))
    }
    
    # Add optional parameters if provided
    if state:
        params['state'] = state
    if report:
        params['report'] = report
    if variable:
        params['variable'] = variable
    if farmtype:
        params['farmtype'] = farmtype
    if category:
        params['category'] = category
    if category_value:
        params['category_value'] = category_value
        
    return _make_request('surveydata', params)

def get_metadata() -> Dict[str, Any]:
    """Get metadata about available reports, variables, and categories."""
    return {
        'reports': _make_request('report'),
        'variables': _make_request('variable'),
        'categories': _make_request('category')
    } 