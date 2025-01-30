"""Module for aggregating data from multiple sources."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from arms_api import get_survey_data, ARMSAPIError
from quandl_api import get_commodity_by_name, get_commodity_data, QuandlAPIError

logger = logging.getLogger(__name__)

class AggregatorError(Exception):
    """Custom exception for aggregation errors."""
    pass

def get_combined_data(
    arms_params: Optional[Dict[str, Any]] = None,
    quandl_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Fetch and combine data from both ARMS and Quandl APIs.
    
    Args:
        arms_params: Dictionary containing ARMS API parameters
        quandl_params: Dictionary containing Quandl API parameters
    
    Returns:
        Dictionary containing combined data from both sources
    """
    result = {
        'timestamp': datetime.now().isoformat(),
        'success': True,
        'errors': [],
        'data': {}
    }
    
    # Fetch ARMS data if requested
    if arms_params:
        try:
            result['data']['arms'] = get_survey_data(**arms_params)
        except ARMSAPIError as e:
            result['success'] = False
            result['errors'].append(str(e))
            result['data']['arms'] = None
            logger.error(f"ARMS data fetch failed: {str(e)}")
    
    # Fetch Quandl data if requested
    if quandl_params:
        try:
            # Handle both dataset_code and commodity_name approaches
            if 'dataset_code' in quandl_params:
                result['data']['quandl'] = get_commodity_data(
                    quandl_params['dataset_code'],
                    quandl_params.get('start_date'),
                    quandl_params.get('end_date')
                )
            elif 'commodity_name' in quandl_params:
                result['data']['quandl'] = get_commodity_by_name(
                    quandl_params['commodity_name'],
                    quandl_params.get('start_date'),
                    quandl_params.get('end_date')
                )
            else:
                raise QuandlAPIError("Either dataset_code or commodity_name must be provided")
        except QuandlAPIError as e:
            result['success'] = False
            result['errors'].append(str(e))
            result['data']['quandl'] = None
            logger.error(f"Quandl data fetch failed: {str(e)}")
    
    return result

def validate_parameters(params: Dict[str, Any]) -> List[str]:
    """
    Validate the parameters for both APIs.
    
    Args:
        params: Dictionary containing parameters for both APIs
    
    Returns:
        List of validation error messages (empty if all valid)
    """
    errors = []
    
    # Validate ARMS parameters
    if 'arms' in params:
        arms_params = params['arms']
        if 'years' not in arms_params:
            errors.append("ARMS parameters must include 'years'")
        elif not isinstance(arms_params['years'], list):
            errors.append("ARMS 'years' must be a list of integers")
    
    # Validate Quandl parameters
    if 'quandl' in params:
        quandl_params = params['quandl']
        if not ('dataset_code' in quandl_params or 'commodity_name' in quandl_params):
            errors.append("Quandl parameters must include either 'dataset_code' or 'commodity_name'")
        
        # Validate date formats if provided
        for date_field in ['start_date', 'end_date']:
            if date_field in quandl_params:
                try:
                    datetime.strptime(quandl_params[date_field], '%Y-%m-%d')
                except ValueError:
                    errors.append(f"Quandl {date_field} must be in YYYY-MM-DD format")
    
    return errors 