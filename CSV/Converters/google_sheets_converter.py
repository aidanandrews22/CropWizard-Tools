import pandas as pd
from urllib.parse import urlparse, parse_qs
import logging

def extract_sheet_id(sheet_url):
    """Extract the Google Sheet ID from the URL."""
    try:
        parsed_url = urlparse(sheet_url)
        if 'docs.google.com' not in parsed_url.netloc:
            raise ValueError("Not a valid Google Sheets URL")
            
        # Handle different URL formats
        if '/spreadsheets/d/' in sheet_url:
            sheet_id = sheet_url.split('/spreadsheets/d/')[1].split('/')[0]
        else:
            params = parse_qs(parsed_url.query)
            sheet_id = params.get('id', [None])[0]
            
        if not sheet_id:
            raise ValueError("Could not extract sheet ID from URL")
            
        return sheet_id
    except Exception as e:
        raise ValueError(f"Invalid Google Sheets URL: {str(e)}")

def convert_to_csv(sheet_url, output_path=None):
    """
    Convert a Google Sheet to CSV format.
    
    Args:
        sheet_url (str): URL of the Google Sheet
        output_path (str, optional): Path to save the CSV file. If None, returns the CSV content as string.
    
    Returns:
        str: CSV content if output_path is None, otherwise saves to file and returns path
    """
    try:
        sheet_id = extract_sheet_id(sheet_url)
        csv_export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        logging.info(f"Converting Google Sheet to CSV: {csv_export_url}")
        df = pd.read_csv(csv_export_url)
        
        if output_path:
            df.to_csv(output_path, index=False)
            return output_path
        else:
            return df.to_csv(index=False)
            
    except Exception as e:
        raise Exception(f"Error converting Google Sheet to CSV: {str(e)}") 