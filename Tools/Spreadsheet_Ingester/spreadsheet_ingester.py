"""
To deploy: beam deploy spreadsheet_ingester.py:main
For testing: beam serve spreadsheet_ingester.py
"""

from beam import endpoint, Image
from typing import Any, Dict
import pandas as pd
import json
import os
import math
import numpy as np

def extract_sheet_id(sheet_url: str) -> str:
    """Extract the sheet ID from a Google Sheets URL."""
    # Handle both old and new Google Sheets URL formats
    if '/d/' in sheet_url:
        start = sheet_url.find('/d/') + 3
        end = sheet_url.find('/', start)
        if end == -1:
            end = sheet_url.find('?', start)
        if end == -1:
            end = len(sheet_url)
        return sheet_url[start:end]
    return ""

def clean_float_values(value):
    """Clean float values to ensure JSON compliance."""
    if isinstance(value, (float, np.float64)):
        if math.isnan(value) or math.isinf(value):
            return None
    return value

def read_google_sheet(sheet_url: str):
    """Read data from a Google Sheet and return it as a dictionary."""
    try:
        sheet_id = extract_sheet_id(sheet_url)
        if not sheet_id:
            return {"error": "Invalid Google Sheets URL"}

        # Create the CSV export URL
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        
        # Read the sheet into a pandas DataFrame
        df = pd.read_csv(csv_url)
        
        # Clean the data to ensure JSON compliance
        df = df.applymap(clean_float_values)
        
        # Convert DataFrame to dictionary format
        data = df.to_dict(orient='records')
        
        # Get column names
        columns = df.columns.tolist()
        
        return {
            "success": True,
            "columns": columns,
            "data": data,
            "row_count": len(data),
            "column_count": len(columns)
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to read Google Sheet: {str(e)}"
        }

@endpoint(
    name="spreadsheet_ingester",
    cpu=1,
    memory="2Gi",
    image=Image(python_version="python3.10").add_python_packages(["pandas==2.1.4"]),
    timeout=60,
    keep_warm_seconds=60 * 3
)
def main(**inputs: Dict[str, Any]):
    """
    API endpoint for Google Sheets Ingester Tool.
    Accepts a 'sheet_url' input and returns the sheet data in a structured format.
    
    You may want to create a markdown table of relevant data used for the user. Like this: 
    | Column 1 | Column 2 | Column 3 |
    |----------|----------|----------|
    | Row 1   | Data 1   | Data 2   |
    """
    sheet_url = inputs.get("sheet_url", "").strip()
    
    if not sheet_url:
        return {
            "error": "No Google Sheets URL provided"
        }
    
    try:
        result = read_google_sheet(sheet_url)
        return result
    except Exception as e:
        return {
            "error": f"An error occurred: {str(e)}"
        } 