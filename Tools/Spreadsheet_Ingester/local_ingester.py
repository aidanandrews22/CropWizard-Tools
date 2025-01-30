"""
Local implementation of the spreadsheet ingester for testing.
Usage: python local_ingester.py <google_sheets_url>
"""

import pandas as pd
import json
import sys
from typing import Dict, Any

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

def read_google_sheet(sheet_url: str) -> Dict[str, Any]:
    """Read data from a Google Sheet and return it as a dictionary."""
    try:
        sheet_id = extract_sheet_id(sheet_url)
        if not sheet_id:
            return {"error": "Invalid Google Sheets URL"}

        # Create the CSV export URL
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        
        # Read the sheet into a pandas DataFrame
        df = pd.read_csv(csv_url)
        
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

def main():
    """Main function for local testing."""
    if len(sys.argv) != 2:
        print("Usage: python local_ingester.py <google_sheets_url>")
        sys.exit(1)
    
    sheet_url = sys.argv[1].strip()
    
    if not sheet_url:
        print(json.dumps({"error": "No Google Sheets URL provided"}, indent=2))
        sys.exit(1)
    
    try:
        result = read_google_sheet(sheet_url)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({"error": f"An error occurred: {str(e)}"}, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main() 