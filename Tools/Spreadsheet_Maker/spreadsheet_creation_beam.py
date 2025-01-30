"""
Example Beam deployment for spreadsheet creation.
To deploy: beam deploy spreadsheet_creation_beam.py
For testing: beam serve spreadsheet_creation_beam.py
"""

from beam import endpoint, Image
from typing import Any, Dict
import json

# Import your existing logic:
from spreadsheet_creation import create_spreadsheet_from_data

@endpoint(
    name="spreadsheet_maker",  # Change to your preferred endpoint name
    cpu=1,
    memory="2Gi",
    image=Image(python_version="python3.10").add_python_packages(["pandas==2.0.3", "openpyxl==3.1.2"]),
    timeout=60,
    keep_warm_seconds=60 * 3
)
def main(**inputs: Dict[str, Any]):
    """
    API endpoint for Spreadsheet Creation.
    Accepts 'data', 'columns', 'filename', 'sheet_name', etc., then returns the path to the created spreadsheet.
    """
    try:
        # Retrieve inputs (all as strings from n8n), parse JSON fields as needed.
        data_str = inputs.get("data", "[]")
        data = json.loads(data_str) if data_str else []
        columns_str = inputs.get("columns", "")
        columns = json.loads(columns_str) if columns_str else None
        filename = inputs.get("filename", "output.xlsx")
        sheet_name = inputs.get("sheet_name", "Sheet1")
        
        # Create the spreadsheet
        path = create_spreadsheet_from_data(
            data=data,
            columns=columns,
            filename=filename,
            sheet_name=sheet_name
        )
        
        # Return the path or other status info
        return {
            "message": "Spreadsheet created successfully.",
            "spreadsheet_path": path
        }
    except Exception as e:
        return {
            "error": f"An error occurred: {str(e)}"
        } 