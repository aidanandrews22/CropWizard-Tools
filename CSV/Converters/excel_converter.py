import pandas as pd
from pathlib import Path

def convert_to_csv(excel_path, output_path=None, sheet_name=0):
    """
    Convert an Excel file to CSV format.
    
    Args:
        excel_path (str): Path to the Excel file or URL
        output_path (str, optional): Path to save the CSV file. If None, returns the CSV content as string.
        sheet_name (str|int, optional): Name or index of the sheet to convert. Defaults to first sheet.
    
    Returns:
        str: CSV content if output_path is None, otherwise saves to file and returns path
    """
    try:
        # Check if the input is a URL or local file
        if excel_path.startswith(('http://', 'https://')):
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
        else:
            if not Path(excel_path).exists():
                raise FileNotFoundError(f"Excel file not found: {excel_path}")
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
        
        if output_path:
            df.to_csv(output_path, index=False)
            return output_path
        else:
            return df.to_csv(index=False)
            
    except Exception as e:
        raise Exception(f"Error converting Excel file to CSV: {str(e)}")

def list_sheet_names(excel_path):
    """
    List all sheet names in an Excel file.
    
    Args:
        excel_path (str): Path to the Excel file or URL
    
    Returns:
        list: List of sheet names
    """
    try:
        if excel_path.startswith(('http://', 'https://')):
            excel = pd.ExcelFile(excel_path)
        else:
            if not Path(excel_path).exists():
                raise FileNotFoundError(f"Excel file not found: {excel_path}")
            excel = pd.ExcelFile(excel_path)
            
        return excel.sheet_names
    except Exception as e:
        raise Exception(f"Error reading Excel file: {str(e)}") 