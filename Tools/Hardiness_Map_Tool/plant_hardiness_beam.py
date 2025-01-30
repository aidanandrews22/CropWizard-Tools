"""
To deploy: beam deploy plant_hardiness_beam.py
For testing: beam serve plant_hardiness_beam.py
"""

from beam import endpoint, Image
from typing import Any, Dict
import json
import csv
import os
import sys

# CSV files containing ZIP-to-zone mappings:
CSV_FILES = [
    "phzm_ak_zipcode_2023.csv",
    "phzm_hi_zipcode_2023.csv",
    "phzm_pr_zipcode_2023.csv",
    "phzm_us_zipcode_2023.csv"
]

def load_map_html():
    """Load the map HTML from file."""
    try:
        with open("plant_hardiness_zone_map.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error loading map HTML: {e}")
        return "<!-- Error loading map HTML -->"

def load_general_info():
    """Load the general information from file."""
    try:
        with open("general_info.md", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error loading general info: {e}")
        return "Error loading general information."

def find_zip_data(zipcode: str):
    """Searches for the given zipcode in all CSV files and returns the zone info if found."""
    zipcode = zipcode.strip()
    for csv_file in CSV_FILES:
        if os.path.exists(csv_file):
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("zipcode") == zipcode:
                        return {
                            "zipcode": zipcode,
                            "zone": row.get("zone"),
                            "temp_range": row.get("trange"),
                            "zone_title": row.get("zonetitle")
                        }
    return None

def run_tool(zipcode: str = ""):
    """Run the tool given an optional zipcode."""
    # Load content from files
    map_html = load_map_html()
    general_info = load_general_info()
    
    if zipcode == "":
        # No ZIP provided: just return general info and map
        result = {
            "general_info": general_info,
            "map_html": map_html,
            "message": "No specific ZIP code provided. Please use the map's ZIP code search function."
        }
        return result
    else:
        # ZIP provided: look it up
        zone_data = find_zip_data(zipcode)
        if zone_data:
            # Found the zone data for this ZIP code
            result = {
                "general_info": general_info,
                "map_html": map_html,
                "zone_data": zone_data
            }
            return result
        else:
            # Not found
            result = {
                "general_info": general_info,
                "map_html": map_html,
                "message": f"No data available for ZIP code {zipcode}."
            }
            return result

@endpoint(
    name="plant_hardiness",
    cpu=1,
    memory="2Gi",
    image=Image(python_version="python3.10").add_python_packages(["python-multipart==0.0.5"]),
    timeout=60,
    keep_warm_seconds=60 * 3
)
def main(**inputs: Dict[str, Any]):
    """
    API endpoint for Plant Hardiness Zone Tool.
    Accepts an optional 'zipcode' input and returns the map and relevant information.
    """
    zipcode = inputs.get("zipcode", "").strip()
    try:
        # Call the core tool logic
        result = run_tool(zipcode)
        return result
    except Exception as e:
        # Handle errors gracefully
        return {
            "error": f"An error occurred: {str(e)}"
        }