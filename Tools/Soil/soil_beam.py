"""
To deploy: beam deploy soil_beam.py:main
For testing: beam serve soil_beam.py
"""

from beam import endpoint, Image
from typing import Any, Dict
import requests

@endpoint(
    name="weather_tool",
    cpu=1,
    memory="2Gi",
    image=Image(python_version="python3.10").add_python_packages([
        "numpy==1.24.3",
        "pandas==1.5.3",
        "pgeocode==0.4.0",
        "geopy==2.3.0",
        "requests==2.31.0"
    ]),
    timeout=30,
    keep_warm_seconds=60 * 3
)
def main(**inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    API endpoint for Soil Tool.
    Accepts either a city name or ZIP code and returns current soil and hourly forecast for that location
    """
    location = inputs.get("location", "").strip()
    return get_soil(location)
