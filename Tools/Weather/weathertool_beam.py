"""
To deploy: beam deploy weathertool_beam.py:main
For testing: beam serve weathertool_beam.py
"""

from beam import endpoint, Image
from typing import Any, Dict
import pgeocode
import requests
from geopy.geocoders import Nominatim
import pandas as pd
from datetime import datetime

def get_coordinates(location: str) -> tuple[float, float]:
    """Get coordinates from either zip code or city name."""
    # Try as zip code first
    if location.isdigit() and len(location) == 5:
        nomi = pgeocode.Nominatim('us')
        response = nomi.query_postal_code(location)
        if not pd.isna(response.latitude):
            return response.latitude, response.longitude
    
    # If not a zip code or zip code not found, try as city name
    geolocator = Nominatim(user_agent="weathertool")
    location_data = geolocator.geocode(location)
    if location_data:
        return location_data.latitude, location_data.longitude
    
    raise ValueError(f"Could not find coordinates for location: {location}")

def get_weather(location: str) -> dict:
    """Get weather data for a given location (city or zip code)."""
    try:
        lat, lon = get_coordinates(location)
        url = f"https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,wind_speed_10m",
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m"
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        weather_data = response.json()
        
        # Format the response in a more user-friendly way
        current = weather_data["current"]
        hourly = weather_data["hourly"]
        
        current_time = datetime.now().isoformat()
        
        return {
            "success": True,
            "request_time": current_time,
            "current_weather": {
                "temperature_celsius": current["temperature_2m"],
                "temperature_fahrenheit": (current["temperature_2m"] * 9/5) + 32,
                "wind_speed_kmh": current["wind_speed_10m"],
                "wind_speed_mph": current["wind_speed_10m"] * 0.621371,
                "time": current["time"]
            },
            "hourly_forecast": [{
                "time": hourly["time"][i],
                "temperature_celsius": hourly["temperature_2m"][i],
                "temperature_fahrenheit": (hourly["temperature_2m"][i] * 9/5) + 32,
                "humidity_percent": hourly["relative_humidity_2m"][i],
                "wind_speed_kmh": hourly["wind_speed_10m"][i],
                "wind_speed_mph": hourly["wind_speed_10m"][i] * 0.621371
            } for i in range(24)]
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

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
    API endpoint for Weather Tool.
    Accepts either a city name or ZIP code and returns current weather and hourly forecast.
    
    Example input:
    {
        "location": "New York"  # or "10001" for ZIP code
    }
    """
    location = inputs.get("location", "").strip()
    
    if not location:
        return {
            "success": False,
            "error": "No location provided. Please provide either a city name or ZIP code."
        }
    
    return get_weather(location)
