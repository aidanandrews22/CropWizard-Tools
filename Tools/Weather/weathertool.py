import pgeocode
import requests
from geopy.geocoders import Nominatim
from typing import Union, Tuple
from datetime import datetime
import pandas as pd

def get_coordinates(location: str) -> Tuple[float, float]:
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
        return response.json()
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python weathertool.py <city or zip code>")
        sys.exit(1)
    
    location = sys.argv[1]
    weather_data = get_weather(location)
    
    if "error" in weather_data:
        print(f"Error: {weather_data['error']}")
    else:
        current = weather_data["current"]
        print(f"\nCurrent Weather:")
        print(f"Temperature: {current['temperature_2m']}°C")
        print(f"Wind Speed: {current['wind_speed_10m']} km/h")
        
        print("\nHourly Forecast (next 24 hours):")
        hourly = weather_data["hourly"]
        for i in range(24):
            print(f"\nTime: {hourly['time'][i]}")
            print(f"Temperature: {hourly['temperature_2m'][i]}°C")
            print(f"Humidity: {hourly['relative_humidity_2m'][i]}%")
            print(f"Wind Speed: {hourly['wind_speed_10m'][i]} km/h")