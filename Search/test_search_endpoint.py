#!/usr/bin/env python3
"""
Script to test the local Beam search endpoint.
"""

import requests
import json
import argparse
from urllib.parse import quote_plus

# Constants
ENDPOINT_URL = "https://4d0193c2-f89c-4d01-8309-abedc47b2348.app.beam.cloud"
API_KEY = "VyneHwQmIAvwNfPVZuwp860GkadAO8wH29dCERTqcX9u2G6A9kKR42AU8HKql4q4GUt_n-Yc0_4ZSSyDXzfNWA=="

def test_search_endpoint(query):
    """Test the search endpoint with the given query."""
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    
    payload = {
        'query': query
    }
    
    print(f"Sending request to {ENDPOINT_URL} with query: '{query}'")
    
    try:
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload)
        
        # Print status code
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            # Pretty print JSON response if valid
            try:
                result = response.json()
                print("\nResponse:")
                print(json.dumps(result, indent=2))
            except json.JSONDecodeError:
                # If not JSON, print text response
                print("\nResponse (text):")
                print(response.text)
        else:
            print(f"Error: {response.text}")
    
    except Exception as e:
        print(f"Exception occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Beam search endpoint")
    parser.add_argument("query", nargs="?", default="What is the fastest electric car?", 
                      help="The search query to send (default: 'What is the fastest electric car?')")
    
    args = parser.parse_args()
    test_search_endpoint(args.query) 