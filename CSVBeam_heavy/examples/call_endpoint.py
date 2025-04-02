#!/usr/bin/env python3
"""
Example script to call the CSV Analyzer Beam endpoint
"""

import requests
import json
import os
import sys

# The endpoint URL - replace with your actual endpoint after deployment
ENDPOINT_URL = "https://83558bcd-3fb2-4427-8fe5-de04e336c682.app.beam.cloud"
# Or use the local URL when testing
# ENDPOINT_URL = "http://localhost:8000"

# Authentication token for the Beam endpoint
AUTH_TOKEN = "Bearer VyneHwQmIAvwNfPVZuwp860GkadAO8wH29dCERTqcX9u2G6A9kKR42AU8HKql4q4GUt_n-Yc0_4ZSSyDXzfNWA=="

def call_endpoint(query, google_sheets=None):
    """Call the CSV Analyzer endpoint with the given query and optional Google Sheets."""
    
    # Prepare the request payload
    payload = {
        "query": query
    }
    
    if google_sheets:
        payload["google_sheets"] = google_sheets
    
    print(f"Sending request to {ENDPOINT_URL}...")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    # Make the API request
    try:
        # Try with different content types and methods
        
        # 1. JSON POST request
        print("\n1. Trying JSON POST request...")
        headers_json = {
            "Content-Type": "application/json",
            "Authorization": AUTH_TOKEN
        }
        
        response_json = requests.post(ENDPOINT_URL, json=payload, headers=headers_json, timeout=120)
        print(f"Response status code: {response_json.status_code}")
        print(f"Response headers: {response_json.headers}")
        print(f"Raw response content: {response_json.text}")
        
        if response_json.text.strip():
            try:
                return response_json.json()
            except json.JSONDecodeError:
                print("Could not parse JSON response")
        
        # 2. Form data POST request
        print("\n2. Trying form data POST request...")
        headers_form = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": AUTH_TOKEN
        }
        
        response_form = requests.post(ENDPOINT_URL, data=payload, headers=headers_form, timeout=120)
        print(f"Response status code: {response_form.status_code}")
        print(f"Response headers: {response_form.headers}")
        print(f"Raw response content: {response_form.text}")
        
        if response_form.text.strip():
            try:
                return response_form.json()
            except json.JSONDecodeError:
                print("Could not parse JSON response")
        
        # 3. GET request with query parameters
        print("\n3. Trying GET request with query parameters...")
        response_get = requests.get(ENDPOINT_URL, params=payload, headers=headers_json, timeout=120)
        print(f"Response status code: {response_get.status_code}")
        print(f"Response headers: {response_get.headers}")
        print(f"Raw response content: {response_get.text}")
        
        if response_get.text.strip():
            try:
                return response_get.json()
            except json.JSONDecodeError:
                print("Could not parse JSON response")
        
        print("\nAll request methods failed to return a valid response.")
        return None
    
    except requests.exceptions.RequestException as e:
        print(f"Error calling endpoint: {e}")
        return None

def print_results(results):
    """Pretty print the results from the endpoint."""
    if not results:
        print("No results received.")
        return
    
    if not results.get("success", False):
        print(f"Error: {results.get('error', 'Unknown error')}")
        return
    
    print("\n===== QUERY RESULTS =====\n")
    print(f"Query: {results['query']}")
    print(f"Sources: {', '.join(results['sources'])}")
    
    print("\n===== ANALYSIS =====\n")
    for csv_path, result_data in results.get("results", {}).items():
        print(f"Source: {result_data.get('url', 'Unknown')}")
        print(f"Relevance score: {result_data.get('relevance_score', 'N/A')}")
        print("\nOutput:")
        print(result_data.get("data", {}).get("output", "No output"))
        print("\n" + "-" * 50 + "\n")

def main():
    """Main function to demonstrate endpoint usage."""
    if len(sys.argv) < 2:
        print("Usage: python call_endpoint.py \"Your query here\" [google_sheet_url1] [google_sheet_url2] ...")
        return
    
    query = sys.argv[1]
    google_sheets = sys.argv[2:] if len(sys.argv) > 2 else None
    
    results = call_endpoint(query, google_sheets)
    print_results(results)

if __name__ == "__main__":
    main() 