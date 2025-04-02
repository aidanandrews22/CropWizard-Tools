import dotenv
import torch
import os

from Workers.csv_agent import CSVAnalyzer
from Workers.find import Find
from Workers.ingestion import Ingester
find = Find()

# Disable MPS acceleration to avoid Metal framework crashes
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Force CPU usage instead of MPS
device = torch.device("cpu")

dotenv.load_dotenv()

# Google Sheets example
print("Enter Google Sheet URLs (one per line, press Enter twice when done):")
sheet_urls = []
while True:
    url = input()
    if not url:
        break
    sheet_urls.append(url)

# Process multiple sheets
if sheet_urls:
    ingester = Ingester(sheet_urls)
    processed_files = ingester.process_sheets()

while True:
    print("\n\nEnter your query: ")
    user_query = input()
    
    if not user_query:
        break
    
    csv_files = find.find_relevant_csv(user_query)  # This returns list of (path, score) tuples
    
    csv_analyzer = CSVAnalyzer(user_query, csv_files)
    results = csv_analyzer.analyze()
    print(f"\n\nExtracted results: {results}")
    

# Print summary of processed files
print("\nProcessed files:")
for file_info in processed_files:
    print(f"\nGoogle Sheet: {file_info['url']}")
    print(f"CSV saved to: {file_info['csv_path']}")
    print(f"Description saved to: {file_info['description_path']}")