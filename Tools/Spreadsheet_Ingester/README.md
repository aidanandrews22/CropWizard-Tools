# Spreadsheet Ingester Tool

A tool for ingesting Google Sheets data and formatting it for LLM processing. Includes both a Beam deployment version and a local testing version.

## Local Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Local Usage

Test the functionality locally before deploying:

```bash
python local_ingester.py "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID"
```

The tool will output the sheet data in JSON format, including:
- Column names
- Row data
- Row and column counts
- Success/error status

## Beam Deployment

1. Deploy to Beam:
```bash
beam deploy spreadsheet_ingester.py
```

2. Update `n8n.json` with your:
   - Beam endpoint URL
   - Beam API key

3. Import the `n8n.json` workflow into your n8n instance

## Important Notes

- The Google Sheet must be either public or have appropriate sharing settings
- The sheet URL should be in the format: `https://docs.google.com/spreadsheets/d/[SHEET_ID]`
- The tool exports the sheet as CSV, so some advanced Google Sheets features may not be preserved 