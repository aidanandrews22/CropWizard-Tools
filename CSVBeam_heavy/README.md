# CSV Analyzer Beam Endpoint

A Beam endpoint for analyzing CSV data and Google Sheets using natural language queries.

## Description

This endpoint allows you to:
- Process Google Sheets documents by URL
- Run natural language queries against the data 
- Get AI-powered analysis of the data

The endpoint uses:
- Sentence transformers for semantic search across multiple datasets
- LangChain for question-answering over CSV data
- Google Sheets API for importing data directly from Google Sheets

## Endpoint Parameters

The endpoint accepts the following parameters:

- **query** (required): A natural language query to run against the CSV data.
- **google_sheets** (optional): An array of Google Sheet URLs to analyze.

## Example Usage

```json
{
  "query": "What is the average sales by region?",
  "google_sheets": [
    "https://docs.google.com/spreadsheets/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/edit"
  ]
}
```

## Response Format

The endpoint returns a JSON response with the following structure:

```json
{
  "success": true,
  "results": {
    "/tmp/sheet_1a2b3c4d5e.csv": {
      "data": {
        "input": "What is the average sales by region?",
        "output": "The average sales by region are as follows:\n- North: $45,678\n- South: $34,567\n- East: $56,789\n- West: $67,890"
      },
      "relevance_score": 0.87,
      "query": "What is the average sales by region?",
      "url": "https://docs.google.com/spreadsheets/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/edit"
    }
  },
  "query": "What is the average sales by region?",
  "sources": ["https://docs.google.com/spreadsheets/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/edit"]
}
```

## Deployment

To deploy this endpoint, run:

```bash
beam deploy CSVBeam/main.py:main
```

For local testing:

```bash
beam serve CSVBeam/main.py
```

## Environment Variables

This endpoint requires Google Sheets API credentials to be set as environment variables:

- `GOOGLE_PROJECT_ID`
- `GOOGLE_PRIVATE_KEY_ID`
- `GOOGLE_PRIVATE_KEY`
- `GOOGLE_CLIENT_EMAIL`
- `GOOGLE_CLIENT_ID` 
- `GOOGLE_CLIENT_X509_CERT_URL`

Additionally, you'll need to set the OpenAI API key for LangChain:

- `OPENAI_API_KEY` 