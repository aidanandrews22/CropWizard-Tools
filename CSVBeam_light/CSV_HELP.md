# Using CSV Files with the CSV Analyzer

## Uploading CSV Files

You can analyze CSV data in two ways:

1. **Direct CSV URL**: Provide any direct link to a publicly accessible CSV file.
2. **Google Sheets**: Provide a Google Sheets URL (the sheet must be publicly accessible or have view access).

Example:
```txt
    query: "What is the total revenue by product category?
    https://example.com/my-data.csv
    https://docs.google.com/spreadsheets/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/edit"
```

## CSV File Nice to Haves

- Properly formatted with headers in the first row
- Files should be publicly accessible via URL
- Maximum file size: Standard CSV files (up to several MB)
- Column names will be normalized (stripped and converted to lowercase)

## Query Capabilities

You can ask natural language questions about your data, such as:

- Statistical analysis: "What is the average sales by region?"
- Data filtering: "Show me all transactions over $1000"
- Comparisons: "Compare sales performance between Q1 and Q2"
- Correlations: "Is there a correlation between marketing spend and revenue?"
- Aggregations: "What is the total revenue by product category?"

## Examples

### Basic Statistical Analysis

```json
{
  "query": "What is the average order value and how does it vary by customer segment?",
  "google_sheets": ["https://docs.google.com/spreadsheets/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/edit"]
}
```

### Time-Based Analysis

```json
{
  "query": "Show me the monthly sales trend for 2023 and identify the best performing month",
  "google_sheets": ["https://example.com/sales_data_2023.csv"]
}
```

### Multi-Dataset Analysis

```json
{
  "query": "Compare customer acquisition cost across marketing channels and correlate with customer lifetime value",
  "google_sheets": [
    "https://example.com/marketing_costs.csv",
    "https://docs.google.com/spreadsheets/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/edit"
  ]
}
``` 