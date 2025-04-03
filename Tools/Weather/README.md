# Tool Templates

This repository contains template files for creating tools that can be deployed as Beam endpoints and connected to n8n workflows.

## Files

- `template_beam.py`: Template for creating a Python-based Beam endpoint
- `template_n8n.json`: Template for creating an n8n workflow that connects to your Beam endpoint

## How to Use

### Beam Template

1. Copy `template_beam.py` and rename it to match your tool's functionality
2. Modify the endpoint parameters as needed
3. Update the input parameters and processing logic
4. Deploy using the command: `beam deploy your_file.py:main`

### n8n Template

1. Copy `template_n8n.json` and rename it
2. Update the form fields to match your tool's input parameters
3. Change the webhook URL to your deployed Beam endpoint
4. Replace the authorization token with your actual Beam API token
5. Import the JSON into n8n

## Template Structure

### Beam Python Template

The Beam template follows this structure:
- Endpoint decorator with resource specifications
- Main function that receives input parameters
- Input validation
- Processing logic
- Structured response with success/error handling

### n8n JSON Template

The n8n template includes:
- Form trigger node for user input
- HTTP request node to connect to your Beam endpoint
- Connection configuration between nodes

## Example

The templates demonstrate a simple tool that:
1. Takes a number as input
2. Returns "Hello world!" and the input number as a response

Customize these templates to create more complex tools with different parameters and functionality. 