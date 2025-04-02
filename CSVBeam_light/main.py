"""
CSV Analysis Beam Endpoint

To deploy: beam deploy main.py:main
For testing: beam serve main.py:main
"""

import os
import dotenv
import re
import requests
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs
import logging
import tempfile
import uuid
import shutil

from beam import endpoint, Image

import pandas as pd

dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class URLHandler:
    """Base class for handling different types of URLs"""
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Check if a URL is valid"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    @staticmethod
    def get_url_type(url: str) -> str:
        """Determine the type of URL"""
        if not URLHandler.is_valid_url(url):
            return "invalid"
            
        parsed_url = urlparse(url)
        
        # Check for Google Sheets
        if 'docs.google.com' in parsed_url.netloc and '/spreadsheets/' in parsed_url.path:
            return "google_sheet"
            
        # Check for direct CSV files
        if parsed_url.path.lower().endswith('.csv'):
            return "csv_file"
            
        return "unknown"

class GoogleSheetsConverter:
    def __init__(self):
        self.session = requests.Session()
        # Set reasonable timeouts to prevent hanging
        self.timeout = (5, 30)  # (connect timeout, read timeout)
        
    def extract_sheet_id(self, sheet_url: str) -> str:
        """
        Extract the Google Sheet ID from the URL.
        
        Args:
            sheet_url (str): URL of the Google Sheet
            
        Returns:
            str: The extracted sheet ID
            
        Raises:
            ValueError: If the URL is not a valid Google Sheets URL or sheet ID cannot be extracted
        """
        try:
            parsed_url = urlparse(sheet_url)
            
            # Validate it's a Google Sheets URL
            if 'docs.google.com' not in parsed_url.netloc:
                raise ValueError("Not a valid Google Sheets URL")
            
            # Pattern 1: /spreadsheets/d/{sheet_id}/...
            if '/spreadsheets/d/' in sheet_url:
                sheet_id = sheet_url.split('/spreadsheets/d/')[1].split('/')[0]
                # Remove any query parameters if present
                sheet_id = sheet_id.split('?')[0]
                return sheet_id
                
            # Pattern 2: ?id={sheet_id}
            params = parse_qs(parsed_url.query)
            sheet_id = params.get('id', [None])[0]
            
            # Pattern 3: /ccc?key={sheet_id}
            if not sheet_id and '/ccc' in parsed_url.path:
                sheet_id = params.get('key', [None])[0]
                
            if not sheet_id:
                raise ValueError("Could not extract sheet ID from URL")
                
            return sheet_id
            
        except Exception as e:
            raise ValueError(f"Invalid Google Sheets URL: {str(e)}")

    def convert_to_csv(self, sheet_url: str, output_path: Optional[str] = None) -> str:
        """
        Convert a Google Sheet to CSV format.
        
        Args:
            sheet_url (str): URL of the Google Sheet
            output_path (str, optional): Path to save the CSV file. If None, returns the CSV content as string.
        
        Returns:
            str: CSV content if output_path is None, otherwise saves to file and returns path
            
        Raises:
            Exception: If there's an error converting the Google Sheet to CSV
        """
        try:
            # Validate URL type
            url_type = URLHandler.get_url_type(sheet_url)
            
            if url_type == "invalid":
                raise ValueError("Invalid URL format")
                
            if url_type == "csv_file":
                # Handle direct CSV URLs
                logger.info(f"Downloading CSV directly from: {sheet_url}")
                response = self.session.get(sheet_url, timeout=self.timeout)
                response.raise_for_status()  # Raise exception for HTTP errors
                
                # Check if content type is CSV
                content_type = response.headers.get('Content-Type', '')
                if 'text/csv' not in content_type and 'application/csv' not in content_type:
                    # Try to parse as CSV anyway, but log a warning
                    logger.warning(f"URL content type is not CSV: {content_type}")
                
                df = pd.read_csv(pd.io.common.StringIO(response.text))
                
            elif url_type == "google_sheet":
                # Handle Google Sheets
                sheet_id = self.extract_sheet_id(sheet_url)
                csv_export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                
                logger.info(f"Converting Google Sheet to CSV: {csv_export_url}")
                
                # Use retries for robustness
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = self.session.get(csv_export_url, timeout=self.timeout)
                        response.raise_for_status()
                        df = pd.read_csv(pd.io.common.StringIO(response.text))
                        break
                    except requests.exceptions.RequestException as e:
                        if attempt == max_retries - 1:
                            raise Exception(f"Failed to download Google Sheet after {max_retries} attempts: {str(e)}")
                        logger.warning(f"Retry {attempt+1}/{max_retries} after error: {str(e)}")
                        import time
                        time.sleep(1)  # Wait before retrying
            else:
                raise ValueError(f"Unsupported URL type: {url_type}")
            
            # Save or return the CSV
            if output_path:
                df.to_csv(output_path, index=False)
                return output_path
            else:
                return df.to_csv(index=False)
                
        except pd.errors.ParserError as e:
            raise Exception(f"Error parsing CSV data: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error accessing URL: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing URL: {str(e)}")

class CSVProcessor:
    def __init__(self, temp_dir):
        """
        Initialize CSV Processor for handling Google Sheets and CSV files
        
        Args:
            temp_dir (str): Directory to store temporary files
        """
        self.temp_dir = temp_dir
        self.processed_files = []
    
    def process_urls(self, urls, convert_func):
        """
        Process all URLs and convert them to CSV files
        
        Args:
            urls (list): List of URLs to process (Google Sheets or CSV files)
            convert_func (callable): Function to convert URLs to CSV
            
        Returns:
            list: List of processed file information
        """
        import hashlib
        
        # Track processing results for reporting
        results = {
            'success': [],
            'failed': []
        }
        
        for url in urls:
            try:
                # Validate URL
                if not URLHandler.is_valid_url(url):
                    error_msg = f"Invalid URL format: {url}"
                    logger.error(error_msg)
                    results['failed'].append({'url': url, 'error': error_msg})
                    continue
                
                # Create unique filenames based on URL hash
                url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
                csv_filename = f"sheet_{url_hash}.csv"
                csv_path = os.path.join(self.temp_dir, csv_filename)
                
                logger.info(f"Processing URL: {url}")
                
                # Convert URL to CSV
                try:
                    convert_func(url, csv_path)
                except Exception as e:
                    error_msg = f"Error converting URL to CSV: {str(e)}"
                    logger.error(error_msg)
                    results['failed'].append({'url': url, 'error': error_msg})
                    continue
                
                # Load the CSV with error handling
                try:
                    df = pd.read_csv(csv_path)
                    
                    # Check if dataframe is empty
                    if df.empty:
                        error_msg = "CSV file is empty"
                        logger.error(error_msg)
                        results['failed'].append({'url': url, 'error': error_msg})
                        continue
                        
                    # Normalize column names
                    df.columns = df.columns.str.strip().str.lower()
                except Exception as e:
                    error_msg = f"Error loading CSV file: {str(e)}"
                    logger.error(error_msg)
                    results['failed'].append({'url': url, 'error': error_msg})
                    continue
                
                file_info = {
                    'url': url,
                    'csv_path': csv_path,
                    'df': df,
                    'row_count': len(df),
                    'column_count': len(df.columns)
                }
                
                self.processed_files.append(file_info)
                results['success'].append({'url': url, 'row_count': len(df), 'column_count': len(df.columns)})
                logger.info(f"Successfully processed {url}: {len(df)} rows, {len(df.columns)} columns")
                
            except Exception as e:
                error_msg = f"Unexpected error processing URL {url}: {str(e)}"
                logger.error(error_msg)
                results['failed'].append({'url': url, 'error': error_msg})
                continue
        
        # Log summary
        logger.info(f"Processing complete: {len(results['success'])} succeeded, {len(results['failed'])} failed")
        
        return self.processed_files

class CSVAnalyzer:
    def __init__(self, query, processed_files):
        """
        Initialize the CSV analyzer with query and dataframes.
        
        Args:
            query (str): Natural language query from user
            processed_files (list): List of processed file info including dataframes
        """
        self.query = query
        self.processed_files = processed_files
        
        # Validate processed files
        if not processed_files or not isinstance(processed_files, list):
            raise ValueError("No valid processed files provided")
            
        # Extract dataframes, ensuring they're valid
        self.dfs = []
        for file_info in processed_files:
            if 'df' in file_info and isinstance(file_info['df'], pd.DataFrame) and not file_info['df'].empty:
                self.dfs.append(file_info['df'])
                
        if not self.dfs:
            raise ValueError("No valid dataframes found in processed files")
        
        # Set up logging
        self.verbose = True
    
    def analyze(self):
        """
        Analyze all CSVs based on the query.
        
        Returns:
            dict: Analysis results for each CSV file
            
        Raises:
            ValueError: If there's an error analyzing the CSV data
        """
        from langchain_openai import ChatOpenAI
        from langchain_experimental.agents import create_pandas_dataframe_agent
        
        try:
            # Initialize the LangChain agent with timeout and error handling
            try:
                llm = ChatOpenAI(
                    temperature=0, 
                    model="gpt-4o-mini",
                    request_timeout=60,  # Longer timeout for complex analysis
                    max_retries=2
                )
                
                # Create a more robust agent
                agent = create_pandas_dataframe_agent(
                    llm=llm,
                    df=self.dfs,
                    agent_type="tool-calling",
                    verbose=self.verbose,
                    allow_dangerous_code=True,  # More restrictive for production
                    max_iterations=10,  # Limit iterations for safety
                    handle_parsing_errors=True
                )
            except Exception as e:
                logger.error(f"Error initializing LangChain agent: {str(e)}")
                raise ValueError(f"Failed to initialize analysis agent: {str(e)}")
            
            # Create a system prompt that includes dataframe info
            system_prompt = self._build_system_prompt()
            
            # Run the query through the agent with timeout handling
            try:
                logger.info(f"Running analysis with query: {self.query}")
                result = agent.invoke({"input": system_prompt})
                
                # Extract the output from the result
                if isinstance(result, dict) and "output" in result:
                    analysis_result = result["output"]
                else:
                    analysis_result = str(result)
                    
                logger.info("Analysis completed successfully")
                
            except Exception as e:
                logger.error(f"Error during analysis: {str(e)}")
                raise ValueError(f"Analysis failed: {str(e)}")
            
            # Format the results
            formatted_results = self._format_results(analysis_result)
            return formatted_results
            
        except Exception as e:
            error_message = f"Error analyzing CSV data: {str(e)}"
            logger.error(error_message)
            
            # Return error information in a structured format
            error_results = {}
            for idx, file_info in enumerate(self.processed_files):
                csv_path = file_info['csv_path']
                error_results[csv_path] = {
                    'data': {
                        'input': self.query,
                        'output': f"Error: {error_message}"
                    },
                    'error': str(e),
                    'query': self.query,
                    'url': file_info.get('url', 'Local CSV')
                }
            
            return error_results
    
    def _build_system_prompt(self):
        """Build a comprehensive system prompt with context about the data"""
        # Start with the user query
        system_prompt = f"QUERY: {self.query}\n\n"
        
        # Add information about each dataframe
        system_prompt += "AVAILABLE DATA SOURCES:\n"
        for idx, file_info in enumerate(self.processed_files):
            df = file_info.get('df')
            if not isinstance(df, pd.DataFrame):
                continue
                
            system_prompt += f"\nDataFrame {idx+1}:\n"
            system_prompt += f"- Source: {file_info.get('url', 'Unknown')}\n"
            system_prompt += f"- Rows: {len(df)}, Columns: {len(df.columns)}\n"
            system_prompt += f"- Columns: {', '.join(df.columns.tolist())}\n"
        
        # Add instructions for the analysis
        system_prompt += "\nINSTRUCTIONS:\n"
        system_prompt += "1. Analyze the provided data to answer the query\n"
        system_prompt += "2. If multiple dataframes are provided, determine which are relevant\n"
        system_prompt += "3. Provide a clear, concise answer with supporting data\n"
        system_prompt += "4. Include relevant statistics or calculations\n"
        system_prompt += "5. If the query cannot be answered with the available data, explain why\n"
        
        return system_prompt
    
    def _format_results(self, analysis_result):
        """Format the analysis results for the API response"""
        formatted_results = {}
        
        for idx, file_info in enumerate(self.processed_files):
            csv_path = file_info['csv_path']
            
            formatted_results[csv_path] = {
                'data': {
                    'input': self.query,
                    'output': analysis_result
                },
                'query': self.query,
                'url': file_info.get('url', 'Local CSV'),
                'row_count': file_info.get('row_count', len(file_info.get('df', pd.DataFrame()))),
                'column_count': file_info.get('column_count', len(file_info.get('df', pd.DataFrame()).columns))
            }
        
        return formatted_results

@endpoint(
    name="csv_analyzer",
    cpu=1,
    memory="2Gi",
    image=Image(python_version="python3.12").add_python_packages([
        "pandas",
        "langchain",
        "langchain-openai",
        "langchain-experimental",
        "requests",
        "openai",
        "tabulate", 
    ]),
    timeout=120,
    keep_warm_seconds=60 * 3
)
def main(**inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    CSV Analysis Endpoint
    
    Analyzes CSV data from Google Sheets or direct CSV URLs based on a user query.
    
    Args:
        google_sheets (Optional[List[str]]): List of Google Sheet or CSV URLs to analyze
        query (str): Natural language query to run against the CSV data
        
    Returns:
        Dict: Analysis results or error information
    """
    # Start timing the request
    import time
    start_time = time.time()
    
    # Set up logging
    logger.info(f"Starting CSV analysis with inputs: {inputs}")
    
    # Input validation
    query = inputs.get("query")
    if not query or not isinstance(query, str) or len(query.strip()) == 0:
        return {
            "success": False,
            "error": "A valid query parameter is required",
            "status_code": 400
        }
    
    # Normalize and validate URLs
    google_sheets = inputs.get("google_sheets", [])
    if not isinstance(google_sheets, list):
        return {
            "success": False,
            "error": "google_sheets must be a list of URLs",
            "status_code": 400
        }
    
    # Create a unique request ID for tracking
    request_id = str(uuid.uuid4())
    logger.info(f"Request ID: {request_id}")
    
    try:
        # Create temporary directory for files
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {temp_dir}")
        
        # Initialize the converter
        converter = GoogleSheetsConverter()
        
        # Validate URLs before processing
        valid_urls = []
        invalid_urls = []
        
        for url in google_sheets:
            if not isinstance(url, str):
                invalid_urls.append({"url": str(url), "error": "URL must be a string"})
                continue
                
            url_type = URLHandler.get_url_type(url)
            if url_type == "invalid":
                invalid_urls.append({"url": url, "error": "Invalid URL format"})
            elif url_type == "unknown":
                invalid_urls.append({"url": url, "error": "URL is not a recognized Google Sheet or CSV file"})
            else:
                valid_urls.append(url)
        
        if not valid_urls:
            return {
                "success": False,
                "error": "No valid URLs provided",
                "invalid_urls": invalid_urls,
                "status_code": 400
            }
        
        # Process the valid URLs
        try:
            # Initialize the CSV Processor
            processor = CSVProcessor(temp_dir)
            
            # Process the URLs
            processed_files = processor.process_urls(valid_urls, converter.convert_to_csv)
            
            if not processed_files:
                return {
                    "success": False,
                    "error": "Failed to process any URLs",
                    "invalid_urls": invalid_urls,
                    "status_code": 500
                }
                
            logger.info(f"Successfully processed {len(processed_files)} files")
            
        except Exception as e:
            logger.error(f"Error processing URLs: {str(e)}")
            return {
                "success": False,
                "error": f"Error processing URLs: {str(e)}",
                "invalid_urls": invalid_urls,
                "status_code": 500
            }
        
        # Initialize the CSV analyzer and run the query
        try:
            csv_analyzer = CSVAnalyzer(query, processed_files)
            results = csv_analyzer.analyze()
            
            logger.info("Analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error during CSV analysis: {str(e)}")
            return {
                "success": False,
                "error": f"Error analyzing CSV data: {str(e)}",
                "status_code": 500
            }
        
        # Clean up temporary files 
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary directory: {str(e)}")
        
        # Calculate execution time
        execution_time = time.time() - start_time
        logger.info(f"Request completed in {execution_time:.2f} seconds")
        
        # Format results for the API response
        response = {
            "success": True,
            "results": results,
            "query": query,
            "sources": [file_info['url'] for file_info in processed_files],
            "invalid_urls": invalid_urls if invalid_urls else None,
            "execution_time": f"{execution_time:.2f} seconds",
            "request_id": request_id
        }
        
        return response
        
    except Exception as e:
        # Catch-all error handler
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Unexpected error: {str(e)}\n{error_trace}")
        
        return {
            "success": False,
            "error": str(e),
            "traceback": error_trace,
            "request_id": request_id,
            "status_code": 500
        } 