"""
CSV Analysis Beam Endpoint

To deploy: beam deploy CSVBeam/main.py:main
For testing: beam serve CSVBeam/main.py
"""

import os
import dotenv
import re
import requests
from typing import Any, Dict, List, Tuple, Optional
from urllib.parse import urlparse, parse_qs
import logging

from beam import endpoint, Image

import torch
import pandas as pd
import numpy as np

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

# Workers for CSV analysis
class Find:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import torch
        
        # Force CPU usage instead of MPS
        self.device = torch.device("cpu")
        # Load the model for semantic search
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.descriptions = {}
    
    def add_description(self, csv_path, description):
        """Add a description for a CSV file."""
        self.descriptions[csv_path] = description
    
    def find_relevant_csv(self, query: str) -> list:
        """
        Find the most relevant CSV files based on the user's query using semantic similarity.
        
        Args:
            query (str): The user's query about the data they're looking for
            
        Returns:
            list: List of tuples (csv_path, score) for relevant CSV files
        """
        # Encode the query and ensure it's on CPU
        query_embedding = self.model.encode(query, convert_to_tensor=True).to(self.device)
        
        # Store all scores
        scores = []
        
        for csv_file, description in self.descriptions.items():
            # Split description into chunks of reasonable size
            chunks = [s.strip() for s in description.split('\n\n') if s.strip()]
            
            # Get embeddings for all chunks and ensure they're on CPU
            chunk_embeddings = self.model.encode(chunks, convert_to_tensor=True).to(self.device)
            
            # Calculate similarity with each chunk
            similarities = torch.cosine_similarity(query_embedding.unsqueeze(0), chunk_embeddings)
            
            # Use the highest similarity score for this description
            max_similarity = similarities.max().item()
            scores.append((csv_file, max_similarity))
        
        # Sort scores in descending order
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if not scores:
            return []
            
        # Get the top score
        top_score = scores[0][1]
        
        # Calculate dynamic threshold as 75% of top score
        threshold = top_score * 0.75
        
        # Filter results that meet the threshold, up to 5 results
        relevant_results = [
            (csv_file, score) 
            for csv_file, score in scores 
            if score >= threshold
        ][:5]
        
        return relevant_results if relevant_results else []

class Ingester:
    def __init__(self, temp_dir):
        """
        Initialize Ingester for processing Google Sheets and CSV files
        
        Args:
            temp_dir (str): Directory to store temporary files
        """
        self.temp_dir = temp_dir
        self.processed_files = []
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_model_loaded = True
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            self.embedding_model_loaded = False
        
    def _generate_embeddings(self, df):
        """
        Generate embeddings for each row in the dataframe
        
        Args:
            df (pd.DataFrame): DataFrame to generate embeddings for
            
        Returns:
            np.ndarray: Array of embeddings or None if generation fails
        """
        if not self.embedding_model_loaded:
            logger.warning("Embedding model not loaded, skipping embedding generation")
            return None
            
        try:
            # Limit to a reasonable number of rows for embedding
            max_rows = min(1000, len(df))
            if len(df) > max_rows:
                logger.info(f"Limiting embeddings to {max_rows} rows out of {len(df)}")
            
            row_texts = []
            for _, row in df.head(max_rows).iterrows():
                # Convert all values to strings and handle NaN values
                row_items = []
                for col, val in row.items():
                    if pd.isna(val):
                        continue
                    row_items.append(f"{col}: {val}")
                
                row_text = ", ".join(row_items)
                row_texts.append(row_text)
            
            # Batch process embeddings for efficiency
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(row_texts), batch_size):
                batch = row_texts[i:i+batch_size]
                batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
                all_embeddings.append(batch_embeddings)
            
            return np.vstack(all_embeddings) if all_embeddings else np.array([])
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return None
    
    def describe(self, df, user_description=None):
        """
        Generate a description of the CSV data
        
        Args:
            df (pd.DataFrame): DataFrame to describe
            user_description (str, optional): User-provided description
            
        Returns:
            str: Generated description or error message
        """
        try:
            from langchain_openai import ChatOpenAI
            from langchain.schema import HumanMessage
            
            # Create a sample of the dataframe for description
            # Limit to a reasonable number of rows and columns
            max_rows = min(5, len(df))
            max_cols = min(20, len(df.columns))
            
            if len(df.columns) > max_cols:
                logger.info(f"Limiting description to first {max_cols} columns out of {len(df.columns)}")
                
            sample_df = df.iloc[:max_rows, :max_cols]
            sample = sample_df.to_string()
            
            # Add column data types information
            dtypes_info = "\nColumn Data Types:\n" + "\n".join([f"{col}: {dtype}" for col, dtype in df.dtypes.items()])
            
            # Define the prompt for generating a description
            prompt = f"""
            I have a CSV file with the following structure (showing first {max_rows} rows and {max_cols} columns):
            
            {sample}
            
            {dtypes_info}
            """
            
            # Add user description if available
            if user_description:
                prompt += f"""
            
            USER-PROVIDED DESCRIPTION: {user_description}
            
            Please take the above user-provided description into special consideration as it represents the user's own understanding of this data.
            """
                
            prompt += """
            
            Please provide a detailed description of this dataset that includes:
            1. An overview of what this data represents
            2. A description of each column and what it contains
            3. Any patterns or insights that are immediately visible
            
            Format your response as plain text with sections separated by blank lines.
            """
            
            # Generate the description using GPT with timeout handling
            try:
                llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", request_timeout=30)
                response = llm.invoke([HumanMessage(content=prompt)])
                return response.content
            except Exception as e:
                logger.error(f"Error generating description with LLM: {str(e)}")
                # Fallback to a basic description
                return self._generate_basic_description(df, user_description)
                
        except Exception as e:
            logger.error(f"Error in describe method: {str(e)}")
            return self._generate_basic_description(df, user_description)
    
    def _generate_basic_description(self, df, user_description=None):
        """Generate a basic description without using LLM"""
        description = f"Dataset with {len(df)} rows and {len(df.columns)} columns.\n\n"
        description += f"Columns: {', '.join(df.columns)}\n\n"
        
        if user_description:
            description += f"User description: {user_description}\n\n"
            
        # Add basic stats
        description += "Basic statistics:\n"
        try:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if not numeric_cols.empty:
                stats = df[numeric_cols].describe().to_string()
                description += stats
        except Exception:
            pass
            
        return description
    
    def process_sheets(self, sheet_urls, descriptions=None, convert_gsheet=None):
        """
        Process all sheets and generate descriptions
        
        Args:
            sheet_urls (list): List of URLs to analyze (Google Sheets or CSV files)
            descriptions (list, optional): List of user-provided descriptions that correspond to each URL
            convert_gsheet (callable): Function to convert Google Sheets to CSV
            
        Returns:
            list: List of processed file information
        """
        import hashlib
        
        # Track processing results for reporting
        results = {
            'success': [],
            'failed': []
        }
        
        for idx, url in enumerate(sheet_urls):
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
                
                # Get user description if available
                user_description = None
                if descriptions and idx < len(descriptions):
                    user_description = descriptions[idx]
                
                logger.info(f"Processing URL {idx+1}/{len(sheet_urls)}: {url}")
                
                # Convert URL to CSV
                try:
                    convert_gsheet(url, csv_path)
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
                
                # Generate embeddings
                embeddings = self._generate_embeddings(df)
                
                # Generate description with user input if available
                description = self.describe(df, user_description)
                
                file_info = {
                    'url': url,
                    'csv_path': csv_path,
                    'description': description,
                    'user_description': user_description,
                    'df': df,
                    'embeddings': embeddings,
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
    def __init__(self, query, processed_files, scores=None):
        """
        Initialize the CSV analyzer with query and dataframes.
        
        Args:
            query (str): Natural language query from user
            processed_files (list): List of processed file info including dataframes
            scores (dict, optional): Relevance scores for each file
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
            
        self.scores = scores or {}
        
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
            
            # Create a system prompt that includes user descriptions and dataframe info
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
                    'relevance_score': self.scores.get(csv_path, 0.0),
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
            
            # Add user description if available
            if file_info.get('user_description'):
                system_prompt += f"- User description: {file_info['user_description']}\n"
        
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
                'relevance_score': self.scores.get(csv_path, 1.0),
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
        "numpy",
        "torch",
        "huggingface-hub",
        "sentence-transformers",
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
        descriptions (Optional[List[str]]): List of short descriptions for each URL
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
    
    # Validate descriptions
    descriptions = inputs.get("descriptions", [])
    if not isinstance(descriptions, list):
        descriptions = []
    
    # Create a unique request ID for tracking
    import uuid
    request_id = str(uuid.uuid4())
    logger.info(f"Request ID: {request_id}")
    
    try:
        # Set environment variables for PyTorch
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings
        torch.device("cpu")  # Force CPU usage
        
        # Create temporary directory for files
        import tempfile
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
            # Initialize the Ingester
            ingester = Ingester(temp_dir)
            
            # Process the sheets with optional descriptions
            processed_files = ingester.process_sheets(
                valid_urls, 
                descriptions[:len(valid_urls)], 
                converter.convert_to_csv
            )
            
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
        
        # Find relevant CSV files based on query if multiple files
        try:
            if len(processed_files) > 1:
                find = Find()
                for file_info in processed_files:
                    find.add_description(file_info['csv_path'], file_info['description'])
                
                relevant_csvs = find.find_relevant_csv(query)
                
                if not relevant_csvs:
                    logger.warning("No relevant CSVs found for the query")
                    # Continue with all files if no relevant ones found
                    scores = {file_info['csv_path']: 0.5 for file_info in processed_files}
                else:
                    # Filter processed_files to only include relevant ones
                    relevant_paths = [path for path, _ in relevant_csvs]
                    processed_files = [file_info for file_info in processed_files 
                                    if file_info['csv_path'] in relevant_paths]
                    # Create scores dictionary
                    scores = {path: score for path, score in relevant_csvs}
                    
                    logger.info(f"Found {len(relevant_csvs)} relevant CSVs for the query")
            else:
                scores = {processed_files[0]['csv_path']: 1.0}
                
        except Exception as e:
            logger.error(f"Error finding relevant CSVs: {str(e)}")
            # Continue with all files if relevance scoring fails
            scores = {file_info['csv_path']: 0.5 for file_info in processed_files}
        
        # Initialize the CSV analyzer and run the query
        try:
            csv_analyzer = CSVAnalyzer(query, processed_files, scores)
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
            import shutil
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