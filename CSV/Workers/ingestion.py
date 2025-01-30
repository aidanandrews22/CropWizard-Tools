import logging
import pandas as pd
from test_llm import llm
from Converters.google_sheets_converter import convert_to_csv as convert_gsheet
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('CSV.Ingester')

class Ingester:
    def __init__(self, sheet_urls):
        """
        Initialize Ingester with a list of Google Sheet URLs
        Args:
            sheet_urls (list): List of Google Sheet URLs to process
        """
        logger.info(f"Initializing Ingester for {len(sheet_urls)} sheets")
        self.sheet_urls = sheet_urls if isinstance(sheet_urls, list) else [sheet_urls]
        self.test_llm = llm()
        self.processed_files = []
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Create directories if they don't exist
        for dir_name in ['_CSV', '_Descriptions', '_Embeddings']:
            os.makedirs(dir_name, exist_ok=True)
        
    def _generate_embeddings(self, df):
        """Generate embeddings for each row in the dataframe"""
        row_texts = []
        for _, row in df.iterrows():
            row_text = ", ".join([f"{col}: {val}" for col, val in row.items()])
            row_texts.append(row_text)
        
        return self.model.encode(row_texts, convert_to_numpy=True)
        
    def process_sheets(self):
        """Process all sheets and generate descriptions"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for idx, url in enumerate(self.sheet_urls, 1):
            try:
                # Create unique filenames for each sheet
                csv_filename = f"sheet_{timestamp}_{idx}.csv"
                desc_filename = f"sheet_{timestamp}_{idx}.txt"
                emb_filename = f"sheet_{timestamp}_{idx}.npy"
                
                csv_path = os.path.join("_CSV", csv_filename)
                desc_path = os.path.join("_Descriptions", desc_filename)
                emb_path = os.path.join("_Embeddings", emb_filename)
                
                # Convert Google Sheet to CSV
                logger.info(f"Converting sheet {idx} to CSV")
                convert_gsheet(url, csv_path)
                
                # Load the CSV and generate description
                logger.info(f"Loading CSV data for sheet {idx}")
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.lower()
                
                # Generate and save embeddings
                logger.info(f"Generating embeddings for sheet {idx}")
                embeddings = self._generate_embeddings(df)
                np.save(emb_path, embeddings)
                
                logger.info(f"Generating description for sheet {idx}")
                description = self.test_llm.describe(df=df)
                
                # Save description
                with open(desc_path, "w") as f:
                    f.write(description)
                
                self.processed_files.append({
                    'url': url,
                    'csv_path': csv_path,
                    'description_path': desc_path,
                    'embedding_path': emb_path
                })
                
                logger.info(f"Successfully processed sheet {idx}")
                
            except Exception as e:
                logger.error(f"Error processing sheet {idx}: {str(e)}")
                continue
        
        return self.processed_files

    def get_processed_files(self):
        """Return information about all processed files"""
        return self.processed_files
