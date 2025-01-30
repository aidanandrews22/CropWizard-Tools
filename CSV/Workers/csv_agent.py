import logging
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('CSV.Analyzer')

class CSVAnalyzer:
    def __init__(self, query, csv_paths):
        """Initialize the CSV analyzer with query and paths.
        
        Args:
            query (str): Natural language query from user
            csv_paths (list): List of tuples (csv_path, score) for CSV files to analyze
        """
        logger.info(f"Initializing CSV Analyzer for files: {[path for path, _ in csv_paths]}")
        self.query = query
        self.csv_paths = csv_paths
        self.dataframes = {}
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
    def _load_data(self):
        """Load CSV data into pandas DataFrames."""
        if not self.dataframes:
            for csv_path, _ in self.csv_paths:
                logger.info(f"Loading CSV data from {csv_path}")
                self.dataframes[csv_path] = pd.read_csv(csv_path)
                logger.info(f"Loaded {len(self.dataframes[csv_path])} rows from {csv_path}")
    
    def _parse_query_to_filters(self, csv_path):
        """Convert natural language query to DataFrame filters for a specific CSV."""
        filters = {}
        numbers = re.findall(r'\d+', self.query)
        
        df = self.dataframes[csv_path]
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in self.query.lower():
                if numbers and df[col].dtype in ['int64', 'float64']:
                    filters[col] = float(numbers[0]) if '.' in numbers[0] else int(numbers[0])
                elif df[col].dtype == 'object':
                    unique_vals = df[col].unique()
                    for val in unique_vals:
                        if str(val).lower() in self.query.lower():
                            filters[col] = val
                            break
        
        return filters
    
    def simple_search(self):
        """Perform simple structured search using pandas for all CSVs."""
        self._load_data()
        results = {}
        
        for csv_path, _ in self.csv_paths:
            filters = self._parse_query_to_filters(csv_path)
            
            if filters:
                result = self.dataframes[csv_path].copy()
                for col, val in filters.items():
                    if isinstance(val, str):
                        result = result[result[col].str.lower() == val.lower()]
                    else:
                        result = result[result[col] == val]
                        
                if not result.empty:
                    results[csv_path] = result
                    
        return results if results else None
    
    def semantic_search(self, max_rows=10):
        """Perform semantic search using pre-computed embeddings for all CSVs.
        
        Args:
            max_rows (int): Maximum number of rows to return per CSV
        """
        self._load_data()
        results = {}
        
        # Encode query once
        query_embedding = self.model.encode([self.query], convert_to_numpy=True)
        
        for csv_path, csv_score in self.csv_paths:
            # Load pre-computed embeddings
            emb_path = csv_path.replace('_CSV', '_Embeddings').replace('.csv', '.npy')
            embeddings = np.load(emb_path)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings.astype(np.float32))
            
            # Search with a larger k to allow for filtering
            distances, indices = index.search(query_embedding.astype(np.float32), min(len(embeddings), max_rows * 2))
            distances = distances[0]  # Flatten distances array
            indices = indices[0]      # Flatten indices array
            
            # Convert distances to similarity scores (closer to 1 is better)
            # FAISS returns L2 distances, so we need to convert them to similarities
            max_distance = np.max(distances)
            similarities = 1 - (distances / max_distance)
            
            # Find the highest similarity score
            top_similarity = similarities[0]
            
            # Calculate dynamic threshold as 75% of top similarity
            threshold = top_similarity * 0.75
            
            # Filter results that meet the threshold
            mask = similarities >= threshold
            filtered_indices = indices[mask][:max_rows]  # Limit to max_rows
            filtered_similarities = similarities[mask][:max_rows]
            
            if len(filtered_indices) > 0:
                # Get results and add similarity scores
                csv_results = self.dataframes[csv_path].iloc[filtered_indices].copy()
                csv_results['similarity_score'] = filtered_similarities
                
                results[csv_path] = {
                    'data': csv_results,
                    'relevance_score': csv_score,
                    'num_results': len(filtered_indices)
                }
        
        return results if results else None
    
    def analyze(self):
        """Main method to analyze all CSVs based on the query.
        
        Returns:
            dict: Results matching the query for each CSV file, with relevance scores
                 and similarity scores for individual rows
        """
        # Try simple search first
        results = self.simple_search()
        
        # If simple search fails, try semantic search
        if results is None:
            results = self.semantic_search()
            
        return results