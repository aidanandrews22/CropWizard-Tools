import logging
from torch import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import pathlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('CSV.Find')

class Find:
    def __init__(self):
        logger.info("Initializing Find component")
        self.descriptions = {}
        self._load_descriptions()
        logger.info("Loading sentence transformer model")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def _load_descriptions(self):
        """Load all description files from _Descriptions folder and map them to their corresponding CSV files."""
        logger.info("Loading description files from _Descriptions folder")
        descriptions_dir = pathlib.Path("_Descriptions")
        loaded_count = 0
        
        for desc_file in descriptions_dir.glob("*.txt"):
            # Read the description file
            with open(desc_file, "r") as f:
                content = f.read()
                
            # Map the description to its corresponding CSV file
            csv_file = f"_CSV/{desc_file.stem}.csv"
            if os.path.exists(csv_file):
                self.descriptions[csv_file] = content
                loaded_count += 1
                
        logger.info(f"Loaded {loaded_count} description files")
    
    def find_relevant_csv(self, query: str) -> list:
        """
        Find the most relevant CSV files based on the user's query using semantic similarity.
        
        Args:
            query (str): The user's query about the data they're looking for
            
        Returns:
            list: List of tuples (csv_path, score) for relevant CSV files
        """
        logger.info(f"Finding relevant CSV for query: {query}")
        # Encode the query
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Store all scores
        scores = []
        
        for csv_file, description in self.descriptions.items():
            # For each description, we'll compare against its sections
            # Split description into chunks of reasonable size
            chunks = [s.strip() for s in description.split('\n\n') if s.strip()]
            
            # Get embeddings for all chunks
            chunk_embeddings = self.model.encode(chunks, convert_to_tensor=True)
            
            # Calculate similarity with each chunk
            similarities = cosine_similarity(query_embedding.unsqueeze(0), chunk_embeddings)
            
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
    