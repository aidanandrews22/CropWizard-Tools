import logging
import os
import dotenv
import pandas as pd

from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI

from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.agents import create_pandas_dataframe_agent


# Set tokenizers parallelism to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

dotenv.load_dotenv()

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
        # Extract just the file paths from csv_paths tuples
        self.csv_files = [path for path, _ in csv_paths]
        self.dfs = [pd.read_csv(path) for path in self.csv_files]
        self.scores = {path: score for path, score in csv_paths}
        self.agent = None
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the LangChain CSV agent."""
        try:
            self.agent = create_pandas_dataframe_agent(
                ChatOpenAI(temperature=0, model="gpt-4o-mini"),
                *self.dfs,
                agent_type="openai-tools",
                verbose=True,
                handle_parsing_errors=True,
                allow_dangerous_code=True
            )

            ### This is the old agent that ended up not working well. It would generate inconsistent code and not work. Kept getting this error "NameError: name 'df' is not defined"
            # self.agent = create_csv_agent(
            #     ChatOpenAI(temperature=0, model="gpt-4o-mini"),
            #     self.csv_files,  # Pass list of CSV files directly
            #     verbose=True,
            #     agent_type=AgentType.OPENAI_FUNCTIONS,
            #     handle_parsing_errors=True,
            #     allow_dangerous_code=True
            # )
            logger.info("Successfully initialized CSV agent")
        except Exception as e:
            logger.error(f"Error initializing CSV agent: {str(e)}")
            raise
    
    def analyze(self):
        """Main method to analyze all CSVs based on the query.
        
        Returns:
            dict: Results from the CSV agent's analysis
        """
        try:
            if not self.agent:
                raise ValueError("CSV agent not initialized")
            
            # Run the query through the agent using invoke()
            result = self.agent.invoke({"input": self.query})["output"]
            logger.info("Successfully executed query through CSV agent")
            
            # Format the results
            formatted_results = {}
            for csv_path in self.csv_files:
                formatted_results[csv_path] = {
                    'data': {
                        'input': self.query,
                        'output': result
                    },
                    'relevance_score': self.scores[csv_path],
                    'query': self.query
                }
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error analyzing CSV data: {str(e)}")
            raise