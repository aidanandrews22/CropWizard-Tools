"""
Template Beam Endpoint

To deploy: beam deploy main.py:main
For testing: beam serve main.py:main
"""

from beam import endpoint, Image
from typing import Any, Dict, List, Optional

import dotenv
dotenv.load_dotenv()

class TemplateTool:
    def __init__(self):
        self.tmp = "Hello"

    def template_function(self, tmp: str) -> str:
        """
        Template function description
        
        Args:
            tmp (str): Template function argument
            
        Returns:
            str: Template function return
            
        Raises:
            ValueError: Template function error
        """
        return "Hello World!"
    
@endpoint(
    name="template",
    cpu=1,
    memory="2Gi",
    image=Image(python_version="python3.12").add_python_packages([
        "package1",
        "package2",
        "package3",
    ]),
    timeout=120,
    keep_warm_seconds=60 * 3
)

def main() -> str:
    """
    Template Endpoint

    Template description
        
    Args:
        None
        
    Returns:
        String: output
    """
    return "Hello World!"