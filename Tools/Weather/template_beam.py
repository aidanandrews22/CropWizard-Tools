"""
To deploy: beam deploy template_beam.py:main
For testing: beam serve template_beam.py
"""

from beam import endpoint, Image
from typing import Any, Dict

@endpoint(
    name="template_tool",
    cpu=1,
    memory="2Gi",
    image=Image(python_version="python3.10").add_python_packages([
        "requests==2.31.0"
    ]),
    timeout=30,
    keep_warm_seconds=60 * 3
)
def main(**inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    API endpoint template.
    Takes an integer parameter and returns "Hello world!".
    
    Example input:
    {
        "number": 42
    }
    """
    number = inputs.get("number")
    
    if number is None:
        return {
            "success": False,
            "error": "No number provided. Please provide an integer parameter."
        }
    
    try:
        number = int(number)
    except ValueError:
        return {
            "success": False,
            "error": "The provided value is not a valid integer."
        }
    
    return {
        "success": True,
        "message": "Hello world!",
        "input_received": number
    } 