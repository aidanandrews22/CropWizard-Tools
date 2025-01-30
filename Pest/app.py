from beam import endpoint, Image
import requests
from PIL import Image as PILImage
from io import BytesIO
import base64
import json

@endpoint(
    name="inference",
    cpu=1,
    memory="1Gi",
    image=Image().add_python_packages(["pillow", "requests"]),
)
def predict(**inputs):
    # Get image URL from input - handle both string and array formats
    image_urls = inputs.get("image_urls")
    
    # If no image URLs provided, return error
    if not image_urls:
        return {"error": "No images provided"}
    
    # If it's a string and not already a JSON array, treat it as a single URL
    if isinstance(image_urls, str):
        if image_urls.startswith("["):
            try:
                image_urls = json.loads(image_urls)
            except json.JSONDecodeError:
                image_urls = [image_urls]
        else:
            image_urls = [image_urls]
    
    # Process the first image
    url = image_urls[0]
    
    # Download the image
    response = requests.get(url)
    img = PILImage.open(BytesIO(response.content))
    
    # Flip the image horizontally
    flipped_img = img.transpose(PILImage.FLIP_LEFT_RIGHT)
    
    # Save the flipped image to bytes
    buffer = BytesIO()
    flipped_img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    # Return the base64 encoded image
    return {
        "image_urls": [f"data:image/png;base64,{img_str}"]
    }