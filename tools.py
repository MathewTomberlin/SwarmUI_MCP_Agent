from langchain.tools import StructuredTool
import requests
from pydantic import BaseModel, Field
from typing import Optional
import logging

# Note: This isn't functional, but should be the basis for multiple inputs for image generation.
class GenerateImageInput(BaseModel):
    prompt: str = Field(..., description="The prompt to generate an image from.")
    negative: Optional[str] = Field("", description="Negative prompt to avoid certain features in the image.")
    images: Optional[int] = Field(1, description="Number of images to generate.")
    steps: Optional[int] = Field(50, description="Number of steps for image generation.")
    cfgScale: Optional[float] = Field(3.0, description="Classifier-free guidance scale.")
    seed: Optional[int] = Field(-1, description="Seed for random number generation, -1 for random.")
    model: Optional[str] = Field("hyphoriaIllustrious20_v001", description="Model name to use.")

# Model for the tool's input
class ImageGenerationString(BaseModel):
    input_string: str = Field(..., description="The input string containing either a prompt or JSON parameters")

# Theoretically, this function should parse the input string to extract parameters for image generation
def parse_input(input_str: str) -> dict:
    """Parse the input string to extract parameters."""
    try:
        # If it's a JSON string, parse it
        if input_str.strip().startswith("{"):
            return json.loads(input_str)
        # If it contains "Action Input:", extract the JSON part
        elif "Action Input:" in input_str:
            json_str = input_str.split("Action Input:")[1].strip()
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    # Default to treating the entire input as the prompt
    return {"prompt": input_str}

# This function is doing a lot more than it needs to in an attempt to implement multi-input functionality
# This method is called with the agent generated prompt or JSON input to generate an image via SwarmUI API
def generate_image(input_string: str) -> dict:
    """Generate an image from a text prompt and other optional values using the SwarmUI MCP server."""
    try:
        # Parse the input string into parameters
        params = parse_input(input_string)
        
        # Create GenerateImageInput instance with parsed parameters
        input_model = GenerateImageInput(**params)
        
        # Convert to dict for request
        payload = input_model.dict()
        
        logging.debug(f"Sending payload to server: {payload}")
        
        response = requests.post(
            "http://localhost:5001/generate-image",
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Error processing input: {input_string}")
        logging.error(f"Error details: {str(e)}")
        raise

# Create the StructuredTool for image generation from the generate_image function
generate_image = StructuredTool(
    func=generate_image,
    name="generate_image",
    description="A tool to generate an image from a text prompt using SwarmUI MCP server. Accepts optional negative prompt, steps, cfgScale, model, images, and seed.",
    args_schema=ImageGenerationString,
    return_direct=True
)