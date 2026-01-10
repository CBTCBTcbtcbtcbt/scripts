"""
Multimodal example using Google Gemini with Inline Data (Images).
Refactored to be a reusable function.
"""
import sys
import os
import json
import yaml
import traceback
from PIL import Image

# Add parent directory to path to import llm_client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LLM.llm_client import LLMClient
from LLM import load_config

def process_image_task(
    image_input,
    prompt_text,
    response_schema,
):
    """
    Process a single image task with the LLM.

    Args:
        image_input (str | PIL.Image.Image): File path to the image or a PIL Image object.
        prompt_text (str): The prompt instructions (system prompt/context) to accompany the image.
        response_schema (dict): The JSON schema defining the expected output structure.
        provider (str, optional): The provider name. Defaults to "google".

    Returns:
        dict: The structured response from the LLM matching response_schema.
    """
    
    # Load config if parameters are missing

    config_path = './config.yaml'
    # Use absolute path if needed, or rely on CWD
    if not os.path.exists(config_path):
        # Try finding it relative to script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_path = os.path.join(script_dir, 'config.yaml')
        if os.path.exists(possible_path):
            config_path = possible_path
    
    if os.path.exists(config_path):
        config = load_config(config_path)
        api_config = config.get('api', {})
        api_key = api_config.get('api_key', 'YOUR_API_KEY')
        base_url = api_config.get('base_url', 'YOUR_BASE_URL')
        model = api_config.get('model', 'gemini-1.5-flash')
        provider = api_config.get('provider', 'google')
    else:
        print("Warning: config.yaml not found, using placeholder values.")
        api_key = 'YOUR_API_KEY'
        base_url = 'YOUR_BASE_URL'
        model = 'gemini-1.5-flash'
        provider = 'google'

    # Handle image input (load if path, else use as is)
    if isinstance(image_input, str):
        print(f"Loading image from: {image_input}")
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found at {image_input}")
        img = Image.open(image_input)
    else:
        img = image_input

    print(f"Initializing Client with provider='{provider}'...")
    try:
        client = LLMClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            provider=provider
        )

        # Construct multimodal message
        message_content = [
            img,
            prompt_text
        ]

        messages = [
            {"role": "user", "content": message_content}
        ]

        print("\nSending multimodal request with JSON schema enforcement...")
        response = client.chat(
            messages, 
            response_mime_type="application/json",
            response_schema=response_schema
        )
        return response

    except Exception as e:
        print(f"\n[ERROR] Request failed: {e}")
        traceback.print_exc()
        raise e

def load_prompt_config(json_path='img_processor.json'):
    """Load schema and prompt from a JSON file."""
    # Use absolute path if needed, or rely on CWD
    if not os.path.exists(json_path):
        # Try finding it relative to script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_path = os.path.join(script_dir, json_path)
        if os.path.exists(possible_path):
            json_path = possible_path
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Config file {json_path} not found.")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('schema'), data.get('prompt')

def main():
    # Load schema and prompt from JSON config
    try:
        schema, prompt = load_prompt_config()
        if not schema or not prompt:
            print("Error: Failed to load schema or prompt from img_processor.json")
            return
    except Exception as e:
        print(f"Error loading prompt config: {e}")
        return

    # Set image path
    image_path = "data/images/1_1.png"
    
    # Call the processing function
    try:
        result = process_image_task(
            image_input=image_path,
            prompt_text=prompt,
            response_schema=schema
        )
        print(f"Response: {result}")
    except Exception:
        print("Failed to process image.")

if __name__ == "__main__":
    main()
