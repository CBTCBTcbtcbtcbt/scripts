import json
import base64
import os

def encode_image(image_path):
    """
    Reads an image file and converts it to a base64 encoded string.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Warning: Image file not found: {image_path}")
        return None

def convert_data(input_file, output_file):
    print(f"Reading from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found.")
        return

    transformed_data = []

    for item in data:
        # Extract ID from the first image filename (e.g., 'data/images/0001.png' -> '0001')
        if item.get('images'):
            first_image_path = item['images'][0]
            image_filename = os.path.basename(first_image_path)
            image_id = os.path.splitext(image_filename)[0]
        else:
            # Fallback ID if no images present (though unlikely based on schema)
            image_id = "unknown_id"

        # Process images and build user content
        images_base64 = []
        image_placeholders = ""
        
        if 'images' in item:
            for img_path in item['images']:
                # The paths in json are like "data/images/0001.png"
                # Ensure we handle the path correctly relative to current working directory
                b64_str = encode_image(img_path)
                if b64_str:
                    images_base64.append(b64_str)
                    image_placeholders += "<image>"

        # Merge instruction and input for user content
        # instruction + input
        instruction = item.get('instruction', '')
        user_input = item.get('input', '')
        
        # Combine with a newline separator if both exist
        if instruction and user_input:
            user_text = f"{instruction}\n{user_input}"
        else:
            user_text = instruction + user_input
            
        final_user_content = image_placeholders + user_text

        # Assistant content is the JSON string of the 'output' dictionary
        output_data = item.get('output', {})
        assistant_content = json.dumps(output_data, ensure_ascii=False)

        # Construct the conversation object
        conversation_item = {
            "id": image_id,
            "conversations": [
                {
                    "role": "user",
                    "content": final_user_content,
                    "images": images_base64
                },
                {
                    "role": "assistant",
                    "content": assistant_content
                }
            ]
        }
        
        transformed_data.append(conversation_item)

    print(f"Writing {len(transformed_data)} items to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, ensure_ascii=False, indent=2)
    
    print("Conversion finished successfully.")

if __name__ == "__main__":
    # Define input and output paths
    input_path = 'data/action_data.json'
    output_path = 'data/train.json'
    
    convert_data(input_path, output_path)
