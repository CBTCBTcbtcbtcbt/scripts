import json
import os
from prompt_generator import generate_prompt
from SDtools import load_and_process_depth_image,load_controlnet_model,load_depth_map_directly,load_prompts_from_file,setup_pipeline,setup_pipeline_without_controlnet,generate_images_from_different_prompt
from img_processer import process_image_task,load_prompt_config


def append_result_to_json(result_data, file_path):
    """
    Appends the result data to a JSON list in the specified file.
    Creates the file and directory if they don't exist.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Load existing data if file exists
    existing_data = []
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Ensure it's a list
                existing_data = data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            # If file is empty or invalid JSON, start with empty list
            existing_data = []

    # Append new data
    existing_data.append(result_data)

    # Save back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)
    print(f"Result appended to {file_path}")

def run_image_analysis(image_path, output_json_path):
    """
    Loads prompt config, processes the image, and saves the result to a JSON file.
    """
    # --- Step 1: Generate Instruction ---
    print("--- Step 1: Generating Instruction ---")
    try:
        # Load instruction config
        instruction_schema, instruction_prompt = load_prompt_config('instruction_generater.json')
    except Exception as e:
        print(f"Error loading instruction config: {e}")
        return

    # Process image task to get instruction
    print(f"Processing image for instruction: {image_path}")
    instruction_result = process_image_task(
        image_input=image_path,
        prompt_text=instruction_prompt,
        response_schema=instruction_schema
    )
    
    # Parse result
    if isinstance(instruction_result, str):
        try:
            instruction_data = json.loads(instruction_result)
        except json.JSONDecodeError:
            print("Warning: Instruction result is not valid JSON")
            instruction_data = {"instruction": instruction_result}
    else:
        instruction_data = instruction_result
        
    instruction_text = instruction_data.get('instruction', '')
    print(f"Generated Instruction: {instruction_text}")



    # --- Step 2: Navigation Task ---
    print("\n--- Step 2: Navigation Task ---")
    # Load configuration
    try:
        schema, prompt = load_prompt_config('img_processor.json')
    except Exception as e:
        print(f"Error loading prompt config: {e}")
        return

    # Inject instruction into prompt
    if instruction_text:
        print(f"Injecting instruction: {instruction_text}")
        prompt = prompt.replace("__instruction__", instruction_text)
    else:
        print("Warning: No instruction generated in Step 1. Using default prompt.")

    print(f"Prompt: {prompt}")
    
    # Process image task
    print(f"Processing image for navigation: {image_path}")
    result = process_image_task(
        image_input=image_path,
        prompt_text=prompt,
        response_schema=schema
    )
    print(f"Response: {result}")

    # Ensure result is a dictionary (parse from string if necessary)
    if isinstance(result, str):
        try:
            result_data = json.loads(result)
        except json.JSONDecodeError:
            print("Warning: Returned result is not a valid JSON string. Saving as raw string/content.")
            result_data = result
    else:
        result_data = result

    # Format final output
    formatted_result = {
        "instruction": instruction_text,
        "input": "",
        "output": result_data,
        "images": [image_path]
    }

    # Save result
    append_result_to_json(formatted_result, output_json_path)

if __name__ == "__main__":


    
    batch_size = 2 # 每批次生成多少张图像
    batch_num = 10 # 生成多少批图像
    with_controlnet = False  # 是否使用 ControlNet 进行提示词控制
    already_depth_map = False # 是否直接加载已有的深度图（否则使用预处理器生成）
    control_images = None # 预加载的深度图列表（总共 batch_size * batch_num 张）
    
    # if with_controlnet:
    #     # 1. 一次性加载所有深度图（batch_size * batch_num 张）
    #     if already_depth_map:
    #         # 如果直接加载已有的深度图，使用下面这行：
    #         original_images, control_images = load_depth_map_directly(batch_size, batch_num)
    #     else:
    #         # 如果要使用深度图预处理器，使用下面这行：
    #         original_images, control_images = load_and_process_depth_image(batch_size, batch_num)
        
        
    #     # 2. 加载 ControlNet 模型
    #     controlnet = load_controlnet_model()
        
    #     # 3. 设置 Pipeline
    #     pipe = setup_pipeline(controlnet)
    # else:
    #     pipe = setup_pipeline_without_controlnet()

    # prompt=generate_prompt()
    # print(prompt)
    # positive_prompt = []
    # negative_prompt = []
    # for key, value in prompt.items():
    #     print(f"{key}: {value}")
    #     if key>batch_num:
    #         break
    #     positive_prompt.append(value['positive_prompt'])
    #     negative_prompt.append(value['negative_prompt'])
    # print(f"正向提示词: {positive_prompt}")
    # print(f"反向提示词: {negative_prompt}")
    # # 5. 批量生成图像（从预加载的图像列表中切片使用）
    # generate_images_from_different_prompt(pipe, positive_prompt, negative_prompt, batch_size, batch_num, control_images)
    

    # Configuration
    OUTPUT_JSON_PATH = "data/action_data.json"

    # Execute for all images
    for i in range(1, batch_num + 1):
        for j in range(1, batch_size + 1):
            image_path = f"data/images/{i}_{j}.png"
            if os.path.exists(image_path):
                print(f"\n========== Processing {image_path} ==========")
                run_image_analysis(image_path, OUTPUT_JSON_PATH)
            else:
                print(f"Warning: Image {image_path} not found, skipping.")
