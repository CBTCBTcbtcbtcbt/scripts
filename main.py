import json
import os
import base64
from prompt_generator import generate_prompt
from SDtools import load_and_process_depth_image,load_controlnet_model,load_depth_map_directly,load_prompts_from_file,setup_pipeline,setup_pipeline_without_controlnet,generate_images_from_different_prompt
from img_processer import process_image_task,load_prompt_config
from pathlib import Path
import re

# 配置：是否将图片转换为 base64 编码
USEBASE64 = False

def image_to_base64(image_path):
    """
    将图片文件转换为 base64 编码的字符串
    
    Args:
        image_path: 图片文件路径
    
    Returns:
        str: base64 编码的图片字符串（带有 data URI 前缀）
    """
    # 根据文件扩展名确定 MIME 类型
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp'
    }
    mime_type = mime_types.get(ext, 'image/png')
    
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    base64_str = base64.b64encode(image_data).decode('utf-8')
    return f"{base64_str}"

def smart_rename(target_dir):
    dir_path = Path(target_dir)
    if not dir_path.exists():
        print(f"错误：目录 {target_dir} 不存在")
        return

    # 1. 获取所有文件并分类
    # 假设规范格式是 4 位数字加上原后缀，如 0001.jpg
    pattern = re.compile(r"^\d{4}$")
    
    all_files = [f for f in dir_path.iterdir() if f.is_file()]
    valid_files = []      # 符合规范的文件
    invalid_files = []    # 不符合规范的文件
    existing_indices = set() # 已经存在的编号

    for f in all_files:
        # 去掉后缀拿文件名进行匹配
        if pattern.match(f.stem):
            valid_files.append(f)
            existing_indices.add(int(f.stem))
        else:
            invalid_files.append(f)

    if not invalid_files:
        print("没有发现不符合规范的文件。")
        return

    print(f"找到 {len(invalid_files)} 个不规范文件，准备开始重命名...")

    # 2. 寻找空位并重命名
    current_idx = 1
    renamed_count = 0

    for file_to_rename in invalid_files:
        # 寻找下一个可用的编号（不在 existing_indices 中）
        while current_idx in existing_indices:
            current_idx += 1
        
        # 格式化为 4 位数字
        new_name = f"{current_idx:04d}{file_to_rename.suffix}"
        new_path = dir_path / new_name
        
        # 防止极端情况下文件冲突（虽然逻辑上已规避）
        if not new_path.exists():
            file_to_rename.rename(new_path)
            existing_indices.add(current_idx) # 标记该位置已被占用
            print(f"已重命名: {file_to_rename.name} -> {new_name}")
            renamed_count += 1
        else:
            print(f"跳过: {new_name} 已存在（无法覆盖）")

    print(f"\n任务完成！共重命名了 {renamed_count} 个文件。")


def is_image_processed(image_path, json_file_path):
    """
    检查指定的图片路径是否已经在 JSON 文件中处理过
    
    Args:
        image_path: 图片路径（字符串）
        json_file_path: JSON 文件路径
    
    Returns:
        bool: 如果图片已处理返回 True，否则返回 False
    """
    if not os.path.exists(json_file_path):
        return False
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 确保数据是列表格式
            if not isinstance(data, list):
                return False
            
            # 遍历所有记录，检查 images 字段
            for record in data:
                if 'images' in record and isinstance(record['images'], list):
                    # 检查当前图片路径是否在 images 列表中
                    if image_path in record['images']:
                        return True
            
            return False
    except (json.JSONDecodeError, IOError):
        return False


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
        instruction_schema, instruction_prompt = load_prompt_config('instruction_generater.yaml')
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
        schema, prompt = load_prompt_config('img_processor.yaml')
    except Exception as e:
        print(f"Error loading prompt config: {e}")
        return

    # Inject instruction into prompt
    if instruction_text:
        print(f"Instruction: {instruction_text}")
        prompt_with_instruction = prompt + f"\n**(指令)Instruction**: {instruction_text}\n"
    else:
        print("Warning: No instruction generated in Step 1. Using default prompt.")
        prompt_with_instruction = prompt

    
    # Process image task
    print(f"Processing image for navigation: {image_path}")
    result = process_image_task(
        image_input=image_path,
        prompt_text=prompt_with_instruction,
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
    # 根据 USEBASE64 配置决定 images 字段存储路径还是 base64 编码
    if USEBASE64:
        images_value = [image_to_base64(image_path)]
    else:
        images_value = [image_path]
    
    formatted_result = {
        "instruction": prompt,
        "input": instruction_text,
        "output": result_data,
        "images": images_value,
    }

    # Save result
    append_result_to_json(formatted_result, output_json_path)

if __name__ == "__main__":


    batch_size = 2 # 每批次生成多少张图像
    batch_num = 20 # 生成多少批图像
    with_controlnet = False  # 是否使用 ControlNet 进行提示词控制
    already_depth_map = False # 是否直接加载已有的深度图（否则使用预处理器生成）
    control_images = None # 预加载的深度图列表（总共 batch_size * batch_num 张）
    output_dir = "data/images"  # 输出图像的目录

    prompt=generate_prompt()
    print(prompt)
    SDprompt_path = "SDprompt.json"
    # 保存 prompt 到 SDprompt.json 文件
    with open('SDprompt_path.json', 'w', encoding='utf-8') as f:
        json.dump(prompt, f, ensure_ascii=False, indent=4)
    print("Prompt 已保存到 SDprompt.json")

    #读取prompt,如果先统一生成并储存prompt，就用以下读取储存的prompt
    # try:
    #     with open(SDprompt_path, 'r', encoding='utf-8') as f:
    #         # 3. 将 JSON 字符串解析为 Python 字典/列表
    #         prompts = json.load(f)
    # except FileNotFoundError:
    #     print(f"错误：找不到文件 {SDprompt_path}")
    # except json.JSONDecodeError:
    #     print(f"错误：{SDprompt_path} 格式不正确，无法解析。")
    #     print(f"成功从 {SDprompt_path} 读取数据。")

    positive_prompt = []
    negative_prompt = []
    for key, value in prompt.items():
        print(f"{key}: {value}")
        if key>batch_num:
            break
        positive_prompt.append(value['positive_prompt'])
        negative_prompt.append(value['negative_prompt'])
    print(f"正向提示词: {positive_prompt}")
    print(f"反向提示词: {negative_prompt}")

    if with_controlnet:
        # 1. 一次性加载所有深度图（batch_size * batch_num 张）
        if already_depth_map:
            # 如果直接加载已有的深度图，使用下面这行：
            original_images, control_images = load_depth_map_directly(batch_size, batch_num)
        else:
            # 如果要使用深度图预处理器，使用下面这行：
            original_images, control_images = load_and_process_depth_image(batch_size, batch_num)
        
        
        # 2. 加载 ControlNet 模型
        controlnet = load_controlnet_model()
        
        # 3. 设置 Pipeline
        pipe = setup_pipeline(controlnet)
    else:
        pipe = setup_pipeline_without_controlnet()

    # 5. 批量生成图像（从预加载的图像列表中切片使用）
    generate_images_from_different_prompt(pipe, positive_prompt, negative_prompt, batch_size, batch_num,output_directory=output_dir, control_images_list=control_images)
    
    # # 重命名目录中的图片文件
    # smart_rename('data/images')

    # # 确保目录存在
    # Path('data').mkdir(parents=True, exist_ok=True)

    # # Configuration
    # OUTPUT_JSON_PATH = "data/action_data.json"
    # IMAGES_DIR = Path("data/images")  # 统一使用 Path 对象

    # # 支持的图片格式 (set 查找速度更快)
    # IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif'}

    # # Execute for all images in the directory
    # if IMAGES_DIR.exists():
    #     # 获取目录中所有图片文件
    #     image_files = [
    #         f for f in IMAGES_DIR.iterdir() 
    #         if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    #     ]
    #     # 按文件名排序
    #     image_files.sort(key=lambda x: x.name)
    #     print(f"找到 {len(image_files)} 张图片文件")
        
    #     processed_count = 0
    #     skipped_count = 0
        
    #     for image_file in image_files:
    #         # image_file 本身就是 Path 对象，使用 as_posix() 强制输出 / 斜杠
    #         image_path_str = image_file.as_posix()
            
    #         # 检查图片是否已经处理过
    #         if is_image_processed(image_path_str, OUTPUT_JSON_PATH):
    #             print(f"\n[跳过] {image_path_str} - 该图片已处理过")
    #             skipped_count += 1
    #             continue
            
    #         print(f"\n========== Processing {image_path_str} ==========")
            
    #         # 传入 run_image_analysis 时，如果函数接受字符串，则传字符串
    #         run_image_analysis(image_path_str, OUTPUT_JSON_PATH)
    #         processed_count += 1
        
    #     print(f"\n======== 处理完成 ========")
    #     print(f"总图片数: {len(image_files)}")
    #     print(f"已处理: {processed_count}")
    #     print(f"已跳过: {skipped_count}")
    # else:
    #     print(f"Warning: Directory {IMAGES_DIR} not found.")
