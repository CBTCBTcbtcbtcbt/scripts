import torch
import numpy as np
from PIL import Image
from datetime import datetime
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler,UniPCMultistepScheduler,StableDiffusionPipeline
from controlnet_aux import MidasDetector


# --- 工具函数 ---
def generate_timestamp():
    """
    生成时间戳字符串，格式为: YYYYMMDD_HHMMSS
    例如: 20231204_143052
    """
    return datetime.now().strftime("%Y%m%d_%H%M")


def load_prompts_from_file(filepath='prompt.txt'):
    """从prompt.txt文件中读取prompt和negative prompt"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    prompt_text = ""
    negative_prompt_text = ""
    lines = content.split('\n')
    current_section = None
    
    for line in lines:
        line_stripped = line.strip()
        
        # 检测prompt部分开始
        if line_stripped.lower().startswith('prompt:'):
            current_section = 'prompt'
            # 提取冒号后面的内容
            prompt_text = line_stripped[7:].strip()
        # 检测negative prompt部分开始
        elif line_stripped.lower().startswith('negative prompt:'):
            current_section = 'negative'
            # 提取冒号后面的内容
            negative_prompt_text = line_stripped[16:].strip()
        # 如果在prompt部分且有内容,继续追加
        elif current_section == 'prompt' and line_stripped:
            prompt_text += ' ' + line_stripped
        # 如果在negative prompt部分且有内容,继续追加
        elif current_section == 'negative' and line_stripped:
            negative_prompt_text += ' ' + line_stripped
    
    # 清理首尾空格并返回
    return prompt_text.strip(), negative_prompt_text.strip()


# --- 核心功能函数 ---
def load_and_process_depth_image(batch_size=2, batch_num=3):
    """
    加载输入图像并生成深度图（一次性加载全部）
    
    Args:
        batch_size: 每批次生成的图像数量
        batch_num: 批次数量
    
    Returns:
        tuple: (original_images, control_images) 原始图像列表和深度图列表
    """
    # --- 参数定义 ---
    # 启用 GPU 加速
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 2. Adaptor (深度图预处理器) ---
    # controlnet_aux 库提供了一系列预处理器，对应 WebUI 中的 Adaptor
    
    print("正在加载深度图预处理器 (Adaptor)...")
    depth_estimator = MidasDetector.from_pretrained("lllyasviel/Annotators")
    
    # 计算需要加载的总图像数
    total_images = batch_size * batch_num
    
    original_images = []
    control_images = []
    
    print(f"正在加载并处理 {total_images} 张输入图像...")
    for i in range(1, total_images + 1):
        input_image_path = f"input/{i:04d}.png"
        try:
            # 加载输入图片
            original_image = Image.open(input_image_path).convert("RGB")
            
            # 生成深度图作为 ControlNet 的输入
            control_image = depth_estimator(original_image)
            
            original_images.append(original_image)
            control_images.append(control_image)
            
            if i == 1:
                # 保存第一张深度图以供查看
                control_image.save("generated_depth_map.png")
                print(f"已保存第一张生成的深度图: generated_depth_map.png")
        except FileNotFoundError:
            print(f"警告: 找不到文件 {input_image_path}，跳过")
    
    print(f"成功加载 {len(original_images)} 张图像和深度图")
    return original_images, control_images

def load_depth_map_directly(batch_size=2, batch_num=3):
    """
    直接加载已有的深度图，不进行转换（一次性加载全部）
    
    Args:
        batch_size: 每批次生成的图像数量
        batch_num: 批次数量
    
    Returns:
        tuple: (original_images, depth_images) 原始图像列表和深度图列表
    """
    # 计算需要加载的总图像数
    total_images = batch_size * batch_num
    
    depth_images = []
    
    print(f"正在一次性加载 {total_images} 张深度图...")
    for i in range(1, total_images + 1):
        depth_map_path = f"input/{i:04d}.png"
        try:
            depth_image = Image.open(depth_map_path)
            
            if depth_image.mode == 'RGBA':
                depth_image = depth_image.convert('RGB')
            elif depth_image.mode != 'RGB':
                depth_image = depth_image.convert('RGB')
            
            depth_images.append(depth_image)
            
            if i == 1:
                print(f"第一张深度图加载完成，尺寸: {depth_image.size}")
        except FileNotFoundError:
            print(f"警告: 找不到文件 {depth_map_path}，跳过")
    
    print(f"成功加载 {len(depth_images)} 张深度图")
    return depth_images, depth_images


def load_controlnet_model():
    """
    加载 ControlNet 模型
    
    Returns:
        ControlNetModel: 加载的 ControlNet 模型
    """
    # --- 参数定义 ---
    # 启用 GPU 加速
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 3. 加载 ControlNet 模型 ---
    # 使用相对路径，方便移植
    controlnet_model_path = "controlnet.safetensors"
    print(f"正在加载 ControlNet 模型: {controlnet_model_path}")
    controlnet = ControlNetModel.from_single_file(
        controlnet_model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32 
    )
    
    return controlnet


def setup_pipeline(controlnet):
    """
    设置 Stable Diffusion Pipeline 和调度器
    
    Args:
        controlnet: ControlNet 模型
        
    Returns:
        StableDiffusionControlNetPipeline: 配置好的 pipeline
    """
    # --- 参数定义 ---
    # 替换成您自己的主模型路径或名称 (例如：'runwayml/stable-diffusion-v1-5')
    base_model_path = "SG161222/RealVisXL_V5.0"
    local_base_model_path = "D:/SD-WEBUI-AKI-V5.0/sd-webui-aki-v4.10/models/Stable-diffusion/realisticVisionV60/realisticVisionV60B1_v51HyperVAE.safetensors"
    
    # 启用 GPU 加速
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 4. 设置 Pipeline ---
    '''
    print(f"正在加载主 Stable Diffusion Pipeline: {base_model_path}")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path, 
        #controlnet=controlnet, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32 
    )
    '''
    # 使用本地模型文件
    print(f"正在加载主 Stable Diffusion Pipeline: {local_base_model_path}")
    pipe = StableDiffusionControlNetPipeline.from_single_file(
        local_base_model_path,
        # 同样建议设置 dtype
        controlnet=controlnet,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32 
    )
    
    
    
    # 1. 获取配置
    scheduler_config = pipe.scheduler.config
    
    # 2. 实例化 DPM++ 2M Karras
    # DPMSolverMultistepScheduler 对应 WebUI 的 DPM++ 2M
    dpm_karras_scheduler = DPMSolverMultistepScheduler.from_config(
        scheduler_config, 
        # 开启 Karras 采样 (对应 WebUI 名称中的 Karras)
        use_karras_sigmas=True 
    )
    
    # 3. 替换 Pipeline 中的调度器
    pipe.scheduler = dpm_karras_scheduler
    print("采样方法已设置为: DPM++ 2M Karras")
    
    # 优化内存和加速
    pipe.to(device)
    #pipe.enable_xformers_memory_efficient_attention() # 如果安装了 xformers
    
    return pipe

def setup_pipeline_without_controlnet():
    """
    设置 Stable Diffusion Pipeline 和调度器
    
    Args:
        controlnet: ControlNet 模型
        
    Returns:
        StableDiffusionControlNetPipeline: 配置好的 pipeline
    """
    # --- 参数定义 ---
    # 替换成您自己的主模型路径或名称 (例如：'runwayml/stable-diffusion-v1-5')
    base_model_path = "SG161222/RealVisXL_V5.0"
    local_base_model_path = "D:/SD-WEBUI-AKI-V5.0/sd-webui-aki-v4.10/models/Stable-diffusion/realisticVisionV60/realisticVisionV60B1_v51HyperVAE.safetensors"
    
    # 启用 GPU 加速
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 4. 设置 Pipeline ---
    '''
    print(f"正在加载主 Stable Diffusion Pipeline: {base_model_path}")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path, 
        #controlnet=controlnet, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32 
    )
    '''
    # 使用本地模型文件
    print(f"正在加载主 Stable Diffusion Pipeline: {local_base_model_path}")
    pipe = StableDiffusionPipeline.from_single_file(
        local_base_model_path,
        # 同样建议设置 dtype
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32 
    )
    
    
    
    # 1. 获取配置
    scheduler_config = pipe.scheduler.config
    
    # 2. 实例化 DPM++ 2M Karras
    # DPMSolverMultistepScheduler 对应 WebUI 的 DPM++ 2M
    dpm_karras_scheduler = DPMSolverMultistepScheduler.from_config(
        scheduler_config, 
        # 开启 Karras 采样 (对应 WebUI 名称中的 Karras)
        use_karras_sigmas=True 
    )
    
    # 3. 替换 Pipeline 中的调度器
    pipe.scheduler = dpm_karras_scheduler
    print("采样方法已设置为: DPM++ 2M Karras")
    
    # 优化内存和加速
    pipe.to(device)
    #pipe.enable_xformers_memory_efficient_attention() # 如果安装了 xformers
    
    return pipe

def generate_images_batch(pipe, prompt, negative_prompt, batch_size, batch_num,control_images_list=None):
    """
    批量生成图像（从预加载的图像列表中切片）
    
    Args:
        pipe: Stable Diffusion Pipeline
        control_images_list: 预加载的深度图列表（总共 batch_size * batch_num 张）
        prompt: 正向提示词
        negative_prompt: 负向提示词
        batch_size: 每批次生成的图像数量
        batch_num: 批次数量
    """
    # --- 参数定义 ---
    # 启用 GPU 加速
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 批量生成参数
    seed_population = [42, 101, 202, 303, 3112, 314, 634, 4, 245] # 设置种子库
    
    # 核心参数 (对应 WebUI)
    guidance_scale = 10 # CFG Scale
    num_inference_steps = 20 # Steps
    output_width = 512 # 宽度 (继承自输入图，或自定义)
    output_height = 512 # 高度 (继承自输入图，或自定义)
    
    # --- 6. 批量生成图像 ---
    for i in range(batch_num):
        print(f"这是第 {i+1} 轮生成...")
        print(f"开始批量生成 {batch_size} 张图像...")
        
        # 从种子库中切片获取当前批次的种子
        seed = seed_population[i*batch_size:(i+1)*batch_size]
        generator = [torch.Generator(device=device).manual_seed(s) for s in seed] # 批量设置种子
        
        # 从预加载的图像列表中切片获取当前批次的图像
        control_image_list = control_images_list[i*batch_size:(i+1)*batch_size] if control_images_list else None
        
        output = pipe(
            prompt=[prompt] * batch_size,          # 提示词列表
            negative_prompt=[negative_prompt] * batch_size, # 反向提示词列表
            image=control_image_list if control_image_list  else None,           # ControlNet 的输入图像列表（从预加载列表切片）
            generator=generator,                   # 随机种子列表
            guidance_scale=guidance_scale,         # CFG Scale
            num_inference_steps=num_inference_steps, # Steps
            width=output_width,                    # 宽度 (继承自输入图，或自定义)
            height=output_height,                  # 高度 (继承自输入图，或自定义)
        ).images
    
        # --- 7. 保存结果（带时间戳） ---
        print("图像生成完成，正在保存...")
        # 为本次批量生成创建一个共同的时间戳
        batch_timestamp = generate_timestamp()
    
        for j, img in enumerate(output):
            # 文件名格式: output_batch_{批次号}_seed_{种子}_{时间戳}.png
            filename = f"output/{i+1}_{seed[j]}_{batch_timestamp}.png"
            img.save(filename)
            print(f"已保存: {filename}")
    
    print("所有图像已成功保存。")

def generate_images_from_different_prompt(pipe, positive_prompt_list, negative_prompt_list, batch_size, batch_num,control_images_list=None):
    """
    批量生成图像（从预加载的图像列表中切片）
    
    Args:
        pipe: Stable Diffusion Pipeline
        control_images_list: 预加载的深度图列表（总共 batch_size * batch_num 张）
        prompt: 正向提示词
        negative_prompt: 负向提示词
        batch_size: 每批次生成的图像数量
        batch_num: 批次数量
    """
    # --- 参数定义 ---
    # 启用 GPU 加速
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 批量生成参数
    seed_population = [284, 952, 17, 439, 761, 82, 593, 312, 104, 678, 
    455, 23, 891, 540, 126, 734, 92, 388, 615, 201, 
    847, 52, 419, 963, 11, 705, 532, 28, 649, 157, 
    822, 471, 39, 908, 246, 783, 55, 612, 134, 492, 
    975, 88, 321, 567, 19, 814, 402, 75, 631, 224, 
    859, 142, 518, 936, 6, 742, 365, 47, 689, 213, 
    801, 554, 31, 925, 448, 719, 64, 582, 176, 395, 
    876, 123, 485, 992, 4, 768, 334, 58, 663, 291, 
    834, 169, 507, 951, 14, 726, 427, 81, 604, 258, 
    888, 115, 463, 984, 9, 753, 309, 72, 642, 187] # 设置种子库
    
    # 核心参数 (对应 WebUI)
    guidance_scale = 10 # CFG Scale
    num_inference_steps = 20 # Steps
    output_width = 512 # 宽度 (继承自输入图，或自定义)
    output_height = 512 # 高度 (继承自输入图，或自定义)
    

    if isinstance(positive_prompt_list, dict):
        positive_prompt_list = list(positive_prompt_list.values())
        negative_prompt_list = list(negative_prompt_list.values()) 
        print("提示词列表和反向提示词列表从字典转换为列表")

    print(f"提示词列表: {positive_prompt_list}")
    print(f"反向提示词列表: {negative_prompt_list}")
    # --- 6. 批量生成图像 ---
    for i in range(batch_num):
        print(f"这是第 {i+1} 轮生成...")
        print(f"开始批量生成 {batch_size} 张图像...")
        
        # 从种子库中切片获取当前批次的种子
        seed = seed_population[i*batch_size:(i+1)*batch_size]
        generator = [torch.Generator(device=device).manual_seed(s) for s in seed] # 批量设置种子
        positive_prompt = positive_prompt_list[i]
        negative_prompt = negative_prompt_list[i]
        
        output = pipe(
            prompt = [positive_prompt]*batch_size,          # 提示词列表
            negative_prompt=[negative_prompt]*batch_size, # 反向提示词列表
            image= control_images_list if control_images_list  else None,           # ControlNet 的输入图像列表（从预加载列表切片）
            generator=generator,                   # 随机种子列表
            guidance_scale=guidance_scale,         # CFG Scale
            num_inference_steps=num_inference_steps, # Steps
            width=output_width,                    # 宽度 (继承自输入图，或自定义)
            height=output_height,                  # 高度 (继承自输入图，或自定义)
        ).images
        
        # --- 7. 保存结果（带时间戳） ---
        print("图像生成完成，正在保存...")
        # 为本次批量生成创建一个共同的时间戳
        batch_timestamp = generate_timestamp()
    
        for j, img in enumerate(output):
            # 文件名格式: output_batch_{批次号}_seed_{种子}_{时间戳}.png
            filename = f"data/images/{i+1}_{j+1}.png"
            img.save(filename)
            print(f"已保存: {filename}")
    
    print("所有图像已成功保存。")
# --- 主函数 ---
def main():
    """
    主函数：执行完整的图像生成流程
    """
    batch_size = 2 # 每批次生成多少张图像
    batch_num = 3 # 生成多少批图像
    with_controlnet = False # 是否使用 ControlNet 进行提示词控制
    already_depth_map = False # 是否直接加载已有的深度图（否则使用预处理器生成）
    control_images = None # 预加载的深度图列表（总共 batch_size * batch_num 张）
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
    # 4. 从文件读取生成参数
    print("正在从prompt.txt加载prompts...")
    prompt, negative_prompt = load_prompts_from_file('prompt.txt')
    print(f"已加载 prompt (前80字符): {prompt[:80]}...")
    print(f"已加载 negative prompt (前80字符): {negative_prompt[:80]}...")
    
    # 5. 批量生成图像（从预加载的图像列表中切片使用）
    generate_images_batch(pipe, prompt, negative_prompt, batch_size, batch_num, control_images)


# --- 程序入口 ---
if __name__ == "__main__":
    main()
