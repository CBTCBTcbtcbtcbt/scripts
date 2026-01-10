# SD-WEBUI-AKI-V5.0 Scripts 项目使用文档

本项目是一个基于 Python 的 Stable Diffusion 图像生成工具，支持批量生成、深度图控制 (ControlNet) 以及提示词管理功能。

本文档将指导您完成环境配置、模型设置以及如何运行脚本。

## 1. 环境准备 (Prerequisites)

在开始之前，请确保您的计算机上已安装以下软件：

*   **Python 3.10+**: 建议使用 Python 3.10 或更高版本。
*   **CUDA Toolkit (可选)**: 如果您有 NVIDIA 显卡，建议安装 CUDA 以获得更快的生成速度。

### 安装依赖

项目根目录下提供了 `requirements.txt` 文件。请打开终端（CMD 或 PowerShell），进入项目根目录，并运行以下命令安装所需依赖：

```bash
pip install -r requirements.txt
```

---

## 2. 模型文件设置 (Crucial)

本项目依赖 Stable Diffusion 主模型和 ControlNet 模型。**请务必按照以下步骤配置模型文件，否则程序将无法运行。**

### 2.1 ControlNet 模型

1.  **准备模型文件**: 您需要准备一个 ControlNet 模型文件（通常是以 `.safetensors` 结尾的文件，例如深度图控制模型）。
2.  **重命名**: 将您的模型文件重命名为 **`controlnet.safetensors`**。
3.  **放置位置**: 将该文件直接放在项目的**根目录**下。

   > **注意**: 该文件已被配置为 Git 忽略，因此不会被提交到版本控制系统中，请放心存放。

### 2.2 Stable Diffusion 主模型

脚本中默认使用本地路径加载主模型。您需要根据您的实际情况修改 `SDtools.py` 中的模型路径。

1.  打开 `SDtools.py` 文件。
2.  找到 `setup_pipeline` 和 `setup_pipeline_without_controlnet` 函数。
3.  修改 `local_base_model_path` 变量为您本地模型的绝对路径。

```python
# 示例：修改为您自己的模型路径
local_base_model_path = "D:/Path/To/Your/Model/model.safetensors"
```

---

## 3. 如何运行 (Usage)

### 3.1 批量生成图像

主要功能的入口文件是 `SDtools.py`。

1.  **配置生成参数**:
    打开 `SDtools.py` 的 `main` 函数部分，您可以调整以下参数：
    *   `batch_size`: 每批次生成的图像数量。
    *   `batch_num`: 生成多少批次。
    *   `with_controlnet`: 是否启用 ControlNet (设置为 `True` 或 `False`)。
    *   `already_depth_map`: 是否直接读取 input 文件夹下的图片作为深度图。

2.  **配置提示词**:
    修改根目录下的 `prompt.txt` 文件，填入您的正向提示词 (prompt) 和反向提示词 (negative prompt)。

    格式示例：
    ```text
    prompt: a beautiful landscape, mountains, river, 8k resolution
    negative prompt: blur, low quality, distortion
    ```

3.  **运行脚本**:
    在终端中运行：

    ```bash
    python SDtools.py
    ```

    生成的图像将保存在 `output/` 文件夹中。

### 3.2 图像分析与处理 (可选)

`main.py` 脚本用于后续的图像分析和指令生成任务。

1.  确保 `data/images/` 目录下有待处理的图像。
2.  运行脚本：

    ```bash
    python main.py
    ```

    结果将保存到 `data/action_data.json` 中。

---

## 4. 文件与目录说明

*   **`SDtools.py`**: 核心图像生成脚本，包含 ControlNet 加载、Pipeline 设置和批量生成逻辑。
*   **`main.py`**: 图像分析与指令生成流程脚本。
*   **`controlnet.safetensors`**: (需手动放置) ControlNet 模型文件。
*   **`prompt.txt`**: 定义生成图像的提示词。
*   **`input/`**: 存放用于生成深度图的原始图片（若启用 ControlNet）。
*   **`output/`**: 存放 `SDtools.py` 生成的图像结果。
*   **`data/`**: 存放 `main.py` 处理的中间数据和结果。
*   **`.gitignore`**: Git 忽略配置文件，已配置为忽略 `.safetensors` 等大文件。

## 5. 常见问题 (FAQ)

**Q: 运行 `git status` 看不到 `controlnet.safetensors`？**
A: 这是正常的。为了防止仓库体积过大，我们已在 `.gitignore` 中忽略了该文件。只要文件在根目录下，脚本就能正常读取。

**Q: 报错 `FileNotFoundError: ... controlnet.safetensors`？**
A: 请检查您是否已将模型文件重命名为 `controlnet.safetensors` 并放在了项目根目录下。

**Q: 报错找不到主模型路径？**
A: 请参考“2.2 Stable Diffusion 主模型”一节，在 `SDtools.py` 中修改为您电脑上正确的主模型路径。
