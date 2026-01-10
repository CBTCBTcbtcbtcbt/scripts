# Photo Renaming Script (照片重命名脚本)

这个脚本可以将指定文件夹中的所有照片重命名为连续的数字编号（0001, 0002, 0003等）。

This script renames all photos in a specified folder to sequential numbers (0001, 0002, 0003, etc.).

## 功能特点 (Features)

- ✅ 支持常见图片格式：JPG, JPEG, PNG, GIF, BMP, TIFF, WEBP, RAW, CR2, NEF
- ✅ 保留原始文件扩展名
- ✅ 重命名前显示预览
- ✅ 需要用户确认才执行
- ✅ 安全的两步重命名过程（避免文件冲突）
- ✅ 自动补零（最少4位数字）

## 使用方法 (Usage)

### 方法1：命令行参数 (Command line argument)
```bash
python rename_photos.py "照片文件夹路径"
```

例如：
```bash
python rename_photos.py "d:/SD-WEBUI-AKI-V5.0/scripts/output"
```

### 方法2：交互式运行 (Interactive mode)
```bash
python rename_photos.py
```
然后输入文件夹路径。

## 示例 (Example)

假设你的 `output` 文件夹包含以下文件：
```
1_42_20251204_1604.png
1_42_20251204_1623.png
2_101_20251204_1604.png
2_101_20251204_1623.png
```

运行脚本后：
```
Found 4 image file(s) to rename

Preview of renaming:
------------------------------------------------------------
1_42_20251204_1604.png -> 0001.png
1_42_20251204_1623.png -> 0002.png
2_101_20251204_1604.png -> 0003.png
2_101_20251204_1623.png -> 0004.png
------------------------------------------------------------

Proceed with renaming 4 file(s)? (yes/no): yes

Renaming files...
Successfully renamed 4 file(s)!
```

结果：
```
0001.png
0002.png
0003.png
0004.png
```

## 注意事项 (Notes)

1. 文件会按原始文件名的字母顺序排序后重命名
2. 脚本会保留文件的原始扩展名
3. 如果有超过9999个文件，会自动增加数字位数
4. 重命名前会显示预览，输入 `yes` 或 `y` 确认后才会执行
5. 使用两步重命名确保安全，避免文件名冲突

## 安全性 (Safety)

- ⚠️ 建议在重命名前备份重要文件
- ⚠️ 重命名操作不可逆，请仔细查看预览
- ✅ 脚本使用临时文件名防止冲突
- ✅ 如发生错误会停止操作并报告

## 支持的图片格式 (Supported Formats)

- `.jpg`, `.jpeg` - JPEG图片
- `.png` - PNG图片
- `.gif` - GIF动图
- `.bmp` - 位图
- `.tiff` - TIFF图片
- `.webp` - WebP图片
- `.raw`, `.cr2`, `.nef` - 相机RAW格式
