import json
import base64

# 读取 JSON 文件
with open('data/action_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 遍历每个条目，将图片路径转换为 base64
for item in data:
    new_images = []
    for img_path in item['images']:
        with open(img_path, 'rb') as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            new_images.append(img_base64)
    item['images'] = new_images

# 保存修改后的 JSON 文件
with open('data/action_data_base64.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print('转换完成！')
