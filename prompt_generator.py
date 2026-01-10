"""Simple chat example."""
import sys
import os
import json
import re
sys.path.append('..')

from LLM import LLMClient, Conversation, load_config


def generate_prompt():
    """Generate prompt from file and interact with LLM."""
    prompt_path = "meta_prompt.json"

    # Load API config and create client
    config = load_config('./config.yaml')
    api = config['api']

    with open(prompt_path, 'r', encoding='utf-8') as f:
            # 使用 json.load() 读取文件对象 f 中的 JSON 数据
            prompt = json.load(f)

    client = LLMClient(
        api_key=api['api_key'],
        base_url=api['base_url'],
        model=api['model'],
        temperature=api.get('temperature', 0.7),
        max_tokens=api.get('max_tokens', 10000)
    )

    # Create conversation
    conv = Conversation(client, system_prompt=prompt['system_prompt'])

    # Chat
    response = conv.send(user_message=prompt['user_prompt'])
    
    # Clean and parse response
    try:
        # Remove markdown code blocks if present
        content = re.sub(r'^```json\s*', '', response.strip(), flags=re.MULTILINE)
        content = re.sub(r'\s*```$', '', content, flags=re.MULTILINE)
        
        data_list = json.loads(content)
        
        # Transform to desired dictionary format
        formatted_response = {}
        for item in data_list:
            formatted_response[item['id']] = {
                'positive_prompt': item['prompt'],
                'negative_prompt': item['negative_prompt']
            }
        
        return formatted_response
        
    except Exception as e:
        print(f"Error processing response: {e}")
        print(f"Raw response: {response}")
        return response


if __name__ == '__main__':
    generate_prompt()
