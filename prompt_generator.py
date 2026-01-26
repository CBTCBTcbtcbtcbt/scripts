"""Simple chat example."""
import sys
import os
import json
import yaml
import re
sys.path.append('..')

from LLM import LLMClient, Conversation, load_config


def generate_prompt():
    """Generate prompt from file and interact with LLM."""
    prompt_path = "meta_prompt.yaml"

    # Load API config and create client
    config = load_config('./config.yaml')
    api = config['api']

    with open(prompt_path, 'r', encoding='utf-8') as f:
            # 使用 yaml.safe_load() 读取文件对象 f 中的 YAML 数据
            prompt = yaml.safe_load(f)

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
    schema = prompt.get('schema')
    response = conv.send(user_message=prompt['user_prompt'], schema=schema)
    
    # Clean and parse response
    try:
        # With structured output, response is already a valid JSON string conforming to schema
        data_list = json.loads(response)
        
        # Transform to desired dictionary format
        formatted_response = {}
        for item in data_list:
            formatted_response[item.get('id', len(formatted_response)+1)] = {
                'positive_prompt': item.get('prompt', ''),
            }
        
        return formatted_response
        
    except Exception as e:
        print(f"Error processing response: {e}")
        print(f"Raw response: {response}")
        return response


if __name__ == '__main__':
    prompt=generate_prompt()
    print(prompt)
    print(type(prompt))

