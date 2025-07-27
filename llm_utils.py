import requests
import os

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'sk-8a32f139960341cab70374c543b16c04')

API_URL = 'https://api.deepseek.com/v1/chat/completions'


def call_deepseek(prompt, api_key=DEEPSEEK_API_KEY, model='deepseek-chat'):
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    data = {
        'model': model,
        'messages': [
            {'role': 'user', 'content': prompt}
        ]
    }
    response = requests.post(API_URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']
