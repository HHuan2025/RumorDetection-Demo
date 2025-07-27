from llm_utils import call_deepseek

def extract_triples(text):
    prompt = f"请从以下文本中抽取所有事实三元组（主语，谓语，宾语），以列表形式返回：\n{text}"
    return call_deepseek(prompt)

if __name__ == '__main__':
    text = input('请输入文本：')
    print(extract_triples(text))
