from llm_utils import call_deepseek

def fact_check(text):
    prompt = f"请对以下内容进行事实核查，并简要说明理由：\n{text}"
    return call_deepseek(prompt)

if __name__ == '__main__':
    text = input('请输入需要核查的文本：')
    print(fact_check(text))
