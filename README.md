# 谣言检测与解释 Demo

## 功能简介
- 基于BERT微调模型进行谣言判断
- 利用DeepSeek LLM实现谣言解释、三元组抽取、事实核查问答
- 提供简单网页界面（Streamlit）

## 依赖安装
```bash
pip install -r requirements.txt
```

## 运行方法
```bash
# 设置DeepSeek API Key（可在系统环境变量中设置DEEPSEEK_API_KEY）
streamlit run app.py
```

## 目录结构
- `bert_rumor_predict.py`：BERT谣言判断
- `llm_utils.py`：DeepSeek LLM API调用
- `triple_extraction.py`：三元组抽取
- `fact_checking.py`：事实核查
- `app.py`：网页Demo入口

## 注意事项
- 需将`BERT_Weibo_Rumor`目录下的模型文件（config.json, vocab.txt, pytorch_model.bin等）准备好
- DeepSeek API Key请替换到`llm_utils.py`或设置环境变量
