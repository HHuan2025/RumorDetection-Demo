import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

MODEL_DIR = './BERT_Weibo_Rumor'

tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2)
model.eval()

def predict_rumor(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    label = '谣言' if pred == 1 else '非谣言'
    return label, confidence

if __name__ == '__main__':
    text = input('请输入文本：')
    label, conf = predict_rumor(text)
    print(f'判断结果：{label}，置信度：{conf:.2f}')
