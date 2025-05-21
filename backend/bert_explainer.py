#pip install transformers
#pip install torch         //transformers 套件需要
#pip install scikit-learn
#pip install transformers torch
#
# bert_explainer.py
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

#供參考用，正常，測試前後端輸入、輸出是否正常，之後會刪除
def analyze_text(text):
    length = len(text)

    return {
        "status": "目前為測試階段",
        "confidence": length,  # 直接用字數當作假可信度
        "suspicious_keywords": [f"目前為測試階段，將回傳輸入內容: {text}"]
    }
