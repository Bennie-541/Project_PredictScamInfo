# bert_explainer.py
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

# 使用預訓練模型（請確保第一次網路通暢）
MODEL_NAME = "uer/roberta-base-finetuned-jd-binary-chinese"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, output_attentions=True)
model.eval()

def analyze_text(text):
    # 將文字轉換為 token
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        attentions = outputs.attentions[-1]  # 最後一層 attention

    # 預測結果
    probs = F.softmax(logits, dim=-1)
    pred_label = torch.argmax(probs).item()
    confidence = round(probs[0][pred_label].item() * 100, 2)

    status = "詐騙" if pred_label == 1 else "正常"

    # 擷取可疑詞（attention 最強的前 3 個詞）
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    attn_weights = attentions[0][0][0][1:]  # [CLS] 對每個詞的注意力
    token_scores = attn_weights.tolist()
    
    # 去除特殊符號 + 排序
    clean_scores = [(t, s) for t, s in zip(tokens[1:], token_scores) if t not in ["[CLS]", "[SEP]", "[PAD]"]]
    top_tokens = sorted(clean_scores, key=lambda x: x[1], reverse=True)[:3]
    suspicious_parts = [t.replace("##", "") for t, _ in top_tokens]

    return {
        "status": status,
        "confidence": confidence,
        "suspicious_parts": suspicious_parts
    }
