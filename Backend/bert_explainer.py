import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch                
import re
import easyocr
import io
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import BertTokenizer
from AI_Model_architecture import BertLSTM_CNN_Classifier
from lime.lime_text import LimeTextExplainer

# OCR 模組
reader = easyocr.Reader(['ch_tra', 'en'], gpu=torch.cuda.is_available())

# 設定裝置（GPU 優先）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入模型與 tokenizer
def load_model_and_tokenizer():
    global model, tokenizer

    if os.path.exists("model.pth"):
        print("✅ 已找到 model.pth 載入模型")
        model_path = "model.pth"
    else:
        print("🚀 未找到 model.pth")
        model_path = hf_hub_download(repo_id="Bennie12/Bert-Lstm-Cnn-ScamDetecter", filename="model.pth")

    model = BertLSTM_CNN_Classifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained("ckiplab/bert-base-chinese", use_fast=False)

    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()
model.eval()

# 預測單一句子的分類結果
def predict_single_sentence(model, tokenizer, sentence, max_len=256):
    sentence = re.sub(r"\s+", "", sentence)
    sentence = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9。，！？:/._-]", "", sentence)

    encoded = tokenizer(sentence, return_tensors="pt", truncation=True, padding="max_length", max_length=max_len)
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        output = model(encoded["input_ids"], encoded["attention_mask"], encoded["token_type_ids"])
        prob = torch.sigmoid(output).item()
        label = int(prob > 0.5)

    risk = "🟢 低風險（正常）"
    if prob > 0.9:
        risk = "🔴 高風險（極可能是詐騙）"
    elif prob > 0.5:
        risk = "🟡 中風險（可疑）"

    pre_label = '詐騙' if label == 1 else '正常'
    
    return {
        "label": pre_label,
        "prob": prob,
        "risk": risk
    }

# 提供 LIME 用的 predict_proba
def predict_proba(texts):
    # tokenizer 批次處理
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )

    # 移動到 GPU 或 CPU
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(encoded["input_ids"], encoded["attention_mask"], encoded["token_type_ids"])
        # outputs shape: (batch_size,)
        probs = torch.sigmoid(outputs).cpu().numpy()

    # 轉成 LIME 格式：(N, 2)
    probs_2d = np.vstack([1-probs, probs]).T
    return probs_2d

    # 移動到 GPU 或 CPU
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(encoded["input_ids"], encoded["attention_mask"], encoded["token_type_ids"])
        # outputs shape: (batch_size,)
        probs = torch.sigmoid(outputs).cpu().numpy()

    # 轉成 LIME 格式：(N, 2)
    probs_2d = np.vstack([1-probs, probs]).T
    return probs_2d

# 初始化 LIME explainer
class_names = ['正常', '詐騙']
lime_explainer = LimeTextExplainer(class_names=class_names)

# 擷取可疑詞彙 (改用 LIME)
def suspicious_tokens(text, explainer=lime_explainer, top_k=5):
    try:
        explanation = explainer.explain_instance(text, predict_proba, num_features=top_k, num_samples=500 )
        keywords = [word for word, weight in explanation.as_list()]
        return keywords
    except Exception as e:
        print("⚠ LIME 失敗，啟用 fallback:", e)
        fallback = ["繳費", "終止", "逾期", "限時", "驗證碼"]
        return [kw for kw in fallback if kw in text]

# 文字清理
def clean_text(text):
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[a-zA-Z0-9:/.%\-_=+]{4,}", "", text)
    text = re.sub(r"\+?\d[\d\s\-]{5,}", "", text)
    text = re.sub(r"[^一-龥。，！？、]", "", text)
    sentences = re.split(r"[。！？]", text)
    cleaned = "。".join(sentences[:4])
    return cleaned[:300]

# 高亮顯示
def highlight_keywords(text, keywords):
    for word in keywords:
        text = text.replace(word, f"<span class='highlight'>{word}</span>")
    return text

# 文字分析主流程
def analyze_text(text):
    cleaned_text = clean_text(text)
    result = predict_single_sentence(model, tokenizer, cleaned_text)
    label = result["label"]
    prob = result["prob"]
    risk = result["risk"]

    suspicious = suspicious_tokens(cleaned_text)
    highlighted_text = highlight_keywords(text, suspicious)

    print(f"\n📩 訊息內容：{text}")
    print(f"✅ 預測結果：{label}")  
    print(f"📊 信心值：{round(prob*100, 2)}")
    print(f"⚠️ 風險等級：{risk}")
    print(f"可疑關鍵字擷取: {suspicious}")

    return {
        "status": label,
        "confidence": round(prob * 100, 2),
        "suspicious_keywords": suspicious,
        "highlighted_text": highlighted_text 
    }

# 圖片 OCR 分析
def analyze_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes))
    image_np = np.array(image)
    results = reader.readtext(image_np)
    
    text = ' '.join([res[1] for res in results]).strip()
    
    if not text:
        return {
            "status" : "無法辨識文字",
            "confidence" : 0.0,
            "suspicious_keywords" : ["圖片中無可辨識的中文英文"],
            "highlighted_text": "無法辨識可疑內容"
        }
    return analyze_text(text)
