
# 引入重要套件Import Library
# PyTorch 主模組，和Tensorflow很像 
# 共通點：都是深度學習框架，支援建構神經網路、訓練與推論，都支援GPU加速、載入模型，和處理tensor等。
# 操作比較直覺，接近Python本身的風格，動態圖架構(每一次forward都即時計算)，更容易除錯、快速迭代，在研究領域非常流行。
# re是Python內建的正則表示式(regular expression)模組，在這專案中用來"用關鍵規則篩選文字內容"。
# requests是一個非常好用的 HTTP 請求套件，能讓你從Python發送GET/POST請求，在專案中用來從Google Drive下載模型檔案(model.pth)。
# BertTokenizer:從Hugging Face的transformers套件載入一個專用的「分詞器（Tokenizer）」。
# pip install shap

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from lime.lime_text import LimeTextExplainer
import torch                
import re
import easyocr
import io
import numpy as np

from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import BertTokenizer



reader = easyocr.Reader(['ch_tra', 'en'], gpu=torch.cuda.is_available())
# 設定裝置（GPU 優先）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 預設模型與 tokenizer 為 None，直到首次請求才載入（延遲載入）
model = None
tokenizer = None
# ✅ 延遲載入模型與 tokenizer
def load_model_and_tokenizer():
    global model, tokenizer
    if os.path.exists("model.pth"):
        model_path = "model.pth"
    else:
        model_path = hf_hub_download(repo_id="Bennie12/Bert-Lstm-Cnn-ScamDetecter", filename="model.pth")
    # 匯入模型架構（避免在模組初始化階段就占用大量記憶體）
    from AI_Model_architecture import BertLSTM_CNN_Classifier

    # 載入模型架構與參數，初始化模型架構並載入訓練權重
    model = BertLSTM_CNN_Classifier()
    
    # 這行的功能是：「從 model_path把.pth 權重檔案讀進來，載入進模型裡」。
    # model.load_state_dict(...)把上面載入的權重「套進模型架構裡」
    # torch.load(...)載入.pth 權重檔案，會變成一份 Python 字典
    # map_location=device指定模型載入到 CPU 還是 GPU，避免報錯
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    
    # 這是PyTorch中的「推論模式」設定
    # model.eval()模型處於推論狀態（關掉 Dropout 等隨機操作）
    # 只要是用來「預測」而不是訓練，一定要加 .eval()！
    model.eval()

    # 初始化 tokenizer(不要從 build_bert_inputs 中取)
    # 載入預訓練好的CKIP中文BERT分詞器
    # 能把中文句子轉成 BERT 模型需要的 input 格式（input_ids, attention_mask, token_type_ids）
    tokenizer = BertTokenizer.from_pretrained("ckiplab/bert-base-chinese", use_fast=False)  # ✅ 強制使用非 fast tokenizer


    return model, tokenizer

all_preds = []
all_labels = []

# 預測單一句子的分類結果（詐騙 or 正常）
# model: 訓練好的PyTorch模型
# tokenizer: 分詞器，負責把中文轉成 BERT 能處理的數值格式
# sentence: 使用者輸入的文字句子
# max_len: 限制最大輸入長度（預設 256 個 token）
def predict_single_sentence(model, tokenizer, sentence, max_len=256):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 使用 with torch.no_grad()，代表這段程式「不需要記錄梯度」
    # 這樣可以加速推論並節省記憶體
    with torch.no_grad():
         # ----------- 文字前處理：清洗輸入句子 -----------
        sentence = re.sub(r"\s+", "", sentence)  # 移除所有空白字元（空格、換行等）
        sentence = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9。，！？:/._-]", "", sentence)
        # 保留常見中文字、英數字與標點符號，其他奇怪符號都移除
        # ----------- 使用 BERT Tokenizer 將句子編碼 -----------
        encoded = tokenizer(sentence,
                            return_tensors="pt",       # 回傳 PyTorch tensor 格式（預設是 numpy 或 list）
                            truncation=True,           # 超過最大長度就截斷
                            padding="max_length",      # 不足最大長度則補空白（PAD token）
                            max_length=max_len)        # 設定最大長度為 256
        # 把 tokenizer 回傳的資料送進模型前，to(device)轉到指定的裝置（GPU or CPU）
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        token_type_ids = encoded["token_type_ids"].to(device)
        # ----------- 模型推論：輸出詐騙的機率值 -----------
        output = model(input_ids, attention_mask, token_type_ids)# 回傳的是一個機率值（float）
        prob = output.item()  # 從 tensor 取出純數字，例如 0.86
        label = int(prob > 0.5)  # 如果機率 > 0.5，標為「詐騙」（1），否則為「正常」（0）
        # ----------- 根據機率進行風險分級 -----------
        if prob > 0.9:
            risk = "🔴 高風險（極可能是詐騙）"
        elif prob > 0.5:
            risk = "🟡 中風險（可疑）"
        else:
            risk = "🟢 低風險（正常）"
        # ----------- 根據 label 判斷文字結果 -----------
        pre_label ='詐騙'if label == 1 else '正常'
        
        # 組成一個 Python 字典（對應 API 的 JSON 輸出格式）
        return {
        "label" : pre_label,                  # 預測分類（"詐騙" or "正常"）
        "prob" : prob, # 預測分類（"詐騙" or "正常"）  
        "risk" : risk     # 用風險分級當作"可疑提示"放進 list（名稱為 suspicious_keywords）
    }

# analyze_text(text)對應app.py第117行
# 這個函式是「對外的簡化版本」：輸入一句文字 → 回傳詐騙判定結果
# 用在主程式或 FastAPI 後端中，是整個模型預測流程的入口點

# ----------- LIME可疑詞句擷取 -----------
def suspicious_tokens(model, tokenizer, text, top_k=4):
    print("\n🔍 [suspicious_tokens] 函式被呼叫")
    print(f"📥 傳入文字內容：{text}")
    print(f"📥 資料型別：{type(text)}")

    if not isinstance(text, str) or not text.strip():
        print("❌ 警告：輸入不是合法文字，直接返回空列表")
        return []
    if len(text.strip()) < 4:  # ✅ 如果輸入太短，直接跳過擾動，避免錯誤
        print("⚠️ 警告：文字長度過短（少於4字），跳過LIME分析")
        return []
    
    def predict_proba(texts):

        encoding = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        encoding = {k: v.to(device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = model(encoding["input_ids"], encoding["attention_mask"], encoding["token_type_ids"])
            probs = torch.stack([1 - outputs, outputs], dim=1)

        return probs.cpu().numpy()

    class_names = ['正常', '詐騙']
    explainer = LimeTextExplainer(
    class_names=class_names,
    split_expression='',  # ✅ 每字為擾動單位（適合中文）
    bow=False             # ✅ 保留語序與上下文語義
)
    try:
        explanation = explainer.explain_instance(
            text, predict_proba,
            num_features=top_k,
            labels=[1],
            num_samples=700
        )
        keyword_scores = explanation.as_list(label=1)
        keywords = [word for word, score in keyword_scores]
        print(f"關鍵字長:{keywords}")
        return keywords
    except Exception as e:
        print(f"⚠️ LIME 擾動分析失敗：{e}")
        return []


    
def analyze_text(text):
    model, tokenizer = load_model_and_tokenizer()
    model.eval()

    # 預測標籤與信心分數
    result = predict_single_sentence(model, tokenizer, text)
    label = result["label"]
    prob = result["prob"]
    risk = result["risk"]

    # 擷取可疑詞
    suspicious = suspicious_tokens(model, tokenizer, text)
    
    # ----------- 顯示推論資訊（後端終端機） -----------
    print(f"\n📩 訊息內容：{text}")
    print(f"✅ 預測結果：{label}")  
    print(f"📊 信心值：{round(prob*100, 2)}")
    print(f"⚠️ 風險等級：{risk}")
    print(f"可疑關鍵字擷取: {[str(s) for s in suspicious]}")
    
    return {
        "status": label,
        "confidence": round(prob * 100, 2),
        "suspicious_keywords": [str(s) for s in suspicious]
    }

def analyze_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes))
    image_np = np.array(image)
    results = reader.readtext(image_np)
    
    text = ' '.join([res[1] for res in results]).strip()
    
    if not text:
        return{
            "status" : "無法辨識文字",
            "confidence" : 0.0,
            "suspicious_keywords" : ["圖片中無可辨識的中文英文"]
        }
    return analyze_text(text)