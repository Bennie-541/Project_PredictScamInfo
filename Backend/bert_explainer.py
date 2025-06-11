
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

import shap
import torch                
import re
import easyocr
import io
import numpy as np

from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import BertTokenizer, pipeline
from AI_Model_architecture import BertLSTM_CNN_Classifier


reader = easyocr.Reader(['ch_tra', 'en'], gpu=torch.cuda.is_available())
# 設定裝置（GPU 優先）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 預設模型與 tokenizer 為 None，直到首次請求才載入（延遲載入）
# 先載入模型與 tokenizer


# ✅ 延遲載入模型與 tokenizer
def load_model_and_tokenizer():
    global model, tokenizer

    if os.path.exists("model.pth"):
        model_path = "model.pth"
    else:
        model_path = hf_hub_download(repo_id="Bennie12/Bert-Lstm-Cnn-ScamDetecter", filename="model.pth")

    from AI_Model_architecture import BertLSTM_CNN_Classifier
    model = BertLSTM_CNN_Classifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained("ckiplab/bert-base-chinese", use_fast=False)

    return model, tokenizer

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

# ----------- 可疑詞句擷取 -----------
class ModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        elif isinstance(texts, str):
            texts = [texts]
        # 其實到這邊 texts 已經是 list of str
        encodings = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        encodings = {k: v.to(device) for k, v in encodings.items()}
        with torch.no_grad():
            outputs = self.model(encodings["input_ids"], encodings["attention_mask"], encodings["token_type_ids"])
            if len(outputs.shape) == 0:
                outputs = outputs.unsqueeze(0)
            probs = torch.stack([1-outputs, outputs], dim=1)
        return probs.cpu().numpy()
# 背景資料可以使用小樣本即可
background_data = [
    "今天下班後有空嗎？想約你去逛逛新開的夜市，聽說很好吃。",
    "你知道怎麼用圖形計算體積嗎？我找了幾個例題我們一起練。",
    "怡萱，設計部交來的提案不太符合規格，你幫忙聯絡一下調整字型與配色規範。",
    "恭喜您！在我們的年用戶回饋活動中，您抽中一台！請在小時內完成登記手續領取補繳逾期將收取手續費。",
    "您信用卡存在高风险消费记录，需支付安全保障费元，详情请访问",
    "您好我們是官方客服，發現您的帳戶有異常行為，請回覆您的密碼及手機號碼以便協助處理。",
    "您好，信用卡账单异常请尽快支付未结款项元，客服热线",
    "您好，您的手机套餐异常，请支付补缴费用元，微信客服。",
    "您的手機已中毒，所有應用程式有被遠端操控的危險，請馬上掃描並刪除可疑軟體，點此下載。",
    "想起你第一次自己搭飛機那天，媽媽其實超緊張的，還偷哭了一下。",
    "我不是你操控的棋子。",
    "我現在的狀態就是一個會呼吸的失誤",
    "擠公車的時候被夾到背包，後面大嬸還一直催我下車",
    "政府公告您符合老人福利津貼資格，請點擊",
    "最近有什麼好聽的音樂推薦嗎？我想換換口味。",
    "有人說台北信義商圈夜晚有人穿古裝逛街，像穿越時空。",
    "本行系統升級需重新驗證用戶資料，請於小時內完成身份確認程序，否則將暫時凍結帳戶所有交易功能。",
    "水利署通知，月初將進行自來水管線例行維修，可能短暫停水。",
    "緊急通知您有一筆防疫補助款待領取，請點擊連結上傳身份證照片及手機驗證碼以完成申請。網址",
    "親愛的用戶，您的電信合約即將到期，請點擊連結確認續約並享受優惠方案，逾期將自動終止服務。"
]



def clean_text(text):
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[a-zA-Z0-9:/.%\-_=+]{4,}", "", text)
    text = re.sub(r"\+?\d[\d\s\-]{5,}", "", text)
    text = re.sub(r"[^一-龥。，！？、]", "", text)
    sentences = re.split(r"[。！？]", text)
    cleaned = "。".join(sentences[:4])
    return cleaned[:300]

def suspicious_tokens(text, explainer, top_k=5):
    try:
        shap_values = explainer.shap_values([text])
        tokens = tokenizer.tokenize(text)
        scores = shap_values[0][0]

        token_score = list(zip(tokens, scores))
        token_score.sort(key=lambda x: abs(x[1]), reverse=True)
        keywords = [t for t, s in token_score if len(t.strip()) > 1][:top_k]
        return keywords
    except Exception as e:
        print("⚠ SHAP 失敗，啟用 fallback:", e)
        fallback = ["繳費", "終止", "逾期", "限時", "驗證碼"]
        return [kw for kw in fallback if kw in text]

model, tokenizer = load_model_and_tokenizer()
model.eval()

wrapped_model = ModelWrapper(model, tokenizer)

background_data = np.array(background_data)

explainer = shap.KernelExplainer(wrapped_model, background_data)
def analyze_text(text):
    cleaned_text = clean_text(text)

    result = predict_single_sentence(model, tokenizer, cleaned_text)
    label = result["label"]
    prob = result["prob"]
    risk = result["risk"]

    suspicious = suspicious_tokens(cleaned_text, explainer)

    highlighted_text = highlight_keywords(cleaned_text, suspicious)

    # ----------- 顯示推論資訊（後端終端機） -----------
    print(f"\n📩 訊息內容：{text}")
    print(f"✅ 預測結果：{label}")  
    print(f"📊 信心值：{round(prob*100, 2)}")
    print(f"⚠️ 風險等級：{risk}")
    print(f"可疑關鍵字擷取: { [str(s).strip() for s in suspicious if isinstance(s, str) and len(s.strip()) > 1]}")

    return {
        "status": label,
        "confidence": round(prob * 100, 2),
        "suspicious_keywords":  suspicious,
        "highlighted_text": highlighted_text 
    }

def highlight_keywords(text, keywords):
    for word in keywords:
        text = text.replace(word, f"<span class='highlight'>{word}</span>")
    return text

def analyze_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes))
    image_np = np.array(image)
    results = reader.readtext(image_np)
    
    text = ' '.join([res[1] for res in results]).strip()
    
    if not text:
        return{
            "status" : "無法辨識文字",
            "confidence" : 0.0,
            "suspicious_keywords" : ["圖片中無可辨識的中文英文"],
            "highlighted_text": "無法辨識可疑內容"
        }
    return analyze_text(text)