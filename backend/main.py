# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 初始化 FastAPI 應用程式
app = FastAPI(
    title="AI 詐騙訊息辨識系統 API", # 設定 API 文件標題
    description="提供文字訊息詐騙辨識預測功能", # 設定 API 文件描述
    version="1.0.0", # 設定 API 版本
)

# --- CORS 設定 ---
# 這是為了讓前端的網頁能夠向後端發送請求，避免瀏覽器的同源政策阻擋。
# 在開發階段，通常會設定為允許所有來源 ("*")，方便測試。
# 但請注意，在實際部署到生產環境時，強烈建議將 'origins' 替換為你的前端網域！
# 例如：origins = ["https://your-frontend-domain.com", "http://localhost:3000"]
# 如果你的前端是直接打開 index.html 檔案，瀏覽器發出的 origin 會是 "null"，
# 因此在開發階段設定 "*" 是最簡單且有效的做法。
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # 允許的來源列表
    allow_credentials=True,      # 允許跨域請求攜帶憑證 (例如 cookies, HTTP 認證)
    allow_methods=["*"],         # 允許所有 HTTP 方法 (GET, POST, PUT, DELETE, OPTIONS)
    allow_headers=["*"],         # 允許所有 HTTP 頭部
)

# --- 請求體模型定義 ---
# 使用 Pydantic 的 BaseModel 來定義 API 接收的 JSON 資料結構，
# 這有助於資料驗證和自動生成 API 文件。
class MessageInput(BaseModel):
    text: str # 預期接收一個名為 'text' 的字串欄位

# --- API 端點定義 ---
# 定義一個 POST 請求的 API 端點，路徑為 /predict
@app.post("/predict")
async def predict_scam_message(input_data: MessageInput):
    """
    接收文字訊息並模擬預測是否為詐騙。
    
    Args:
        input_data (MessageInput): 包含要預測的文字訊息。
        
    Returns:
        dict: 包含以下資訊的 JSON 回應：
            - is_scam (bool): 是否為詐騙訊息。
            - confidence (float): 模型預測的可信度 (0.0 到 1.0)。
            - suspicious_phrases (list[str]): 偵測到的可疑詞句列表。
    """
    message_text = input_data.text
    
    # --- 模擬預測邏輯 (用於初期測試) ---
    # 在你整合真正的 BERT+LSTM+CNN 模型之前，這段程式碼會回傳假資料，
    # 幫助你測試前後端的連線和介面顯示。
    
    is_scam_result = False
    confidence_score = 0.0
    suspicious_phrases_list = []

    # 簡易判斷規則：如果訊息包含特定關鍵字，則模擬為詐騙訊息
    if "中獎" in message_text or "點擊連結" in message_text or "領取獎金" in message_text:
        is_scam_result = True
        confidence_score = 0.95
        if "中獎" in message_text:
            suspicious_phrases_list.append("中獎")
        if "點擊連結" in message_text:
            suspicious_phrases_list.append("點擊連結")
        if "領取獎金" in message_text:
            suspicious_phrases_list.append("領取獎金")
    elif "驗證碼" in message_text and "請勿告知他人" in message_text:
        is_scam_result = True
        confidence_score = 0.8
        suspicious_phrases_list.append("驗證碼")
        suspicious_phrases_list.append("請勿告知他人")
    else:
        # 如果沒有偵測到特定關鍵字，則模擬為非詐騙訊息
        is_scam_result = False
        confidence_score = 0.1
        suspicious_phrases_list.append("此訊息目前未偵測到明顯詐騙跡象。")

    # 返回 JSON 格式的預測結果
    return {
        "is_scam": is_scam_result,
        "confidence": confidence_score,
        "suspicious_phrases": suspicious_phrases_list
    }

# 啟動指令範例：
# uvicorn main:app --reload --port 8000
# FastAPI 應用程式會在 http://127.0.0.1:8000 上運行。