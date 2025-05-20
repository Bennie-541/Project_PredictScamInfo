# main.py (簡化版本，僅保留文字詐騙分析功能)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
from bert_explainer import analyze_text as bert_analyze_text
import firebase_admin
from firebase_admin import credentials, firestore

# 初始化 FastAPI 應用
app = FastAPI(
    title="詐騙訊息辨識 API",
    description="使用 BERT 模型分析輸入文字是否為詐騙內容",
    version="1.0.0"
)

# 允許所有來源跨域存取（適用於本機或前後端分離情境）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 定義資料請求與回應模型 ---

class TextAnalysisRequest(BaseModel):
    text: str                        # 使用者輸入的訊息內容
    user_id: Optional[str] = None   # 選填：使用者 ID，可用於追蹤來源

class TextAnalysisResponse(BaseModel):
    status: str                     # 模型預測為「詐騙」或「正常」
    confidence: float               # 模型的可信度 (0-100)
    suspicious_keywords: List[str]  # 可疑詞語（從 attention 擷取）
    risk_level: str                 # 系統標註風險等級（低 / 中 / 高）
    analysis_timestamp: datetime    # 預測完成時間戳
    text_id: str                    # 自動產生的分析 ID

# --- 初始化 Firebase 連線 ---
try:
    cred = credentials.Certificate("firebase-credentials.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    print(f"Firebase 初始化錯誤: {e}")

# --- 根目錄（確認服務運行狀態） ---
@app.get("/")
async def root():
    return {
        "message": "詐騙文字辨識 API 已啟動",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs"
    }

# --- BERT 文字詐騙分析 API ---
@app.post("/predict", response_model=TextAnalysisResponse)
async def analyze_text_api(request: TextAnalysisRequest):
    try:
        # 為此次請求產生唯一 ID
        text_id = f"TXT_{datetime.now().strftime('%Y%m%d%H%M%S')}_{request.user_id or 'anonymous'}"

        # 使用 BERT 模型進行預測（來自 bert_explainer.py）
        result = bert_analyze_text(request.text)

        # 根據可信度分類風險等級
        if result["confidence"] < 50:
            risk_level = "高"
        elif result["confidence"] < 80:
            risk_level = "中"
        else:
            risk_level = "低"

        # 儲存分析結果至 Firebase（可略過這段，如不需紀錄）
        record = {
            "text_id": text_id,
            "text": request.text,
            "user_id": request.user_id,
            "analysis_result": {
                "status": result["status"],
                "confidence": result["confidence"],
                "suspicious_keywords": result["suspicious_parts"],
                "risk_level": risk_level
            },
            "timestamp": datetime.now(),
            "type": "text_analysis"
        }
        db.collection("text_analyses").document(text_id).set(record)

        # 回傳結果給前端
        return TextAnalysisResponse(
            status=result["status"],
            confidence=result["confidence"],
            suspicious_keywords=result["suspicious_parts"],
            risk_level=risk_level,
            analysis_timestamp=datetime.now(),
            text_id=text_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
