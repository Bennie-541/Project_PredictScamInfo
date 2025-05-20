# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from datetime import datetime
import json
from bert_explainer import analyze_text as bert_analyze_text  # ✅ 整合 BERT 模型

load_dotenv()

app = FastAPI(
    title="詐騙辨識系統 API",
    description="提供詐騙交易與文字內容辨識服務的 API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TransactionData(BaseModel):
    amount: float
    timestamp: datetime
    ip_address: str
    user_id: str
    transaction_type: str
    merchant_id: Optional[str] = None
    location: Optional[str] = None
    device_info: Optional[Dict] = None

class UserInfo(BaseModel):
    user_id: str
    registration_date: datetime
    usual_location: str
    risk_level: str = "低"
    transaction_history: Optional[List[Dict]] = None

class FraudAnalysisRequest(BaseModel):
    transaction_data: TransactionData
    user_info: Optional[UserInfo] = None

class FraudAnalysisResponse(BaseModel):
    status: str
    confidence: float
    suspicious_points: List[str] = []
    risk_level: str
    analysis_timestamp: datetime
    transaction_id: str

class FeedbackRequest(BaseModel):
    transaction_id: str
    feedback: str

class TextAnalysisRequest(BaseModel):
    text: str
    user_id: Optional[str] = None

class TextAnalysisResponse(BaseModel):
    status: str
    confidence: float
    suspicious_keywords: List[str] = []
    risk_level: str
    analysis_timestamp: datetime
    text_id: str

try:
    cred = credentials.Certificate("firebase-credentials.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    print(f"Firebase 初始化錯誤: {e}")

def check_fraud_rules(transaction: TransactionData, user_info: Optional[UserInfo] = None) -> Dict:
    suspicious_points = []
    risk_score = 0

    if transaction.amount > 100000:
        suspicious_points.append("交易金額異常")
        risk_score += 30

    if user_info and transaction.ip_address.split('.')[0] != user_info.usual_location.split('.')[0]:
        suspicious_points.append("IP 位置與常用位置不符")
        risk_score += 20

    current_hour = transaction.timestamp.hour
    if current_hour < 6 or current_hour > 22:
        suspicious_points.append("交易時間異常")
        risk_score += 15

    if user_info and user_info.transaction_history:
        recent_transactions = [t for t in user_info.transaction_history 
                             if isinstance(t.get('timestamp'), datetime) and (datetime.now() - t['timestamp']).total_seconds() < 3600]
        if len(recent_transactions) > 5:
            suspicious_points.append("交易頻率異常")
            risk_score += 25

    if risk_score >= 70:
        risk_level = "高"
    elif risk_score >= 40:
        risk_level = "中"
    else:
        risk_level = "低"

    return {
        "is_fraud": risk_score >= 70,
        "confidence": 100 - risk_score,
        "suspicious_points": suspicious_points,
        "risk_level": risk_level
    }

@app.get("/")
async def root():
    return {
        "message": "詐騙辨識系統 API 正在運行",
        "version": "1.0.0",
        "status": "active",
        "documentation": "/docs"
    }

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="只接受 CSV 檔案")

    try:
        df = pd.read_csv(file.file)
        required_columns = ['amount', 'timestamp', 'ip_address', 'user_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"CSV 檔案缺少必要欄位: {', '.join(missing_columns)}"
            )

        batch = db.batch()
        for index, row in df.iterrows():
            transaction_id = f"CSV_{datetime.now().strftime('%Y%m%d%H%M%S')}_{index}"
            doc_ref = db.collection('transactions').document(transaction_id)
            batch.set(doc_ref, row.to_dict())
        batch.commit()

        return {
            "message": "CSV 檔案上傳成功",
            "rows": len(df),
            "stored_in": "transactions"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-fraud", response_model=FraudAnalysisResponse)
async def analyze_fraud(request: FraudAnalysisRequest):
    try:
        transaction_id = f"TXN_{datetime.now().strftime('%Y%m%d%H%M%S')}_{request.transaction_data.user_id}"
        analysis_result = check_fraud_rules(request.transaction_data, request.user_info)
        analysis_data = {
            "transaction_id": transaction_id,
            "transaction_data": request.transaction_data.dict(),
            "analysis_result": analysis_result,
            "timestamp": datetime.now(),
            "user_feedback": "未提供"
        }
        db.collection('fraud_analysis').document(transaction_id).set(analysis_data)
        return FraudAnalysisResponse(
            status="詐騙" if analysis_result["is_fraud"] else "正常",
            confidence=analysis_result["confidence"],
            suspicious_points=analysis_result["suspicious_points"],
            risk_level=analysis_result["risk_level"],
            analysis_timestamp=datetime.now(),
            transaction_id=transaction_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-text", response_model=TextAnalysisResponse)
async def analyze_text_api(request: TextAnalysisRequest):
    try:
        text_id = f"TXT_{datetime.now().strftime('%Y%m%d%H%M%S')}_{request.user_id or 'anonymous'}"
        result = bert_analyze_text(request.text)

        analysis_result = {
            "is_fraud": result["status"] == "詐騙",
            "confidence": result["confidence"],
            "suspicious_keywords": result["suspicious_parts"],
            "risk_level": "高" if result["confidence"] < 50 else "中" if result["confidence"] < 80 else "低"
        }

        text_analysis_data = {
            "text_id": text_id,
            "text": request.text,
            "user_id": request.user_id,
            "analysis_result": analysis_result,
            "timestamp": datetime.now(),
            "type": "text_analysis"
        }
        db.collection('text_analyses').document(text_id).set(text_analysis_data)

        return TextAnalysisResponse(
            status=result["status"],
            confidence=result["confidence"],
            suspicious_keywords=result["suspicious_parts"],
            risk_level=analysis_result["risk_level"],
            analysis_timestamp=datetime.now(),
            text_id=text_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
