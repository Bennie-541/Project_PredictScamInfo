
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import requests
import os
import tempfile

# === 初始化 FastAPI 應用 ===
app = FastAPI(
    title="詐騙訊息辨識 API (Proxy)",
    description="前端轉送使用者輸入給 Hugging Face 模型後端，並回傳預測結果",
    version="1.0.0"
)

# === 設定 CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開發階段用，正式建議限制來源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === 挂載靜態檔案給前端用 ===
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "frontend")), name="static")

# === Hugging Face Space 的 API 位置 ===
HF_TEXT_API = "https://bennie12-project-predictscaminfo.hf.space/run/predict_text"
HF_IMAGE_API = "https://bennie12-project-predictscaminfo.hf.space/run/predict_image"

# === 定義前後端傳輸資料模型 ===
class TextAnalysisRequest(BaseModel):
    text: str
    user_id: Optional[str] = None
    explain_mode: Optional[str] = "cnn"

class TextAnalysisResponse(BaseModel):
    status: str
    confidence: float
    suspicious_keywords: List[str]
    analysis_timestamp: datetime

# === 轉送文字預測請求到 Hugging Face Space ===
@app.post("/predict", response_model=TextAnalysisResponse)
async def analyze_text_api(request: TextAnalysisRequest):
    try:
        payload = {
            "data": [request.text, request.explain_mode]
        }
        hf_response = requests.post(HF_TEXT_API, json=payload)
        if hf_response.status_code != 200:
            raise Exception(f"Hugging Face API 錯誤：{hf_response.status_code}")

        result = hf_response.json()
        return TextAnalysisResponse(
            status=result["data"][0],
            confidence=float(result["data"][1].replace("%", "")),
            suspicious_keywords=result["data"][2].split(", "),
            analysis_timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"後端錯誤：{str(e)}")

# === 轉送圖片預測請求到 Hugging Face Space ===
@app.post("/predict-image", response_model=TextAnalysisResponse)
async def predict_image_api(file: UploadFile = File(...), explain_mode: str = Form("cnn")):
    try:
        # 將圖片暫存為檔案路徑（HF Space 需要實體路徑）
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        with open(temp_file_path, "rb") as f:
            files = {"file_path": (os.path.basename(temp_file_path), f, "image/png")}
            data = {"data": [temp_file_path, explain_mode]}
            hf_response = requests.post(HF_IMAGE_API, data=data, files=files)

        os.remove(temp_file_path)

        if hf_response.status_code != 200:
            raise Exception(f"Hugging Face API 錯誤：{hf_response.status_code}")

        result = hf_response.json()
        return TextAnalysisResponse(
            status=result["data"][0],
            confidence=float(result["data"][1].replace("%", "")),
            suspicious_keywords=result["data"][2].split(", "),
            analysis_timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"圖片處理錯誤：{str(e)}")

# === 健康檢查 ===
@app.get("/health")
def health_check():
    return {"status": "ok"}

# === 首頁導向 index.html （可選）===
@app.get("/", response_class=FileResponse)
async def read_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html"))
