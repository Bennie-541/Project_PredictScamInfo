
"""
-------一定要做步驟-------
如果以anaconda開啟vscode請先確認有安狀下列套件
ctrl+shift+x找Live Server並安裝。Live Server是很好用的html前端工具。安裝後，html文件內，右鍵往下找Open with Live server
在anaconda啟動頁面找anaconda_powershell_prompt下在下列套件，複製貼上就好
pip install transformers
pip install torch         
pip install scikit-learn
pip install transformers torch
pip install --upgrade torch --extra-index-url https://download.pytorch.org/whl/cu118
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu
pip install tqdm

---測試本地前後端連接---
->打開terminal再來按+號
->點git bash
->看到這表示正常，注意專案資料夾位置，像我的是D槽Project_PredictScamInfo
(user@LAPTOP-GPASQDRA MINGW64 /d/Project_PredictScamInfo (Update)$)
->輸入 "cd Backend" (進入後端資料夾)
->(/d/Project_PredictScamInfo/Backend)位址有Backend就是OK
->輸入" uvicorn app:app --reload "
->(INFO:     Will watch for changes in these directories: ['D:\\Project_PredictScamInfo\\Backend']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit
INFO:     Waiting for application startup.
INFO:     Application startup complete.)
INFO:     Started reloader process [70644] using StatReload)這樣表示正常
->
----正確顯示----
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Started reloader process...
"""

from fastapi import FastAPI, HTTPException                   # 匯入 FastAPI 主功能模組與 HTTP 錯誤處理
from fastapi.middleware.cors import CORSMiddleware           # 匯入 CORS 模組：用來允許前端跨來源存取 API
from pydantic import BaseModel                               # 用於定義 API 的資料結構模型
from datetime import datetime                                # 處理時間格式（如分析時間戳）
from typing import Optional, List                            # 型別註解：可選、列表
from bert_explainer import analyze_text as bert_analyze_text # 匯入自定義的 BERT 模型分析函式
from firebase_admin import credentials, firestore            # Firebase 管理工具
import firebase_admin

# ---------------- 初始化 FastAPI 應用 ---------------
#團隊合作:前端工程師、測試人員知道你這API做什麼。會影響 /docs 文件清晰度與專案可讀性，在專案開發與交接時非常有用。
app = FastAPI(
    title="詐騙訊息辨識 API",    # 顯示「詐騙訊息辨識 API」
    description="使用 BERT 模型分析輸入文字是否為詐騙內容",# 說明這個 API 的功能與用途
    version="1.0.0"             # 顯示版本，例如 v1.0.0
)
# ---------------- 設定 CORS（允許跨網域請求） ----------------
#FastAPI提供的內建方法，，用來加入中介層(middleware)。在請求抵達API前，或回應送出前，先做某些處理的程式邏輯。
#(Cross-Origin Resource Sharing)一個瀏覽器安全機制。當你的前端(http://localhost:3000)，

app.add_middleware(        # 要請求後端(http://localhost:8000)時，因為「不同來源」，瀏覽器會阻擋請求。      
    CORSMiddleware,        # CORSMiddleware的功能就是「允許或拒絕」哪些來源能存取這個 API。
    allow_origins=["*"],   # 代表所有前端網域(如React前端、Vue前端)都可以發送請求。允許所有來源（不建議正式上線用 *）
    allow_credentials=True,# 允許前端攜帶登入憑證或 Cookies 等認證資訊。如果你使用身份驗證、JWT Token、Session cookie，就要開啟這個。若你是公開 API，沒用到登入，那設成 False 也可以。
    allow_methods=["*"],   # 允許 GET/POST/DELETE 等方法
    allow_headers=["*"],   # 允許自訂標頭(如Content-Type)對應JS第46段。如果沒在後端加上這行，附加在HTTP請求或回應中的「額外資訊」會被擋住。
)
# ---------------- 請求與回應資料模型 ----------------
#繼承自pydantic.BaseModel。FastAPI用來驗證與定義資料結構的標準方式，Pydantic提供自動的：
class TextAnalysisRequest(BaseModel):# 資料驗證(Validation)，自動產生 API 文件(Swagger)，自動將 Python 物件轉成 JSON
    text: str                        # 使用者輸入的訊息
    user_id: Optional[str] = None    # 可選的使用者 ID
class TextAnalysisResponse(BaseModel):
    status: str                      # 預測結果：詐騙/正常
    confidence: float                # 信心分數（通常為 100~0）
    suspicious_keywords: List[str]   # 可疑詞語清單(目前只會回傳風險分級顯示)
    analysis_timestamp: datetime     # 分析完成時間(偏向資料庫用途，目前沒用到)
    text_id: str                     # 系統自動產生 ID(偏向資料庫用途，目前用不到)

# ---------------- 初始化 Firebase ----------------
try:#這是資料庫暫時不會用到
    cred = credentials.Certificate("firebase-credentials.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()  # 建立資料庫 client
except Exception as e:
    print(f"Firebase 初始化錯誤: {e}")

# ---------------- 根目錄測試 API ----------------
@app.get("/")       #
async def root():
    return {
        "message": "詐騙文字辨識 API 已啟動",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs"  # FastAPI 內建自動 API 文件
    }

# ---------------- 主要 /predict 預測端點 ----------------
@app.post("/predict", response_model=TextAnalysisResponse)
async def analyze_text_api(request: TextAnalysisRequest):
    try:
        # 建立唯一分析 ID：以時間+使用者組成
        text_id = f"TXT_{datetime.now().strftime('%Y%m%d%H%M%S')}_{request.user_id or 'anonymous'}"

        # 使用模型分析該文字（實際邏輯在 bert_explainer.py）
        result = bert_analyze_text(request.text)

        # 儲存結果到 Firebase
        record = {
            "text_id": text_id,
            "text": request.text,
            "user_id": request.user_id,
            "analysis_result": {
                "status": result["status"],
                "confidence": result["confidence"],
                "suspicious_keywords": result["suspicious_keywords"],
            },
            "timestamp": datetime.now(),
            "type": "text_analysis"
        }
        db.collection("text_analyses").document(text_id).set(record)

        # 回傳結果給前端
        return TextAnalysisResponse(
            status=result["status"],
            confidence=result["confidence"],
            suspicious_keywords=result["suspicious_keywords"],
            analysis_timestamp=datetime.now(),
            text_id=text_id
        )

    except Exception as e:
        # 若中途錯誤，拋出 HTTP 500 錯誤並附上錯誤訊息
        raise HTTPException(status_code=500, detail=str(e))
