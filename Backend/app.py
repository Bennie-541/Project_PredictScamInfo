
"""
-------一定要做步驟-------
如果以anaconda開啟vscode請先確認有安狀下列套件
ctrl+shift+x找Live Server並安裝。Live Server是很好用的html前端工具。安裝後,html文件內,右鍵往下找Open with Live server
在anaconda啟動頁面找anaconda_powershell_prompt下在下列套件,複製貼上就好

pip install fastapi uvicorn pydantic python-multipart aiofiles transformers huggingface_hub torch
pip install transformers huggingface_hub requests torch torchvision
pip install torch         
pip install scikit-learn
pip install transformers torch
pip install --upgrade torch --extra-index-url https://download.pytorch.org/whl/cu118
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu
pip install tqdm
pip install easyocr
pip install lime


---測試本地前後端連接---
->打開terminal再來按+號
->點git bash
->看到這表示正常,注意專案資料夾位置,像我的是D槽Project_PredictScamInfo
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

from fastapi import FastAPI, HTTPException, UploadFile, File, Form                 # 匯入 FastAPI 主功能模組與 HTTP 錯誤處理
from fastapi.middleware.cors import CORSMiddleware           # 匯入 CORS 模組：用來允許前端跨來源存取 API
from pydantic import BaseModel                               # 用於定義 API 的資料結構模型
from datetime import datetime                                # 處理時間格式(如分析時間戳)
from typing import Optional, List                            # 型別註解：可選、列表
from bert_explainer import analyze_text, analyze_image  # 匯入自定義的 BERT 模型分析函式

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import os
import requests

# ---------------- 初始化 FastAPI 應用 ---------------
#團隊合作:前端工程師、測試人員知道你這API做什麼。會影響 /docs 文件清晰度與專案可讀性,在專案開發與交接時非常有用。
app = FastAPI(
    title="詐騙訊息辨識 API",    # 顯示「詐騙訊息辨識 API」
    description="使用 BERT 模型分析輸入文字是否為詐騙內容",# 說明這個 API 的功能與用途
    version="1.0.0"             # 顯示版本,例如 v1.0.0
)

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "frontend")), name="static")

"""
---------------- 設定 CORS(允許跨網域請求) ----------------
FastAPI提供的內建方法,用來加入中介層(middleware)。在請求抵達API前,或回應送出前,先做某些處理的程式邏輯。
(Cross-Origin Resource Sharing)瀏覽器的安全機制。
"""

app.add_middleware(        
    CORSMiddleware,        # CORSMiddleware的功能就是「允許或拒絕」哪些來源能存取這個 API。
    allow_origins=["*"],   # 代表所有前端網域(如React前端、Vue前端)都可以發送請求。允許所有來源(不建議正式上線用 *)
    allow_credentials=True,# 允許前端攜帶登入憑證或 Cookies 等認證資訊。如果你使用身份驗證、JWT Token、Session cookie,就要開啟這個。若你是公開 API,沒用到登入,那設成 False 也可以。
    allow_methods=["*"],   # 允許 GET, POST, PUT, DELETE, OPTIONS 等方法
    allow_headers=["*"],   # 允許自訂標頭(如Content-Type)對應JS第46段。如果沒在後端加上這行,附加在HTTP請求或回應中的「額外資訊」會被擋住。
)
# ---------------- 請求與回應資料模型 ----------------
#繼承自pydantic.BaseModel。FastAPI用來驗證與定義資料結構的標準方式,Pydantic提供自動的：
class TextAnalysisRequest(BaseModel):# 接收前端
    text: str                        # 使用者輸入的訊息
    user_id: Optional[str] = None    # 可選的使用者 ID
    
class TextAnalysisResponse(BaseModel): # 回傳前端
    status: str                      # 預測結果：詐騙/正常
    confidence: float                # 信心分數(通常為 100~0)
    suspicious_keywords: List[str]   # 可疑詞語清單(目前只會回傳風險分級顯示)
    highlighted_text: str
    analysis_timestamp: datetime     # 分析完成時間(偏向資料庫用途,目前沒用到)
    
    
"""
這是 FastAPI 的路由裝飾器,代表：當使用者對「根目錄 /」發送 HTTP GET 請求時,要執行下面這個函數。
"/" 是網址的根路徑,例如開啟："http://localhost:8000/"就會觸發這段程式。
程式碼中/是API的根路徑。@app.get("/")代表使用者訪問網站最基本的路徑：http://localhost:8000/。這個/是URL路徑的根,不是資料夾。
"""

@app.get("/", response_class=FileResponse)
async def read_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html"))
# 宣告一個非同步函數 root(),FastAPI 支援 async,
# 寫出高效能的非同步處理(像連資料庫、外部 API 等)
# 雖然這裡只是回傳資料,但仍建議保留 async      
# Q:什麼是"非同步函數"(async def)？A:因為有些操作「會花時間」：等後端模型處理,等資料庫查詢,等外部 API 回應。用於處理"等待型操作"如資料庫、模型等。
# 還有保留 async 可以讓你未來擴充時不用重構。
# ---------------- 根目錄測試 API ----------------
@app.get("/")
async def root():
# 這是回傳給前端或使用者的一段 JSON 格式資料(其實就是 Python 的 dict)
    return {
        "message": "詐騙文字辨識 API 已啟動", # 說明這支 API 成功啟動
        "version": "1.0.0", # 告訴使用者目前 API 的版本號
        "status": "active", # 標示服務是否運行中(通常是 active 或 down)
        "docs": "/docs"     # 告訴使用者：自動生成的 API 文件在 /docs
# Q:/docs 是什麼？A:FastAPI 自動幫你建一個文件頁：看每個 API 的用途、參數格式
    }   
"""
---------------- 主要 /predict 預測端點 ----------------
當前端呼叫這個 API,並傳入一段文字時,這段程式會依序做以下事情：
程式碼內有特別註解掉資料庫部份,因為目前資料庫對該專案並不是特別重要,所以註解的方式,避免再Render佈署前後端網頁時出錯。
"""
@app.post("/predict", response_model=TextAnalysisResponse)
async def analyze_text_api(request: TextAnalysisRequest):
        try:
            print("📥 收到請求：", request.text)
            result = analyze_text(request.text)
            print("✅ 模型回傳結果：", result)
            return TextAnalysisResponse(
            status=result["status"],
            confidence=result["confidence"],
            suspicious_keywords=result["suspicious_keywords"],
            highlighted_text=result["highlighted_text"],
            analysis_timestamp=datetime.now(),
        )
        except Exception as e:
            print("❌ 錯誤訊息：", str(e))
            raise HTTPException(status_code=500, detail="內部伺服器錯誤")


@app.post("/predict-image", response_model=TextAnalysisResponse)
async def predict_image_api(file: UploadFile = File(...)):
    try:
        print("📷 收到圖片上傳：", file.filename)
        contents = await file.read()        
        result = analyze_image(contents)
        return TextAnalysisResponse(
            status=result["status"],
            confidence=result["confidence"],
            suspicious_keywords=result["suspicious_keywords"],
            highlighted_text=result["highlighted_text"],
            analysis_timestamp=datetime.now()
        )
    except Exception as e:
        print("❌ 圖片處理錯誤：", str(e))
        raise HTTPException(status_code=500, detail="圖片辨識或預測失敗")
"""
使用模型分析該文字(實際邏輯在 bert_explainer.py)
         呼叫模型進行詐騙分析,這會呼叫模型邏輯(在bert_explainer.py),把輸入文字送去分析,得到像這樣的回傳結果(假設)：
        result = {
            "status": "詐騙",
            "confidence": 0.93,
            "suspicious_keywords": ["繳費", "網址", "限時"]
        }
        
        # 回傳結果給前端。對應script.js第60段註解。
        # status、confidence、suspicious_keywords在script.js、app.py和bert_explainer是對應的變數,未來有需大更動,必須注意一致性。
"""
        
    