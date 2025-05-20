# 🛡️ AI 訊息辨識與預警系統 — ScamGuard AI

## 🔍 專案簡介
面對日益猖獗的網路詐騙，每年詐騙案件數量與金額屢創新高。詐騙訊息形式多樣，從傳統的電話、簡訊，延伸到 LINE、Facebook、Instagram 等社交平台，讓人們難以辨識真偽，所有人都有可能成為受害者。

ScamGuard AI 旨在建立一套智慧型 AI 系統，能即時辨識潛在的詐騙訊息，並提示可疑字詞，協助使用者防範詐騙風險。

## 🎯 專案目標
- 即時詐騙辨識：使用者輸入訊息文字，AI 模型分析語意並預測是否為詐騙。
- 可疑詞句提示：標示訊息中被辨識為可疑的關鍵字或片段（如：「中獎」、「連結」、「帳號異常」）。
- 模型可信度顯示：呈現模型預測可信度百分比，讓使用者掌握判斷依據。
- 歷史紀錄儲存（規劃中）：建立個人警示資料庫，追蹤偵測紀錄。
- 多平台擴充性（規劃中）：未來可擴展為 LINE 機器人、Chrome 擴充外掛、手機 App 等。

## 🧠 技術領域與模型架構

### 模型架構：BERT + LSTM + CNN

1. 資料前處理  
   - 移除雜訊符號與特殊字元  
   - 使用 Tokenizer 進行分詞與向量化  
   - BERT 專用 Subword Tokenization  

2. 語意特徵擷取（BERT）  
   - 使用 BERT 預訓練模型提取 contextualized word embeddings  
   - 捕捉詞語語境與語氣變化  

3. 語序理解（LSTM）  
   - 使用 LSTM 理解長期依賴與語序邏輯  
   - 增強語意遞進與上下文理解  

4. 局部特徵擷取（CNN）  
   - 擷取詐騙常見片語，如「立即點擊」、「恭喜中獎」等  
   - 使用不同大小卷積核強化關鍵特徵辨識  

5. 分類層（Dense + Softmax）  
   - 將特徵映射至二元分類空間  
   - 輸出詐騙（0）或非詐騙（1）機率  

## 📤 輸出顯示
- ⚠️ 風險提示訊息  
- 模型預測可信度  
- 可疑詞句標紅或列表顯示  

---

## 🗂️ 專案架構

```bash
Project_PredictScamInfo/
├── frontend/               # 使用者前端介面
│   ├── index.html          # 主頁面（輸入框與結果顯示）
│   ├── style.css           # 前端樣式
│   └── script.js           # 發送 POST 請求至後端 API
├── backend/                # 後端 FastAPI 應用
│   ├── main.py             # 主應用程式與 /predict API
│   ├── model/              # 模型與預測邏輯
│   │   ├── model.pkl       # 訓練後模型（尚未加入）
│   │   ├── predictor.py    # 載入模型與執行預測
│   │   └── bert_tokenizer_files/ # BERT tokenizer 檔案
│   ├── requirements.txt    # 所需套件
│   └── .env                # 環境變數（模型路徑等）
├── data/                   # 訓練資料與前處理資料
│   └── messages.csv        # 範例資料
├── README.md               # 專案說明文件
└── .gitignore              # 忽略上傳的檔案規則
```

---
## 🚀 如何運行專案

### 1️⃣ 前置準備

下載專案並切換目錄：

```bash
git clone <你的專案 Git 網址>
cd Project_PredictScamInfo
```

建立虛擬環境（建議）：
```bash
python3 -m venv venv_backend
source venv_backend/bin/activate  # macOS/Linux
# Windows 使用：
# .\venv_backend\Scripts\activate
```

安裝後端依賴套件：
```bash
cd backend
pip install -r requirements.txt
```

### 2️⃣ 啟動後端 API
```bash
cd backend
uvicorn main:app --reload --port 8000
```

你會看到類似訊息：
```bash
Uvicorn running on http://127.0.0.1:8000
```

### 3️⃣ 啟動前端網頁
直接用瀏覽器打開以下檔案：
```bash
Project_PredictScamInfo/frontend/index.html
```
---

## 🧪 使用說明

1. 輸入欲檢測的訊息（例如 LINE 或簡訊內容）  
2. 點擊「檢測！」按鈕  
3. 檢視下方結果：  
   - 是否為詐騙  
   - 可疑關鍵字  
   - 模型可信度百分比

---

## 🔭 開發進度與未來展望

### ✅ 目前進度
- 前後端基本串接完成  
- 模擬辨識功能可正常運作  

### 📌 未來規劃
- 整合實際訓練後模型  
- 實現歷史紀錄保存  
- 延伸應用至 LINE 機器人、Chrome 外掛、手機 App
