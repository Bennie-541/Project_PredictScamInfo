#pip install transformers
#pip install torch         //transformers 套件需要
#pip install scikit-learn
#pip install transformers torch
#
# bert_explainer.py


#引入重要套件Import Library
import torch                            #   PyTorch 主模組               
import torch
import re
import os
import requests

from AI_Model_architecture import BertLSTM_CNN_Classifier 
from transformers import BertTokenizer


# 這是一個函式，讓你自動從 Google Drive 抓下模型檔案 .pth。
# file_id: 你從 Google Drive 拿到的模型 ID
# destination: 你要儲存的本地檔案路徑（例如 "model.pth"）
def download_model_from_Gdrive(file_id, destination):#Model.pth連結(https://drive.google.com/file/d/19t6NlRFMc1i8bGtngRwIRtRcCmibdP9q/view?usp=drive_link)
    # url變數用來產生真正的「直接下載連結」file_id預設是Google雲端裡的model.pth檔案id，讓程式能下載model.pth
    url = f"https://drive.google.com/uc?export=download&id={file_id}"  
    if not os.path.exists(destination):   # 如果本地還沒有這個檔案 → 才下載（避免重複）
        print("📥 Downloading model from Google Drive...")
        r = requests.get(url)             # 用requests發送GET請求到Google Drive
        with open(destination, 'wb')as f: # 把下載的檔案內容寫入到 model.pth 本地檔案
            f.write(r.content)
        print("✅ Model downloaded.")     
    else:
        print("📦 Model already exists.")
        
# 在你 load 模型之前執行
# 這行是為了取得模型儲存路徑，確保即使換不同電腦或上雲部署也能正確找到檔案。
# 簡單講：這行就是「找到這支程式所在的資料夾，並指定那裡的 model.pth 檔案」
# os.path.join(資料夾,"model.pth")把資料夾+檔名「安全地合併」，變成完整路徑(跨平台兼容)
# __file__是Python內建變數，代表「目前這支程式碼的檔案路徑」
# os.path.dirname(__file__)取得這支.py 檔的「資料夾路徑」
model_path = os.path.join(os.path.dirname(__file__), "model.pth")#功能說明：

download_model_from_Gdrive("19t6NlRFMc1i8bGtngRwIRtRcCmibdP9q", model_path)

# 裝置選擇
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型架構並載入訓練權重
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

# 初始化 tokenizer（不要從 build_bert_inputs 中取）
tokenizer = BertTokenizer.from_pretrained("ckiplab/bert-base-chinese")

all_preds = []
all_labels = []

def predict_single_sentence(model, tokenizer, sentence, max_len=256):
    model.eval()
    with torch.no_grad():
        # 清洗文字
        sentence = re.sub(r"\s+", "", sentence)
        sentence = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9。，！？]", "", sentence)

        # 編碼
        encoded = tokenizer(sentence,
                            return_tensors="pt",
                            truncation=True,
                            padding="max_length",
                            max_length=max_len)

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        token_type_ids = encoded["token_type_ids"].to(device)

        # 預測
        output = model(input_ids, attention_mask, token_type_ids)
        prob = output.item()
        label = int(prob > 0.5)

        # 風險分級顯示
        if prob > 0.9:
            risk = "🔴 高風險（極可能是詐騙）"
        elif prob > 0.5:
            risk = "🟡 中風險（可疑）"
        else:
            risk = "🟢 低風險（正常）"

        pre_label = ""
        if label == 1:
            pre_label = '詐騙'
        else:
            pre_label = '正常'
   
        print(f"\n📩 訊息內容：{sentence}")
        print(f"✅ 預測結果：{'詐騙' if label == 1 else '正常'}")
        print(f"📊 信心值：{round(prob*100, 2)}")
        print(f"⚠️ 風險等級：{risk}")
        return pre_label, prob, risk


def analyze_text(text):
    label, prob, risk = predict_single_sentence(model, tokenizer, text)
    return {
        "status": label,
        "confidence": round(prob*100, 2),  
        "suspicious_keywords": [risk]
    }

