#pip install transformers
#pip install torch         //transformers 套件需要
#pip install scikit-learn
#pip install transformers torch
#
# bert_explainer.py


#引入重要套件Import Library
import torch                            #   PyTorch 主模組               
import torch
from AI_Model_architecture import BertLSTM_CNN_Classifier, BertPreprocessor #	用來產生模擬資料
from transformers import BertTokenizer
import re
#BertTokenizer	把文字句子轉換成 BERT 格式的 token ID，例如 [CLS] 今天 天氣 不錯 [SEP] → [101, 1234, 5678, ...]

# 裝置選擇
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型架構並載入訓練權重
model = BertLSTM_CNN_Classifier()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
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

