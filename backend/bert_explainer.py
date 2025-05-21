#pip install transformers
#pip install torch         //transformers 套件需要
#pip install scikit-learn
#pip install transformers torch
#
# bert_explainer.py


#引入重要套件Import Library
import torch                            #   PyTorch 主模組               
import torch.nn as nn                   #	神經網路相關的層（例如 LSTM、Linear）
import torch.nn.functional as F         #   提供純函式版的操作方法，像是 F.relu()、F.cross_entropy()，通常不帶參數、不自動建立權重
import numpy as np                      #	用來產生模擬資料
from transformers import BertTokenizer, BertForSequenceClassification

#BertTokenizer	把文字句子轉換成 BERT 格式的 token ID，例如 [CLS] 今天 天氣 不錯 [SEP] → [101, 1234, 5678, ...]
##BertForSequenceClassification	是 Hugging Face 提供的一個完整 BERT 模型，接了分類用的 Linear 層，讓你直接拿來做分類任務（例如詐騙 vs 正常）

from torch.utils.data import DataLoader, Dataset #	提供 Dataset、DataLoader 類別


#供參考用，正常，測試前後端輸入、輸出是否正常，之後會刪除
def analyze_text(text):
    length = len(text)

    return {
        "status": "目前為測試階段",
        "confidence": length,  # 直接用字數當作假可信度
        "suspicious_keywords": [f"目前為測試階段，將回傳輸入內容: {text}"]
    }

#假設我們有 100 筆句子資料，每筆是長度 50 的序列（已編碼為整數）
#X_data: 已轉成數字的 padded 文字序列（例如 [12, 41, 3, 0, 0, ...]）
#y_data: 對應標籤（如 0 / 1）
class TextDataset(Dataset):     #Dataset 定義（訓練資料格式）
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)       # 將 X 轉為 tensor，且型別為 long（int64）
        self.y = torch.tensor(y, dtype=torch.float32)    # y 為 float32，因為要搭配 BCELoss
        
    def __len__(self):                                   # 回傳資料筆數
        return len(self.X)
    
    def __getitem__(self, idx):                          # 根據索引回傳一筆資料
        return self.X[idx], self.y[idx]

# 模擬資料
X_data = np.random.randint(1, 1000, size=(100, 50))  # 100 筆，每筆長度 50 (100, 50) 整數序列
y_data = np.random.randint(0, 2, size=(100,))        # 對應標籤（0 or 1）

dataset = TextDataset(X_data, y_data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True) #每次取出 16 筆資料做訓練;	每一 epoch 隨機打亂資料順序


# 建立模型：RNN / LSTM 範本
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, use_lstm=True):
        super(RNNClassifier, self).__init__()
        # 嵌入層：將整數序列轉為向量（詞嵌入）
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn_type = 'LSTM' if use_lstm else 'RNN'
        # 建立 RNN 或 LSTM 層
        if use_lstm:
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)                    # [batch, seq, embed]   輸入轉向量：[batch, seq] -> [batch, seq, embed]
        output, (hn, cn) = self.rnn(embedded) if self.rnn_type == 'LSTM' else (self.rnn(embedded)[0], None) # LSTM 輸出最後 hidden state hn
        out = self.fc(hn[-1])                           # [batch, hidden] → [batch, 1]# hn[-1]: 最後一層的 hidden state → [batch, 1]
        return self.sigmoid(out).squeeze(1)


# 3. 訓練模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNNClassifier(vocab_size=1000, embed_dim=128, hidden_dim=64, use_lstm=True).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        preds = model(X_batch)                  # 預測
        loss = criterion(preds, y_batch)        # 計算損失
        loss.backward()                         # 反向傳播
        optimizer.step()                        # 更新權重
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

#評估範例
model.eval()
with torch.no_grad():
    test_sample = torch.tensor(X_data[0:1], dtype=torch.long).to(device)
    pred = model(test_sample)
    print("Prediction:", pred.item())
