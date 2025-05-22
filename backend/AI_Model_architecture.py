"""流程圖
讀取資料 → 分割資料 → 編碼 → 建立 Dataset / DataLoader
↓
建立模型（BERT+LSTM+CNN）
        ↓
        BERT 輸出 [batch, seq_len, 768]
        ↓
        BiLSTM  [batch, seq_len, hidden_dim*2]
        ↓
        CNN 模組 (Conv1D + Dropout + GlobalMaxPooling1D)
        ↓
        Linear 分類器（輸出詐騙機率）
        ↓
訓練模型（Epochs）
↓
評估模型（Accuracy / F1 / Precision / Recall）
↓
儲存模型（.pth）

"""#引入重要套件Import Library
import torch                            #   PyTorch 主模組               
import torch.nn as nn                   #	神經網路相關的層（例如 LSTM、Linear）
import torch.nn.functional as F         #   提供純函式版的操作方法，像是 F.relu()、F.cross_entropy()，通常不帶參數、不自動建立權重
import numpy as np                      
import pandas as pd
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:16"#讓 CUDA 使用「更小記憶體分配塊」的方法，能有效減少 OOM 錯誤。
import re

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset #	提供 Dataset、DataLoader 類別
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from transformers import BertModel
#BertTokenizer	把文字句子轉換成 BERT 格式的 token ID，例如 [CLS] 今天 天氣 不錯 [SEP] → [101, 1234, 5678, ...]
##BertForSequenceClassification	是 Hugging Face 提供的一個完整 BERT 模型，接了分類用的 Linear 層，讓你直接拿來做分類任務（例如詐騙 vs 正常）


#正常訊息資料集在這新增
normal_files = [r"D:\Project_PredictScamInfo\data\NormalInfo_data1.csv"]

#詐騙訊息資料集在這新增
scam_files = [
    r"D:\Project_PredictScamInfo\data\ScamInfo_data1.csv"]

#資料前處理
class BertPreprocessor:
    def __init__(self, tokenizer_name="ckiplab/bert-base-chinese", max_len=128):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def load_and_clean(self, filepath):
        #載入 CSV 並清理 message 欄位。
        df = pd.read_csv(filepath)
        df = df.dropna().drop_duplicates().reset_index(drop=True)
        # 文字清理：移除空白、保留中文英數與標點
        df["message"] = df["message"].astype(str)
        df["message"] = df["message"].apply(lambda text: re.sub(r"\s+", "", text))
        df["message"] = df["message"].apply(lambda text: re.sub(r"[^\u4e00-\u9fffA-Za-z0-9。，！？]", "", text))
        return df[["message", "label"]]  # 保留必要欄位

    def encode(self, messages):
        #使用 HuggingFace BERT Tokenizer 將訊息編碼成模型輸入格式。
        return self.tokenizer(
            list(messages),
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_len
        )
#自動做資料前處理
def build_bert_inputs(normal_files, scam_files):
    #將正常與詐騙資料分別指定 label，統一清理、編碼，回傳模型可用的 input tensors 與 labels。
    processor = BertPreprocessor()
    dfs = []
    # 合併正常 + 詐騙檔案清單
    all_files = normal_files + scam_files

    for filepath in all_files:
        df = processor.load_and_clean(filepath)
        dfs.append(df)

    # 合併所有資料。在資料清理過程中dropna()：刪除有空值的列，drop_duplicates()：刪除重複列，filter()或df[...]做條件過濾，concat():將多個 DataFrame合併
    # 這些操作不會自動重排索引，造成索引亂掉。
    # 合併後統一編號（常見於多筆資料合併）all_df = pd.concat(dfs, 關鍵-->ignore_index=True)
    all_df = pd.concat(dfs, ignore_index=True)
    #製作 train/val 資料集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
    all_df["message"], all_df["label"],
    stratify=all_df["label"],
    test_size=0.2,
    random_state=25,
    shuffle=True
    )
    
    # 進行 BERT tokenizer 編碼
    train_inputs = processor.encode(train_texts)
    val_inputs = processor.encode(val_texts)

    return train_inputs, train_labels, val_inputs, val_labels, processor

#AUTO YA~以for迴圈自動新增個別變數內，build_bert_inputs能自動擷取新增資料
normal_files_labels = [normal for normal in normal_files] 
scam_files_labels = [scam for scam in scam_files] 

#print(bert_inputs.keys())

#定義 PyTorch Dataset 類別
class ScamDataset(Dataset):
    def __init__(self, inputs, labels):
        self.input_ids = inputs["input_ids"]                           # input_ids：句子的 token ID; attention_mask：注意力遮罩（0 = padding）
        self.attention_mask = inputs["attention_mask"]                 # token_type_ids：句子的 segment 區分
        self.token_type_ids = inputs["token_type_ids"]                 # torch.tensor(x, dtype=...)將資料(x)轉為Tensor的標準做法。
        self.labels = torch.tensor(labels.values, dtype=torch.float32) # x可以是 list、NumPy array、pandas series...
# dtypefloat32：浮點數(常用於 回歸 或 BCELoss 二分類);long：整數(常用於 多分類 搭配 CrossEntropyLoss)。labels.values → 轉為 NumPy array
    def __len__(self):          # 告訴 PyTorch 這個 Dataset 有幾筆資料
        return len(self.labels) # 給 len(dataset) 或 for i in range(len(dataset)) 用的
    
    def __getitem__(self, idx): #回傳第 idx 筆資料（會自動在訓練中一筆筆抓）
        return {                #DataLoader 每次會呼叫這個方法多次來抓一個 batch 的資料
            "input_ids":self.input_ids[idx],
            "attention_mask":self.attention_mask[idx],
            "token_type_ids":self.token_type_ids[idx],
            "labels":self.labels[idx]
        }

# 這樣可以同時處理 scam 和 normal 資料，不用重複寫清理與 token 處理
train_inputs, train_labels, val_inputs, val_labels, processor = build_bert_inputs(normal_files, scam_files)

train_dataset = ScamDataset(train_inputs, train_labels)
val_dataset = ScamDataset(val_inputs, val_labels)

train_loader = DataLoader(train_dataset, batch_size=8)
val_loader = DataLoader(val_dataset, batch_size=8)

#模型
class BertLSTM_CNN_Classifier(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=1, dropout=0.3):
        super(BertLSTM_CNN_Classifier, self).__init__()
        self.bert = BertModel.from_pretrained("ckiplab/bert-base-chinese") #載入預訓練 BERT 模型（ckiplab 中文版）
        # LSTM 接在 BERT 的 token 輸出後（輸入是768維）
        self.LSTM = nn.LSTM(input_size=768,         # 把 BERT 的 token 序列再交給雙向 LSTM 做時間序列建模
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
         # CNN 模組：接在 LSTM 後的輸出上
        self.conv1 =  nn.Conv1d(in_channels=hidden_dim*2,
                                out_channels=128,
                                kernel_size=3,
                                padding=1)
        self.dropout = nn.Dropout(dropout) 
        self.global_maxpool = nn.AdaptiveAvgPool1d(1)        # 等效於 GlobalMaxPooling1D

        self.classifier = nn.Linear(128,1)
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, 768]

        LSTM_out, _ = self.LSTM(hidden_states)     # [batch, seq_len, hidden_dim*2]
        LSTM_out = LSTM_out.transpose(1, 2)        # [batch, hidden_dim*2, seq_len]

        x = self.conv1(LSTM_out)                   # [batch, 128, seq_len]
        x = self.dropout(x)
        x = self.global_maxpool(x).squeeze(2)      # [batch, 128]

        logits = self.classifier(x)
        return torch.sigmoid(logits).squeeze()
        
# 設定 GPU 裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 設定使用的最大執行緒數（視 CPU 而定）
torch.set_num_threads(8)  # 建議設成你系統的實體核心數
# 初始化模型
model = BertLSTM_CNN_Classifier().to(device)
# 定義 optimizer 和損失函數
optimizer = torch.optim.Adam(model.parameters(),lr=2e-5)
criterion = nn.BCELoss()

# 訓練迴圈
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    
    for batch in train_loader:
        optimizer.zero_grad()
        #將輸入搬到 GPU（如果有的話）
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)
         # 前向傳播
        outputs = model(input_ids, attention_mask, token_type_ids)
        # 計算 loss 並反向傳播
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[Epoch{epoch+1}]Training Loss:{total_loss:.4f}")
        


# 儲存模型權重
torch.save(model.state_dict(), "model.pth")
print(" 模型已儲存為 model.pth")
