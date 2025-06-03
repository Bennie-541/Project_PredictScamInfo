"""
流程圖
讀取資料 → 分割資料 → 編碼 → 建立 Dataset / DataLoader
↓
建立模型(BERT+LSTM+CNN)
        ↓
        BERT 輸出 [batch, seq_len, 768]
        ↓
        BiLSTM  [batch, seq_len, hidden_dim*2]
        ↓
        CNN 模組 (Conv1D + Dropout + GlobalMaxPooling1D)
        ↓
        Linear 分類器(輸出詐騙機率)
        ↓
訓練模型(Epochs)
↓
評估模型(Accuracy / F1 / Precision / Recall)
↓
儲存模型(.pth)

"""
#引入重要套件Import Library
import os
import torch                            #   PyTorch 主模組               
import torch.nn as nn                   #	神經網路相關的層(例如 LSTM、Linear)        
import pandas as pd
import re

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset #	提供 Dataset、DataLoader 類別
from transformers import BertTokenizer # BertTokenizer把文字句子轉換成 BERT 格式的 token ID,例如 [CLS] 今天 天氣 不錯 [SEP] → [101, 1234, 5678, ...]
from sklearn.model_selection import train_test_split
from transformers import BertModel

# ------------------- 載入 .env 環境變數 -------------------
load_dotenv()
base_dir = os.getenv("DATA_DIR", "./data")  # 如果沒設環境變數就預設用 ./data

# ------------------- 使用相對路徑找 CSV -------------------
#,os.path.join(base_dir, "NorANDScamInfo_data1.csv"),os.path.join(base_dir, "ScamInfo_data1.csv"),os.path.join(base_dir, "NormalInfo_data1.csv")
#如有需要訓練複數筆資料可以使用這個方法csv_files = [os.path.join(base_dir, "檔案名稱1.csv"),os.path.join(base_dir, "檔案名稱2.csv")]
#程式碼一至131行

# GPU 記憶體限制(可選)
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:16"

#資料前處理
class BertPreprocessor:
    def __init__(self, tokenizer_name="ckiplab/bert-base-chinese", max_len=128):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def load_and_clean(self, filepath):
        #載入 CSV 並清理 message 欄位。
        df = pd.read_csv(filepath)
        df = df.dropna().drop_duplicates().reset_index(drop=True)
        # 文字清理:移除空白、保留中文英數與標點
        df["message"] = df["message"].astype(str)
        df["message"] = df["message"].apply(lambda text: re.sub(r"\s+", "", text))
        df["message"] = df["message"].apply(lambda text: re.sub(r"[^\u4e00-\u9fffA-Za-z0-9。,！？]", "", text))
        return df[["message", "label"]]  # 保留必要欄位

    def encode(self, messages):
        #使用 HuggingFace BERT Tokenizer 將訊息編碼成Bert模型輸入格式。
        return self.tokenizer(
            list(messages),
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_len
        )
#自動做資料前處理
def build_bert_inputs(files):
    #將正常與詐騙資料分別指定 label,統一清理、編碼,回傳模型可用的 input tensors 與 labels。
    processor = BertPreprocessor()
    dfs = []
    # 合併正常 + 詐騙檔案清單
    all_files = files

    for filepath in all_files:
        df = processor.load_and_clean(filepath)
        dfs.append(df)
    
    # 合併所有資料。在資料清理過程中dropna():刪除有空值的列,drop_duplicates():刪除重複列,filter()或df[...]做條件過濾,concat():將多個 DataFrame合併
    # 這些操作不會自動重排索引,造成索引亂掉。
    # 合併後統一編號(常見於多筆資料合併)all_df = pd.concat(dfs, 關鍵-->ignore_index=True)
    all_df = pd.concat(dfs, ignore_index=True)
    print(f"✅ 已讀入 {len(all_df)} 筆資料")
    print(all_df["label"].value_counts())
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


#定義 PyTorch Dataset 類別。ScamDataset 繼承自 torch.utils.data.Dataset
#將 BERT 輸出的 token 與對應標籤封裝成 PyTorch 能使用的格式
class ScamDataset(Dataset):
    def __init__(self, inputs, labels):
        self.input_ids = inputs["input_ids"]                           # input_ids:句子的 token ID;attention_mask:注意力遮罩(0 = padding)
        self.attention_mask = inputs["attention_mask"]                 # token_type_ids:句子的 segment 區分
        self.token_type_ids = inputs["token_type_ids"]                 # torch.tensor(x, dtype=...)將資料(x)轉為Tensor的標準做法。
        self.labels = torch.tensor(labels.values, dtype=torch.float32) # x可以是 list、NumPy array、pandas series...
# dtypefloat32:浮點數(常用於 回歸 或 BCELoss 二分類);long:整數(常用於 多分類 搭配 CrossEntropyLoss)。labels.values → 轉為 NumPy array
    
    def __len__(self):          # 告訴 PyTorch 這個 Dataset 有幾筆資料
        return len(self.labels) # 給 len(dataset) 或 for i in range(len(dataset)) 用的
    
    def __getitem__(self, idx): #每次調用 __getitem__() 回傳一筆 {input_ids, attention_mask, token_type_ids, labels}
        return {                #DataLoader 每次會呼叫這個方法多次來抓一個 batch 的資料
            "input_ids":self.input_ids[idx],
            "attention_mask":self.attention_mask[idx],
            "token_type_ids":self.token_type_ids[idx],
            "labels":self.labels[idx]
        }

# 這樣可以同時處理 scam 和 normal 資料,不用重複寫清理與 token 處理
if __name__ == "__main__":
    csv_files = [os.path.join(base_dir, "NorANDScamInfo_data3k.csv")]
    train_inputs, train_labels, val_inputs, val_labels, processor = build_bert_inputs(csv_files)
    
    train_dataset = ScamDataset(train_inputs, train_labels)
    val_dataset = ScamDataset(val_inputs, val_labels)

    # batch_size每次送進模型的是 8 筆資料(而不是一筆一筆)
    # 每次從 Dataset 中抓一批(batch)資料出來
    train_loader = DataLoader(train_dataset, batch_size=8)
    val_loader = DataLoader(val_dataset, batch_size=8)

"""
class BertLSTM_CNN_Classifier(nn.Module)表示:你定義了一個子類別,
繼承自 PyTorch 的基礎模型類別 nn.Module。

若你在 __init__() 裡沒有呼叫 super().__init__(),
那麼父類別 nn.Module 的初始化邏輯(包含重要功能)就不會被執行,
導致整個模型運作異常或錯誤。
"""

# nn.Module是PyTorch所有神經網路模型的基礎類別,nn.Module 是 PyTorch 所有神經網路模型的基礎類別
class BertLSTM_CNN_Classifier(nn.Module):
    
    def __init__(self, hidden_dim=128, num_layers=1, dropout=0.3):
        
        # super()是Python提供的一個方法,用來呼叫「父類別的版本」的方法。
        # 呼叫:super().__init__()讓父類別(nn.Module)裡面那些功能、屬性都被正確初始化。
        # 沒super().__init__(),這些都不會正確運作,模型會壞掉。
        # super() 就是 Python 提供給「子類別呼叫父類別方法」的方式
        super().__init__()
        
        # 載入中文預訓練的 BERT 模型,輸入為句子token IDs,輸出為每個 token 的向量,大小為 [batch, seq_len, 768]。
        self.bert = BertModel.from_pretrained("ckiplab/bert-base-chinese") # 這是引入hugging face中的tranceformat
        
        # 接收BERT的輸出(768 維向量),進行雙向LSTM(BiLSTM)建模,輸出為 [batch, seq_len, hidden_dim*2],例如 [batch, seq_len, 256]
        """
        LSTM 接收每個token的768維向量(來自 BERT)作為輸入,
        透過每個方向的LSTM壓縮成128維的語意向量。
        由於是雙向LSTM,會同時從左到右(前向)和右到左(後向)各做一次,
        最後將兩個方向的輸出合併為256維向量(128×2)。
        每次處理一個 batch(例如 8 句話),一次走完整個時間序列。
        """
        self.LSTM = nn.LSTM(input_size=768,        
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        
         # CNN 模組:接在 LSTM 後的輸出上。將LSTM的輸出轉成卷積層格式,適用於Conv1D,CNN可學習位置不變的局部特徵。
        self.conv1 =  nn.Conv1d(in_channels=hidden_dim*2,
                                out_channels=128,
                                kernel_size=3,  # 這裡kernel_size=3 為 3-gram 特徵
                                padding=1)
        
        self.dropout = nn.Dropout(dropout) # 隨機將部分神經元設為 0,用來防止 overfitting。
        
        self.global_maxpool = nn.AdaptiveAvgPool1d(1)  #將一整句話的特徵濃縮成一個固定大小的句子表示向量
        
        # 將CNN輸出的128維特徵向量輸出為一個「機率值」(詐騙或非詐騙)。
        self.classifier = nn.Linear(128,1)
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        #BERT 編碼
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        #.last_hidden_state是BertModel.from_pretrained(...)內部的key，會輸出    [batch, seq_len, 768]
        hidden_states = outputs.last_hidden_state 

        # 送入 BiLSTM
        # transpose(1, 2) 的用途是：讓 LSTM 輸出的資料形狀符合 CNN 所要求的格式
        #   假設你原本 LSTM 輸出是：    [batch_size, seq_len, hidden_dim*2] = [8, 128, 256]
        # 但CNN(Conv1d)的輸入格式需要是：[batch_size, in_channels, seq_len] = [8, 256, 128]
        # 因此你需要做：.transpose(1, 2)把 seq_len 和 hidden_dim*2 調換
        LSTM_out, _ = self.LSTM(hidden_states)     # [batch, seq_len, hidden_dim*2]
        LSTM_out = LSTM_out.transpose(1, 2)        # [batch, hidden_dim*2, seq_len]

        # 卷積 + Dropout
        x = self.conv1(LSTM_out)                   # [batch, 128, seq_len]
        x = self.dropout(x)
        
        #全局池化
        # .squeeze(dim) 的作用是：把某個「維度大小為 1」的維度刪掉
        # x = self.global_maxpool(x).squeeze(2)   # 輸出是 [batch, 128, 1]
        # 不 .squeeze(2)，你會得到 shape 為 [batch, 128, 1]，不方便後面接 Linear。
        # .squeeze(2)=拿掉第 2 維（數值是 1） → 讓形狀變成 [batch, 128]
        x = self.global_maxpool(x).squeeze(2)      # [batch, 128]

        #分類 & Sigmoid 機率輸出
        logits = self.classifier(x)
        
        #.sigmoid() → 把 logits 轉成 0~1 的機率.squeeze() → 變成一維 [batch] 長度的機率 list
        """例如：
logits = [[0.92], [0.05], [0.88], [0.41], ..., [0.17]]
→ sigmoid → [[0.715], [0.512], ...]
→ squeeze → [0.715, 0.512, ...]
"""
        return torch.sigmoid(logits).squeeze() # 最後輸出是一個值介於 0 ~ 1 之間,代表「為詐騙訊息的機率」。
        
# 設定 GPU 裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = BertLSTM_CNN_Classifier().to(device)
# 定義 optimizer 和損失函數
optimizer = torch.optim.Adam(model.parameters(),lr=2e-5)
criterion = nn.BCELoss()

#只保留推論即可,模型訓練應該在本地完成！
if os.path.exists("model.pth"):
    print("✅ 已找到 model.pth,載入模型跳過訓練")
    model.load_state_dict(torch.load("model.pth", map_location=device))
else:
    print("❌ 未找到 model.pth")

# 本機訓練迴圈,要訓練再取消註解,否則在線上版本一律處於註解狀態
"""
if __name__ == "__main__": # 只有當我「直接執行這個檔案」時,才執行以下訓練程式(不是被別人 import 使用時)。
    if os.path.exists("model.pth"):
        print("✅ 已找到 model.pth,載入模型跳過訓練")
        model.load_state_dict(torch.load("model.pth", map_location=device))
    else:
        print("🚀 未找到 model.pth,開始訓練模型...")
        num_epochs = 15 # batch_size設定在train_loader和test_loader那
        for epoch in range(num_epochs):
            model.train() # 從nn.Module繼承的方法。將模型設為「訓練模式」,有些層(像 Dropout 或 BatchNorm)會啟用訓練行為。
            total_loss = 0.0
            for batch in train_loader:
            
                # 清理舊梯度,以免累加。為甚麼要?因為PyTorch 預設每次呼叫 .backward() 都會「累加」梯度(不會自動清掉)
                # 沒 .zero_grad(),梯度會越累積越多,模型會亂掉。
                optimizer.zero_grad() 
                
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids, attention_mask, token_type_ids)
                
                loss = criterion(outputs, labels) # 比較 預測結果 outputs(Sigmoid 的機率)和 真實答案 labels
                
                # 用鏈式法則(Chain Rule)計算每一層「參數對 loss 的影響」,也就是梯度
                # PyTorch 利用自動微分(autograd)幫你計算整個計算圖的偏導數,然後存在每一層的 .grad 裡。
                loss.backward()
                
                # 用 .grad 中的梯度資訊根
                # 據學習率和優化器的規則
                # 改變每一個參數的值,以讓下一次預測更接近真實
                optimizer.step()
                
                # loss 是一個 tensor(需要 backward);.item() 把它轉成 Python 的純數字(float)
                total_loss += loss.item()
            print(f"[Epoch{epoch+1}]Training Loss:{total_loss:.4f}")
        torch.save(model.state_dict(), "model.pth")# 儲存模型權重
        print("✅ 模型訓練完成並儲存為 model.pth")
"""

"""
整個模型中每一個文字(token)始終是一個向量,隨著層數不同,這個向量代表的意義會更高階、更語意、更抽象。
在整個 BERT + LSTM + CNN 模型的流程中,「每一個文字(token)」都會被表示成一個「向量」來進行後續的計算與學習。
今天我輸入一個句子:"早安你好,吃飯沒"
BERT 的輸入包含三個部分:input_ids、attention_mask、token_type_ids,
這些是 BERT 所需的格式。BERT 會將句子中每個 token 編碼為一個 768 維的語意向量,

進入 BERT → 每個 token 變成語意向量:
BERT 輸出每個字為一個 768 維的語意向量
「早」 → [0.23, -0.11, ..., 0.45]   長度為 768
「安」 → [0.05, 0.33, ..., -0.12]   一樣 768
...
batch size 是 8,句子長度是 8,輸出 shape 為:    
[batch_size=8, seq_len=8, hidden_size=768]

接下來這些向量會輸入到 LSTM,LSTM不會改變「一個token是一個向量」的概念,而是重新表示每個token的語境向量。
把每個原本 768 維的 token 壓縮成 hidden_size=128,雙向 LSTM → 拼接 → 每個 token 成為 256 維向量:

input_size=768 是從 BERT 接收的向量維度
hidden_size=128 表示每個方向的 LSTM 會把 token 壓縮為 128 維語意向量
num_layers=1 表示只堆疊 1 層 LSTM
bidirectional=True 表示是雙向

LSTM,除了從左讀到右,也會從右讀到左,兩個方向的輸出會合併(拼接),變成:
[batch_size=8, seq_len=8, hidden_size=256]  # 因為128*2

接下來進入 CNN,CNN 仍然以「一個向量代表一個字」的形式處理:

in_channels=256(因為 LSTM 是雙向輸出)

out_channels=128 表示學習出 128 個濾波器,每個濾波器專門抓一種 n-gram(例如「早安你」),每個「片段」的結果輸出為 128 維特徵

kernel_size=3 表示每個濾波器看 3 個連續 token(像是一個 3-gram)或,把相鄰的 3 個字(各為 256 維)一起掃描

padding=1 為了保留輸出序列長度和輸入相同,避免邊界資訊被捨棄

CNN 輸出的 shape 就會是:

[batch_size=8, out_channels=128, seq_len=8],還是每個 token 有對應一個向量(只是這向量是 CNN 抽出的新特徵)

"""