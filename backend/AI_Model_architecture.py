#引入重要套件Import Library
import torch                            #   PyTorch 主模組               
import torch.nn as nn                   #	神經網路相關的層（例如 LSTM、Linear）
import torch.nn.functional as F         #   提供純函式版的操作方法，像是 F.relu()、F.cross_entropy()，通常不帶參數、不自動建立權重
import numpy as np                      
import pandas as pd
import os 
import re

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset #	提供 Dataset、DataLoader 類別
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
#BertTokenizer	把文字句子轉換成 BERT 格式的 token ID，例如 [CLS] 今天 天氣 不錯 [SEP] → [101, 1234, 5678, ...]
##BertForSequenceClassification	是 Hugging Face 提供的一個完整 BERT 模型，接了分類用的 Linear 層，讓你直接拿來做分類任務（例如詐騙 vs 正常）


#正常訊息資料集在這新增
normal_files = [r"D:\Project_PredictScamInfo\data\NormalInfo_data1.csv"]

#詐騙訊息資料集在這新增
scam_files = [
    r"D:\Project_PredictScamInfo\data\ScamInfo_data1.csv"]

#資料前處理
class BertPreprocessor:
    def __init__(self, tokenizer_name="ckiplab/bert-base-chinese", max_len=256):
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

    # 進行 BERT tokenizer 編碼
    bert_inputs = processor.encode(all_df["message"])

    return bert_inputs

#AUTO YA~以for迴圈自動新增個別變數內，build_bert_inputs能自動擷取新增資料
normal_files_labels = [normal for normal in normal_files] 
scam_files_labels = [scam for scam in scam_files] 
# 這樣可以同時處理 scam 和 normal 資料，不用重複寫清理與 token 處理
bert_inputs= build_bert_inputs(normal_files, scam_files)

#定義 PyTorch Dataset 類別
class ScamDataset(Dataset):
    def __init__(self, inputs, labels):
        self.input_ids = inputs["input_ids"]
        self.attention_mask = inputs["attention_mask"]
        self.token_type_ids = inputs["token_type_ids"]
        self.labels = torch.tensor(labels.values, dtype=torch.float32) # 若為 binary classification
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {            
            "input_ids":self.input_ids[idx],
            "attention_mask":self.attention_mask[idx],
            "token_type_ids":self.token_type_ids[idx],
            "labels":self.labels[idx]
        }

#製作 train/val 資料集與 DataLoader
train_texts, val_texts, train_labels, val_labels = train_test_split(
    bert_inputs["message"], bert_inputs["label"],
    stratify=bert_inputs["label"],
    test_size=0.2,
    random_state=25,
    shuffle=True
)