
#引入重要套件Import Library
import torch                            #   PyTorch 主模組               
import torch.nn as nn                   #	神經網路相關的層（例如 LSTM、Linear）
import torch.nn.functional as F         #   提供純函式版的操作方法，像是 F.relu()、F.cross_entropy()，通常不帶參數、不自動建立權重
import numpy as np                      
import pandas as pd
import os 
import re

from torch.utils.data import DataLoader, Dataset #	提供 Dataset、DataLoader 類別
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

#BertTokenizer	把文字句子轉換成 BERT 格式的 token ID，例如 [CLS] 今天 天氣 不錯 [SEP] → [101, 1234, 5678, ...]
##BertForSequenceClassification	是 Hugging Face 提供的一個完整 BERT 模型，接了分類用的 Linear 層，讓你直接拿來做分類任務（例如詐騙 vs 正常）

#df1 = pd.read_csv("D:\Project_PredictScamInfo\data\ScamInfo_data1.csv")
#print(f"檢查:{df1.duplicated().sum()}")

df_data = pd.read_csv(r"D:\Project_PredictScamInfo\data\ScamInfo_data1.csv")

#資料前處裡
class clean_data:
    def __init__(self, df):
        self.df = df
        self.tokenizer = BertTokenizer.from_pretrained("ckiplab/bert-base-chinese")
        
    def Data_Cleaning(self):
        clean_df = self.df.copy()
        
        #檢查空值
        if clean_df.isnull().sum().sum() > 0:
            clean_df = self.df.dropna()
            print(f"已清除空值，尚有空值？ {clean_df.isnull().values.any()}")
        
        #檢查重複值
        if clean_df.duplicated().sum() > 0 :
            clean_df = clean_df.drop_duplicates()
            print("已清除重複資料")
        
        #下列做文字前處理
        # 對 message 欄位進行正規化處理（逐筆文字處理）
        clean_df["message"] = clean_df["message"].apply(lambda text: re.sub(r"\s+", "", str(text)))  # 移除所有空白
        clean_df["message"] = clean_df["message"].apply(lambda text: re.sub(r"[^\u4e00-\u9fffA-Za-z0-9。，！？]", "", str(text)))  # 保留中英文、數字、標點
        
        # 使用 HuggingFace 的 BertTokenizer 對每一筆文字進行分詞與編碼
        C_text = self.tokenizer(clean_df, 
                                return_tensors="pt",        # 指定回傳為 PyTorch tensor 格式（可改為 'tf' 如果你用 TensorFlow）
                                truncation=True,            # 若句子太長，會自動截斷超過 max_length 的部分
                                padding='max_length',       # 若句子太短，會自動補齊到 max_length（使用特殊填充 token）
                                max_length=256              # 固定每筆資料的長度為 256 個 token，過短則補齊，過長則截斷
                                )
                                
        return C_text
    
df_ScamInfo = clean_data(df_data)
df_ScamInfo_cleaned = df_ScamInfo.Data_Cleaning()
print(df_ScamInfo_cleaned.head())