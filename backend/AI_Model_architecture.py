
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
        
        #檢查空值;計算空值數量，若有空值才執行 dropna()
        n_missing = clean_df.isnull().sum().sum()
        if n_missing > 0:
            clean_df = self.df.dropna()
            print(f"已清除空值，尚有空值？ {clean_df.isnull().values.any()}")
            print(f"已清除 {n_missing} 筆空值")
            
        #檢查重複值;計算重複筆數，若有才刪除
        n_dup = clean_df.duplicated().sum()
        if n_dup > 0 :
            clean_df = clean_df.drop_duplicates()
            print("已清除重複資料")
            print(f"已清除 {n_dup} 筆重複資料")
            
        #下列做文字前處理
        # 🔸 確保 message 欄位為字串（避免某些為 float 造成錯誤）
        clean_df["message"] = clean_df["message"].astype(str)
        
        # 對 message 欄位進行正規化處理（逐筆文字處理; 移除所有空白字元（包含換行、Tab 等）
        clean_df.loc[:,"message"] = clean_df["message"].apply(lambda text: re.sub(r"\s+", "", str(text)))  
        # 僅保留常用中文字、英數與標點符號，移除奇怪符號或 emoji
        clean_df.loc[:,"message"] = clean_df["message"].apply(lambda text: re.sub(r"[^\u4e00-\u9fffA-Za-z0-9。，！？]", "", str(text)))  
        # 清理後再 reset index，確保索引連續
        return clean_df.reset_index(drop=True)
        
        # 使用 HuggingFace 的 BertTokenizer 對每一筆文字進行分詞與編碼
    def tokenize_messages(self, cleaned_df, max_len=256):
        return self.tokenizer(list(cleaned_df["message"]), 
                                return_tensors="pt",        # 指定回傳為 PyTorch tensor 格式（可改為 'tf' 如果你用 TensorFlow）
                                truncation=True,            # 若句子太長，會自動截斷超過 max_length 的部分
                                padding='max_length',       # 若句子太短，會自動補齊到 max_length（使用特殊填充 token）
                                max_length=256              # 固定每筆資料的長度為 256 個 token，過短則補齊，過長則截斷
                            )
                                

    

df_cleaner = clean_data(df_data)           # 初始化前處理器

df_cleaned = df_cleaner.Data_Cleaning()    # 先執行資料清洗（移除空值、重複，清理 message）

bert_inputs = df_cleaner.tokenize_messages(df_cleaned)  # 將清理後資料丟進 tokenizer，得到 BERT 輸入格式

# 檢查第一筆的編碼對應文字
tokens = bert_inputs["input_ids"][0]
token_strs = df_cleaner.tokenizer.convert_ids_to_tokens(tokens)
print(token_strs)