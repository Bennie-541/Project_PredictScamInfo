import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

def test_firebase_connection():
    try:
        # 初始化 Firebase
        cred = credentials.Certificate("firebase-credentials.json")
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        
        # 測試寫入
        test_data = {
            "test_field": "測試資料",
            "timestamp": datetime.now()
        }
        
        # 寫入測試資料
        doc_ref = db.collection('test').document('test_doc')
        doc_ref.set(test_data)
        
        # 讀取測試資料
        doc = doc_ref.get()
        if doc.exists:
            print("Firebase 連接測試成功！")
            print("測試資料：", doc.to_dict())
        else:
            print("無法讀取測試資料")
            
    except Exception as e:
        print(f"Firebase 連接測試失敗：{str(e)}")

if __name__ == "__main__":
    test_firebase_connection() 