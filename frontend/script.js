// frontend/script.js

// 當 DOM (文件物件模型) 完全載入後才執行裡面的程式碼
document.addEventListener('DOMContentLoaded', () => {
    // 取得 HTML 元素
    const predictInfoTextarea = document.getElementById('predict_info'); // 輸入訊息的文字區域
    const detectButton = document.getElementById('detect_button');     // 檢測按鈕
    const clearButton = document.getElementById('clear_button');       // 清除按鈕

    // 取得顯示結果的 HTML 元素
    const isScamSpan = document.getElementById('is_scam');             // 顯示是否為詐騙
    const confidenceScoreSpan = document.getElementById('confidence_score'); // 顯示模型預測可信度
    const suspiciousPhrasesDiv = document.getElementById('suspicious_phrases'); // 顯示可疑詞句列表

    // 後端 FastAPI API 的 URL
    // 在開發階段，通常是 http://127.0.0.1:8000 或 http://localhost:8000
    // 請根據你實際運行 FastAPI 的位址和 Port 進行設定
    const API_URL = 'http://127.0.0.1:8000/predict'; 

    // --- 檢測按鈕點擊事件監聽器 ---
    // 當檢測按鈕被點擊時，執行非同步函數
    detectButton.addEventListener('click', async () => {
        const message = predictInfoTextarea.value.trim(); // 取得輸入框的內容並去除首尾空白

        // 檢查輸入框是否為空
        if (message.length === 0) {
            alert('請輸入您想檢測的訊息內容。'); // 彈出提示
            return; // 終止函數執行
        }

        // 顯示載入中的狀態，給使用者視覺回饋
        isScamSpan.textContent = '檢測中...';
        isScamSpan.style.color = 'gray';
        confidenceScoreSpan.textContent = '計算中...';
        suspiciousPhrasesDiv.innerHTML = '<p>正在分析訊息，請稍候...</p>';

        try {
            // 使用 fetch API 發送 POST 請求到後端 API
            const response = await fetch(API_URL, {
                method: 'POST', // 指定 HTTP 方法為 POST
                headers: {
                    'Content-Type': 'application/json', // 告訴伺服器發送的資料是 JSON 格式
                },
                // 將訊息文字轉換為 JSON 字串作為請求的主體 (body)
                body: JSON.stringify({ text: message }), 
            });

            // 檢查 HTTP 回應是否成功 (例如：狀態碼 200 OK)
            if (!response.ok) {
                // 如果回應狀態碼不是 2xx (成功)，則拋出錯誤
                const errorData = await response.json(); // 嘗試解析伺服器返回的錯誤訊息
                throw new Error(`伺服器錯誤: ${response.status} ${response.statusText} - ${errorData.detail || '未知錯誤'}`);
            }

            // 解析伺服器回傳的 JSON 資料
            const data = await response.json(); 

            // 根據後端回傳的資料更新前端介面
            updateResults(data.is_scam, data.confidence, data.suspicious_phrases);

        } catch (error) {
            // 捕獲並處理任何在 fetch 過程中發生的錯誤 (例如網路問題、CORS 錯誤)
            console.error('訊息檢測失敗:', error); // 在開發者工具的控制台顯示錯誤
            alert(`訊息檢測失敗，請檢查後端服務是否運行或網路連線。\n錯誤詳情: ${error.message}`); // 彈出錯誤提示
            resetResults(); // 將介面恢復到初始狀態
        }
    });

    // --- 清除按鈕點擊事件監聽器 ---
    // 當清除按鈕被點擊時，執行函數
    clearButton.addEventListener('click', () => {
        predictInfoTextarea.value = ''; // 清空輸入框內容
        resetResults(); // 重置顯示結果
    });

    /**
     * 更新結果顯示的輔助函數
     * @param {boolean} isScam - 是否為詐騙訊息 (從後端獲取)
     * @param {number} confidence - 模型預測可信度 (0-1, 從後端獲取)
     * @param {string[]} suspiciousParts - 可疑詞句陣列 (從後端獲取)
     */
    function updateResults(isScam, confidence, suspiciousParts) {
        // 根據 isScam 的布林值顯示不同的文字和顏色
        if (isScam) {
            isScamSpan.textContent = '是，這極有可能是詐騙訊息！🚨';
            isScamSpan.style.color = '#c0392b'; // 深紅色
        } else {
            isScamSpan.textContent = '否，這看起來不像是詐騙訊息。✅';
            isScamSpan.style.color = '#27ae60'; // 深綠色
        }

        // 顯示模型預測的可信度，轉換為百分比並保留兩位小數
        confidenceScoreSpan.textContent = `${(confidence * 100).toFixed(2)}%`; 

        suspiciousPhrasesDiv.innerHTML = ''; // 清空之前顯示的可疑詞句

        // 如果有可疑詞句，則以列表形式顯示
        if (suspiciousParts && suspiciousParts.length > 0) { 
            const ul = document.createElement('ul'); // 建立一個無序列表元素
            suspiciousParts.forEach(phrase => { // 遍歷每個可疑詞句
                const li = document.createElement('li'); // 建立列表項目元素
                li.textContent = phrase; // 設定列表項目文字
                ul.appendChild(li); // 將列表項目加入無序列表
            });
            suspiciousPhrasesDiv.appendChild(ul); // 將無序列表加入顯示區塊
        } else {
            // 如果沒有可疑詞句，則顯示預設文字
            suspiciousPhrasesDiv.innerHTML = '<p>沒有偵測到特別可疑的詞句。</p>';
        }
    }

    /**
     * 重置所有顯示結果為初始狀態的輔助函數
     */
    function resetResults() {
        isScamSpan.textContent = '待檢測';
        isScamSpan.style.color = 'inherit'; // 恢復預設顏色
        confidenceScoreSpan.textContent = '待檢測';
        suspiciousPhrasesDiv.innerHTML = '<p>請輸入訊息並點擊「檢測！」按鈕。</p>';
    }
});