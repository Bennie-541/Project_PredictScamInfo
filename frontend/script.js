// frontend/script.js
//  當 DOM (文件物件模型) 完全載入後才執行裡面的程式碼
document.addEventListener('DOMContentLoaded', () => {
// 取得 HTML 元素:輸入、按鈕、顯示區塊。document.getElementById('對應的html元素id')
    const inputTextArea = document.getElementById('predict_info'); // 輸入訊息的文字區域
    const inputButton = document.getElementById('detect_button');  // 檢測按鈕
    const clearButton = document.getElementById('clear_button');   // 清除按鈕

    // 取得圖片上傳欄位與圖片按鈕
    const imageInput = document.getElementById('imageInput');
    const imageButton = document.getElementById('image_button');
    
    // 取得顯示結果的 HTML 元素
    const normalOrScam = document.getElementById('is_scam');                    // 顯示正常或詐騙
    const confidenceScoreSpan = document.getElementById('confidence_score');    // 顯示模型預測可信度
    const suspiciousPhrasesDiv = document.getElementById('suspicious_phrases'); // 顯示可疑詞句列表

    /*
    後端 FastAPI API 的 URL
    在開發階段，通常是 http://127.0.0.1:8000 或 http://localhost:8000
    請根據你實際運行 FastAPI 的位址和 Port 進行設定
     */
    const API_URL = "http://127.0.0.1:8000/predict";
    const API_IMAGE_URL = "http://127.0.0.1:8000/predict-image"

    // --- 檢測按鈕點擊事件監聽器 ---
    // 當檢測按鈕被點擊時，執行非同步函數
    //addEventListener('click', async () => {...})
    // click 是一種 DOM 事件，代表使用者點擊按鈕。
    // async 是「非同步函數」的關鍵字，允許你在函數中使用 await。它讓你可以像同步一樣撰寫非同步程式(例如網路請求)。

    inputButton.addEventListener('click', async () => {  
        const message = inputTextArea.value.trim(); //.value取得<textarea>的輸入內容。
                                        //.trim()刪掉文字開頭和結尾的空白，避免誤判空訊息。
        // 檢查輸入框是否為空
        if (message.length === 0) {//alert("輸出內容")是瀏覽器內建的彈出提示。視窗屬於 JavaScript 提供的「視覺提示方法」
            alert('請輸入您想檢測的訊息內容。'); // alert彈出提示
            return; // 終止函數執行
        }

        // 顯示載入中的狀態，給使用者視覺回饋
        normalOrScam.textContent = '檢測中...';       //.textContent->插入純文字
        normalOrScam.style.color = 'gray';           //styly.color改變顏色，style可以想為UI設計
        confidenceScoreSpan.textContent = '計算中...';//.innerHTML->插入 HTML 語法
        suspiciousPhrasesDiv.innerHTML = '<p>正在分析訊息，請稍候...</p>';//<></>大部分html語法長這樣

        try {//try...catch處理程式錯誤。
//fetch(API_URL, {...})。API_URL第17段的變數。method:為html方法，POST送出請求。headers告訴伺服器傳送的資料格式是什麼。
//這段是用 fetch 來呼叫後端 API，送出 POST 請求：
            const response = await fetch(API_URL, {     
                method: 'POST', // 指定 HTTP 方法為 POST
                headers: {      // 告訴伺服器發送的資料是 JSON 格式。選JSON原因:
//我們使用前端(JavaScript)與後端(Python FastAPI)是兩種不同的語言，而JSON是前後端通訊的「共通語言」，交換最通用、最方便、最安全的格式。
                    'Content-Type': 'application/json'
                },
                //把JavaScript物件{text:message}轉換成JSON格式字串，字串作為請求的主體 (body)
                body: JSON.stringify({ text: message}), 
            });

            // 檢查 HTTP 回應是否成功 (例如：狀態碼 200 OK)
            if (!response.ok) {                          // 如果回應狀態碼不是 2xx (成功)，則拋出錯誤
//建立一個錯誤對象並丟出來，強制跳到catch{}區塊。response.status:HTTP狀態碼(如500)。
                throw new Error(`伺服器錯誤: ${response.status} ${response.statusText} `);
            }
            // 變數data，儲存後端成功回傳的資料。
            const data = await response.json(); 
            // 因為後端回傳的欄位名稱status、confidence、suspicious_keywords跟傳進去的參數一致，所以拿的到後端回傳的資料。
            updateResults(              //呼叫function，分別對應function的
                data.status,            //isScam,輸出正常或詐騙
                data.confidence,        //confidence,輸出信心值
                data.suspicious_keywords,//suspiciousParts,可疑關鍵字
                data.highlighted_text
            );
        } catch (error) {// 捕獲並處理任何在 fetch 過程中發生的錯誤 (例如網路問題、CORS 錯誤)
            
            console.error('訊息檢測失敗:', error);// 在開發者工具的控制台顯示錯誤
            alert(`訊息檢測失敗，請檢查後端服務是否運行或網路連線。\n錯誤詳情: ${error.message}`); // 彈出錯誤提示
            resetResults();                      // 將介面恢復到初始狀態
        }
    });

    imageButton.addEventListener('click', async()=>{
        const file = imageInput.files[0]; //取得上傳相片
        if (!file){
            alert("請先選擇圖片");
            return;
        }
        // 顯示載入中提示
        normalOrScam.textContent = '圖片分析中...';
        normalOrScam.style.color = 'gray';
        confidenceScoreSpan.textContent = '計算中...';
        suspiciousPhrasesDiv.innerHTML = '<p>正在從圖片擷取文字與分析中...</p>';

        try{
            const formData = new FormData();
            formData.append("file", file); // 附加圖片檔案給後端
            const response = await fetch(API_IMAGE_URL,{
                    method : "POST",
                    body : formData
            });
            if (!response.ok){
                throw new Error(`圖片分析失敗: ${response.status} ${response.statusText}`);
            }
            const data = await response.json();
            updateResults(
                data.status,
                data.confidence,
                data.suspicious_keywords,
                data.highlighted_text
            )
        }catch(error) {
            console.error("圖片上傳失敗",error);
            alert("圖片分析失敗")
            resetResults();
        }
    });

    function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
/*
function highlightSuspiciousWords(text, suspiciousParts) {
    let highlighted = text;
    suspiciousParts.forEach(word => {
        if (word.length < 2) return; // 避免標記太短詞（如單個字或符號）
        const pattern = new RegExp(escapeRegExp(word), 'g');
        highlighted = highlighted.replace(pattern, `<span class="highlight">${word}</span>`);
    });
    return highlighted;
}
*/
    // --- 清除按鈕點擊事件監聽器 ---
    // 當清除按鈕被點擊時，執行函數
    clearButton.addEventListener('click', () => {
        inputTextArea.value = '';// 清空輸入框內容
        resetResults();          // 重置顯示結果
    });

    // --- 清除按鈕點擊事件監聽器 ---
    // 當清除按鈕被點擊時，執行函數
    clearButton.addEventListener('click', () => {
        inputTextArea.value = '';// 清空輸入框內容
        resetResults();          // 重置顯示結果
    });

    /**這是 JSDoc 註解格式，給開發者與編輯器看的，不會執行。
     * 更新結果顯示的輔助函數
     * @param {string} isScam       - 是否為詐騙訊息 (從後端獲取)(原始要得"@"param {string} isScam )
     * @param {number} confidence   - 模型預測可信度 (從後端獲取)
     * @param {string[]} suspiciousParts - 可疑詞句陣列 (從後端獲取)
     */

    //回傳輸出給index.html顯示
    function updateResults(isScam, confidence, suspiciousParts, highlightedText) {
    normalOrScam.textContent = isScam;
    confidenceScoreSpan.textContent = confidence;

    if (confidence < 15) {
        suspiciousPhrasesDiv.innerHTML = '<p>此訊息為低風險，未發現可疑詞句。</p>';
    } else {
        suspiciousPhrasesDiv.innerHTML = highlightedText;
    }
}

    /**
     * 重置所有顯示結果為初始狀態的輔助函數
     */
    function resetResults() {
        normalOrScam.textContent = '待檢測';
        normalOrScam.style.color = 'inherit'; // 恢復預設顏色
        confidenceScoreSpan.textContent = '待檢測';
        suspiciousPhrasesDiv.innerHTML = '<p>請輸入訊息並點擊「檢測！」按鈕。</p>';
    }
});