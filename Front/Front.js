// 綁定 HTML 中的 DOM 元素，方便後續操作
const submit_Btn = document.getElementById("submitBtn")
const reset_Btn = document.getElementByid("resetBtn");
const inputElement = document.getElementByid("inputText");
const resultDiv = document.getElementById("result");   

//送出資料按鈕事件
submit_Btn.addEventListener("click", async ()=>{
    const userInput = inputElement.value.trim();

    if (!userInput){ 
        // 如果沒有輸入內容，顯示提醒
        resultDiv.innerText = "請先輸入訊息再送出。";
        return;
    }

    try{
        const response = await fetch("https://127.0.0.1:8000/predict", {
            // 使用 POST 方法傳送資料
            method: "POST",
            headers:{
                "Content-Type":"application/json" // 告訴後端傳送的是 JSON 格式
            },
            body: JSON.stringify({message:userInput })// 把輸入訊息轉成 JSON 格式送出
        });
        // 解析後端回傳的 JSON 結果
        const data = await response.json();

        // 將預測結果顯示在畫面上
        resultDiv.innerText = 
        `判斷結果：${data.label}（信心值：${(data.confidence * 100).toFixed(1)}%）`;
    }catch (error){
         // 如果發送請求時出錯，顯示錯誤訊息
        resultDiv.innerText="⚠️ 發送請求時發生錯誤，請確認後端是否啟動。";
        console.error("錯誤細節:",error); // 顯示錯誤資訊在開發者主控台
    };
});

//  綁定「清除內容」按鈕的點擊事件處理函數
reset_Btn.addEventListener("click",()=>{
  inputElement.value = "";     // 清空使用者輸入的文字
  resultDiv.innerText = "";    // 清空結果顯示區
})