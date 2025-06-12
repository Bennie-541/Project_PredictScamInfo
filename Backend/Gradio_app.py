import gradio as gr
from bert_explainer import analyze_text, analyze_image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os

# ✅ 初始化 FastAPI
api = FastAPI()

# ✅ 開放 CORS（避免跨網域錯誤）
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ API 路由：健康檢查
@api.get("/health")
def health_check():
    return {"status": "ok"}

# ✅ API 路由：測試文字分析 GET 方法
@api.get("/run/predict_text")
def test_predict_text():
    result = analyze_text("這是測試訊息", explain_mode="cnn")
    return {
        "status": result["status"],
        "confidence": f'{result["confidence"]}%',
        "suspicious_keywords": result["suspicious_keywords"]
    }

# ✅ API 路由：正式 POST 方法
@api.post("/run/predict_text")
def predict_text_api(payload: dict):
    try:
        text, mode = payload["data"]
        result = analyze_text(text=text, explain_mode=mode)
        return {
            "data": [
                result["status"],
                f'{result["confidence"]}%',
                ", ".join(result["suspicious_keywords"])
            ]
        }
    except Exception as e:
        return {"error": str(e)}

# ✅ API 路由：圖片分析（POST），這裡可延伸實作 predict_image 版本
@api.post("/run/predict_image")
async def predict_image_api(file: UploadFile = File(...), explain_mode: str = Form(...)):
    
    try:
        if not file:
            raise ValueError("未上傳圖片")
        if not explain_mode:
            raise ValueError("未指定分析模式")
        
        img_bytes = await file.read()
        if not img_bytes:
            raise ValueError("圖片內容為空")
        print(f"收到圖片: {file.filename}, 模式: {explain_mode}")
        result = analyze_image(img_bytes, explain_mode=explain_mode)

        return {
            "status": result["status"],
            "confidence": f'{result["confidence"]}%',
            "suspicious_keywords": result["suspicious_keywords"]
        }

    except Exception as e:
        return {"error": str(e)}

# ✅ Gradio UI 功能
def predict_text(text, mode):
    result = analyze_text(text=text, explain_mode=mode)
    return result["status"], f"{result['confidence']}%", ", ".join(result["suspicious_keywords"])

def predict_image(file_path, mode):
    with open(file_path, "rb") as f:
        result = analyze_image(f.read(), explain_mode=mode)
    return result["status"], f"{result['confidence']}%", ", ".join(result["suspicious_keywords"])

with gr.Blocks() as demo:
    with gr.Tab("文字模式"):
        text_input = gr.Textbox(lines=3, label="輸入文字")
        text_mode = gr.Radio(["cnn", "bert", "both"], value="cnn", label="分析模式")
        text_btn = gr.Button("提交")
        text_output1 = gr.Textbox(label="判斷結果")
        text_output2 = gr.Textbox(label="置信度")
        text_output3 = gr.Textbox(label="可疑詞彙")
        text_btn.click(fn=predict_text, inputs=[text_input, text_mode], outputs=[text_output1, text_output2, text_output3])

    with gr.Tab("圖片模式"):
        image_input = gr.Image(type="filepath", label="上傳圖片")
        image_mode = gr.Radio(["cnn", "bert", "both"], value="cnn", label="分析模式")
        image_btn = gr.Button("提交")
        image_output1 = gr.Textbox(label="判斷結果")
        image_output2 = gr.Textbox(label="置信度")
        image_output3 = gr.Textbox(label="可疑詞彙")
        image_btn.click(fn=predict_image, inputs=[image_input, image_mode], outputs=[image_output1, image_output2, image_output3])

# ✅ 啟用 Gradio + FastAPI 整合
app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
