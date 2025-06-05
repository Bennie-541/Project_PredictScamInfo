import gradio as gr
from bert_explainer import analyze_text

def predict_fn(text, mode):
    result = analyze_text(text=text, explain_mode=mode)
    status = result['status']
    confidence = f"{result['confidence']}%"
    keywords = ', '.join(result['suspicious_keywords'])
    return status, confidence, keywords

iface =gr.Interface(
    fn=predict_fn,
    inputs = [
        gr.TextArea(label="輸入訊息"),
        gr.Radio(choices=["cnn", "bert", "both"], label="分析模式", value="cnn")
    ],
    outputs = [
        gr.Textbox(label = "判斷結果"),
        gr.Textbox(label = "可疑分數"),
        gr.Textbox(label = "可疑詞彙")
    ],
    title= "預判詐騙訊息",
    description="輸入訊息，AI 將判定是否為詐騙並標記可疑詞  "
)
iface.launch()