import os
import gradio as gr
import base64
import tempfile
import requests
from dotenv import load_dotenv

# Load API Key
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"

# Fungsi OCR
def ocr_image(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode("utf-8")
    data_url = f"data:image/{image_path.split('.')[-1]};base64,{b64}"

    payload = {
        "model": "opengvlab/internvl3-14b:free",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "OCR this image. Extract all text only."},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Fungsi untuk Gradio
def process_image(img):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name)
        return ocr_image(tmp.name)

# Gradio UI
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Textbox(label="Extracted Text"),
    title="OCR with InternVL3 via OpenRouter",
    description="Upload an image and extract the text using OpenRouter LLM."
)

if __name__ == "__main__":
    iface.launch()
