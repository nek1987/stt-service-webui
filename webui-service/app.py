import os
import requests
import gradio as gr
import logging

# 1) Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("webui")

# 2) Read env vars
STT_API = os.getenv("STT_API", "http://stt-service:5085/transcribe")
UI_USER = os.getenv("UI_USER", "")
UI_PASS = os.getenv("UI_PASS", "")
API_TOKEN = os.getenv("API_TOKEN", "")

logger.info(f"STT_API endpoint: {STT_API}")
if not (UI_USER and UI_PASS):
    logger.warning("UI_USER/UI_PASS not set — UI will be unprotected!")
if not API_TOKEN:
    logger.warning("API_TOKEN not set — UI will not send API key!")

def transcribe_audio(file_path):
    logger.info(f"Uploading {file_path}")
    headers = {}
    if API_TOKEN:
        headers["X-API-KEY"] = API_TOKEN
    with open(file_path, "rb") as f:
        response = requests.post(STT_API, files={"file": f}, headers=headers, timeout=120)
    logger.info(f"Response status: {response.status_code}")
    response.raise_for_status()
    text = response.json().get("text", "")
    logger.info(f"Transcribed text: {text[:80]}…")
    return text

# 3) Build Gradio interface with basic auth
iface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.File(
        label="Upload Audio",
        type="filepath",
        file_types=[".wav", ".mp3", ".m4a", ".ogg", ".opus", ".flac", ".amr"]
    ),
    outputs=gr.Textbox(label="Transcription"),
    title="STT Web UI",
    description="Upload an audio file and receive transcription.",
    flagging_mode="never"
)

if __name__ == "__main__":
    launch_kwargs = {"server_name": "0.0.0.0", "server_port": 7860}
    if UI_USER and UI_PASS:
        launch_kwargs["auth"] = (UI_USER, UI_PASS)
        logger.info("UI basic auth enabled")
    iface.launch(**launch_kwargs)
