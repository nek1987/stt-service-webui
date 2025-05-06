import os
import requests
import gradio as gr
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

STT_API = os.environ.get("STT_API", "http://stt-service:5085/transcribe")
logger.info(f"STT_API endpoint set to {STT_API}")

def transcribe_audio(audio_path):
    logger.info(f"Received audio file at {audio_path}")
    try:
        with open(audio_path, "rb") as f:
            files = {"file": f}
            response = requests.post(STT_API, files=files, timeout=120)
            logger.info(f"Request returned status {response.status_code}")
            response.raise_for_status()
            text = response.json().get("text", "[No text returned]")
            logger.info(f"Result: {text[:80]}…")
            return text
    except Exception as e:
        logger.exception("Error during transcription")
        return f"Error: {e}"

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
    logger.info("Launching Gradio interface...")
    iface.launch(server_name="0.0.0.0", server_port=7860, share=False)
