import os
import requests
import gradio as gr
import logging

# Настройка базового логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# URL вашего STT-сервиса, задаётся через переменную окружения STT_API
STT_API = os.environ.get("STT_API", "http://stt-service:8000/transcribe")
logger.info(f"STT_API endpoint set to {STT_API}")


def transcribe_audio(audio_path):
    """
    Получает путь к загруженному аудиофайлу из Gradio,
    отправляет его на STT-сервис и возвращает текст транскрипции.
    """
    logger.info(f"Received audio file at {audio_path}")
    try:
        with open(audio_path, "rb") as f:
            files = {"file": f}
            response = requests.post(STT_API, files=files, timeout=120)
            logger.info(f"Request to STT service returned status {response.status_code}")
            response.raise_for_status()
            result = response.json()
            text = result.get("text", "[No text returned]")
            logger.info(f"Transcription result: {text[:100]}...")
            return text
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return f"Error: {e}"


# Создаём Gradio-интерфейс
iface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(label="Upload Audio", type="filepath"),
    outputs=gr.Textbox(label="Transcription"),
    title="STT Web UI",
    description="Upload an audio file and receive transcription from the STT service.",
    allow_flagging="never"
)

if __name__ == "__main__":
    logger.info("Launching Gradio interface...")
    iface.launch(server_name="0.0.0.0", server_port=7860, share=False)
    logger.info("Gradio interface stopped.")
