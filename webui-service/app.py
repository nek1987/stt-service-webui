import os
import requests
import gradio as gr

# URL вашего STT-сервиса, задаётся через переменную окружения STT_API
STT_API = os.environ.get("STT_API", "http://stt-service:5085/transcribe")


def transcribe_audio(audio_path):
    """
    Получает путь к загруженному аудиофайлу из Gradio,
    отправляет его на STT-сервис и возвращает текст транскрипции.
    """
    try:
        with open(audio_path, "rb") as f:
            files = {"file": f}
            response = requests.post(STT_API, files=files, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("text", "[No text returned]")
    except Exception as e:
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
    iface.launch(server_name="0.0.0.0", server_port=7860, share=False)
