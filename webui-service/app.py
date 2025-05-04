import gradio as gr
from openai import OpenAI
import os

# Адрес вашего запущенного vLLM STT сервиса, считывается из переменных окружения
# В docker-compose.yml мы настроим VLLM_API_BASE=http://stt-service:8000/v1
VLLM_API_BASE = os.environ.get("VLLM_API_BASE", "http://localhost:5085/v1") # Default для локального теста

# API ключ для vLLM (если он настроен на сервере vLLM)
API_KEY = os.environ.get("API_KEY", "sk-any-key")

# Название модели STT, обслуживаемой vLLM
MODEL_NAME = os.environ.get("MODEL_NAME", "islomov/navaistt_v1_medium")

client = OpenAI(base_url=VLLM_API_BASE, api_key=API_KEY)

def transcribe_audio(audio_file):
    """
    Принимает путь к аудиофайлу, отправляет его на транскрипцию в vLLM.
    """
    if audio_file is None:
        return "Пожалуйста, загрузите аудиофайл."

    try:
        with open(audio_file, "rb") as audio:
            print(f"Отправка файла {audio_file} модели {MODEL_NAME} на транскрипцию VLLM по адресу {VLLM_API_BASE}...")
            # Используем клиент OpenAI для отправки запроса транскрипции
            transcript = client.audio.transcriptions.create(
                model=MODEL_NAME,
                file=audio
            )
            return transcript.text

    except Exception as e:
        return f"Ошибка транскрипции: {e}"

# Создаем интерфейс Gradio
# Используем gr.Audio для загрузки аудио, указываем тип "filepath"
# Используем gr.Textbox для отображения результата
iface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(type="filepath", label="Загрузите аудиофайл"),
    outputs=gr.Textbox(label="Транскрибированный текст"),
    title="STT Демо с VLLM",
    description=f"Загрузите аудиофайл для транскрипции моделью {MODEL_NAME} через VLLM."
)

# Запускаем веб-сервер Gradio
if __name__ == "__main__":
    # server_name="0.0.0.0" нужен для доступа из Docker контейнера
    # server_port=7860 порт, который мы пробрасываем в docker-compose
    iface.launch(server_name="0.0.0.0", server_port=7860, share=False)