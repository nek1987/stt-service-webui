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

# Инициализация клиента OpenAI
# max_retries=0 может помочь, если не хотите, чтобы клиент пытался повторно подключиться
# в случае временных сбоев VLLM (например, при разогреве)
client = OpenAI(base_url=VLLM_API_BASE, api_key=API_KEY, max_retries=0)

def transcribe_audio(audio_file):
    """
    Принимает путь к аудиофайлу, отправляет его на транскрипцию в vLLM.
    """
    if audio_file is None:
        return "Пожалуйста, загрузите аудиофайл."

    # Проверка существования файла и размера (опционально)
    if not os.path.exists(audio_file):
        return f"Ошибка: Файл не найден по пути {audio_file}"
    # Добавить проверку размера файла, если есть ограничения API

    try:
        with open(audio_file, "rb") as audio:
            print(f"Отправка файла {os.path.basename(audio_file)} модели {MODEL_NAME} на транскрипцию VLLM по адресу {VLLM_API_BASE}...")
            # Используем клиент OpenAI для отправки запроса транскрипции
            # Убедитесь, что формат аудиофайла поддерживается VLLM/Whisper
            # VLLM принимает файл как part of a multipart/form-data request,
            # также как OpenAI API
            transcript = client.audio.transcriptions.create(
                model=MODEL_NAME, # Указываем имя модели
                file=audio,
                # Можно добавить другие параметры Whisper API, например:
                # language="uz", # Указать язык
                # response_format="text", # Формат ответа: text, json, srt, vtt, verbose_json (json по умолчанию)
                # temperature=0.0, # Температура для семплирования
            )

            # Ответ может быть текстом или JSON, в зависимости от response_format
            if isinstance(transcript, str):
                 return transcript
            elif hasattr(transcript, 'text'):
                 return transcript.text
            else:
                 return str(transcript) # На всякий случай, если формат ответа неожиданный

    except Exception as e:
        print(f"Произошла ошибка при транскрипции: {e}")
        # В случае ошибки подключения, vLLM может вернуть специфичные ошибки
        # Например, openai.APIStatusError, openai.APIConnectionError
        return f"Ошибка транскрипции: {e}"

# Создаем интерфейс Gradio
iface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(type="filepath", label="Загрузите аудиофайл (.wav, .mp3, etc.)"), # Укажите ожидаемые форматы
    outputs=gr.Textbox(label="Транскрибированный текст"),
    title="STT Демо с VLLM",
    description=f"Загрузите аудиофайл для транскрипции моделью {MODEL_NAME} через VLLM."
)

# Запускаем веб-сервер Gradio
if __name__ == "__main__":
    # server_name="0.0.0.0" нужен для доступа из Docker контейнера
    # server_port=7860 порт, который мы пробрасываем в docker-compose
    # share=False для локального развертывания (не шарим в интернет через Gradio сервис)
    iface.launch(server_name="0.0.0.0", server_port=7860, share=False)
