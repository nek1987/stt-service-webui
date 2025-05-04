# Используем легковесный базовый образ Python
FROM python:3.10-slim

# Устанавливаем необходимые системные пакеты (для работы с аудио, если Gradio или зависимости требуют)
# Хотя основные зависимости обработки аудио будут в VLLM, некоторые могут быть нужны и UI
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл с требованиями и устанавливаем зависимости
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копируем скрипт приложения
COPY app.py .

# Expose порт, на котором Gradio будет слушать
EXPOSE 7860

# Команда запуска приложения
CMD ["python", "app.py"]