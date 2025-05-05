# 1) базовый образ с CUDA
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# 2) системные зависимости
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# 3) окружение Python + faster-whisper
WORKDIR /app
RUN pip install --no-cache-dir \
      torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir faster-whisper[all] fastapi uvicorn

# 4) копируем приложение
COPY app.py .

# 5) открываем порт и запускаем
EXPOSE 5085
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5085"]
