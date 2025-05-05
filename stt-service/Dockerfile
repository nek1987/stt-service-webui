# 1) Базовый образ — с правильным тегом
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 2) Отключаем GnuTLS-оптимизации, чтобы метод https не падал SIGILL
ENV GNUTLS_CPUID_OVERRIDE=0x1

# 3) Обновляем пакеты и устанавливаем ffmpeg, git
RUN apt-get update -o Acquire::ForceIPv4=true \
 && apt-get install -y --no-install-recommends \
      apt-transport-https ca-certificates \
      ffmpeg git \
 && rm -rf /var/lib/apt/lists/*

# 4) Устанавливаем Python-зависимости и сам сервис
WORKDIR /app
RUN pip install --no-cache-dir \
      torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir faster-whisper[all] fastapi uvicorn

COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
