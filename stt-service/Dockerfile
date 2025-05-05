# 1) Базовый CUDA-runtime (правильный тег)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 2) Отключаем проблемные GnuTLS-оптимизации
ENV GNUTLS_CPUID_OVERRIDE=0x1

# 3) Устанавливаем system-зависимости + Python 3 + pip3
RUN apt-get update -o Acquire::ForceIPv4=true \
 && apt-get install -y --no-install-recommends \
      apt-transport-https ca-certificates \
      python3 python3-pip \
      ffmpeg git \
 && rm -rf /var/lib/apt/lists/*

# 4) Рабочая директория
WORKDIR /app

# 5) Устанавливаем Python-пакеты через pip3
RUN pip3 install --no-cache-dir \
      torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir faster-whisper[all] fastapi uvicorn

# 6) Копируем код
COPY app.py .

# 7) Экспоним порт и запускаем UVicorn
EXPOSE 5085
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5085"]
