# Базовый образ с CUDA 12.1 и cuDNN 8
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV GNUTLS_CPUID_OVERRIDE=0x1

# Системные утилиты + Python3
RUN apt-get update -o Acquire::ForceIPv4=true \
 && apt-get install -y --no-install-recommends \
      python3 python3-pip \
      apt-transport-https ca-certificates \
      ffmpeg git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Устанавливаем PyTorch + Transformers + FastAPI
RUN pip3 install --no-cache-dir \
      torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
      transformers fastapi uvicorn python-multipart

# Копируем код
COPY app.py .

EXPOSE 5085

# Запускаем FastAPI сервер
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5085"]
