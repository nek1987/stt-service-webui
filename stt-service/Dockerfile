FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
ENV GNUTLS_CPUID_OVERRIDE=0x1

RUN apt-get update -o Acquire::ForceIPv4=true \
 && apt-get install -y --no-install-recommends \
      python3 python3-pip \
      ffmpeg git libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Ставим torch и torchvision из PyTorch-индекса
RUN pip3 install --no-cache-dir \
      torch torchvision \
      --extra-index-url https://download.pytorch.org/whl/cu121

# 2) Ставим transformers и остальные из PyPI
RUN pip3 install --no-cache-dir \
      transformers \
      fastapi \
      uvicorn \
      python-multipart

# Устанавливаем PyTorch, transformers и аудиозависимости
RUN pip3 install --no-cache-dir \
      torch torchvision \
      --extra-index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir \
      transformers \
      fastapi uvicorn python-multipart \
      soundfile torchaudio


COPY app.py .

EXPOSE 5085

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5085"]
