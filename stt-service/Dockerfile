FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV GNUTLS_CPUID_OVERRIDE=0x1

RUN apt-get update -o Acquire::ForceIPv4=true \
 && apt-get install -y --no-install-recommends \
      python3 python3-pip \
      apt-transport-https ca-certificates \
      ffmpeg git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ——————————————————————————————————————————————
# Установка зависимостей
# 1) PyTorch с CUDA
RUN pip3 install --no-cache-dir \
      torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2) Конвертер и фреймворки
RUN pip3 install --no-cache-dir \
      transformers>=4.23 \
      ctranslate2>=3.0 \
      faster-whisper \
      fastapi uvicorn
# ——————————————————————————————————————————————

# Конвертация модели в CTranslate2-формат
RUN mkdir /models \
 && ct2-transformers-converter \
      --model islomov/navaistt_v1_medium \
      --output_dir /models/islomov_navaistt_v1_medium_ct2 \
      --copy_files \
         vocab.json \
         merges.txt \
         normalizer.json \
         special_tokens_map.json \
         tokenizer_config.json \
         preprocessor_config.json \
      --quantization float16

# Копируем приложение
COPY app.py .

EXPOSE 5085

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5085"]
