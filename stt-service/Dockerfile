# 1) Базовый образ CUDA (правильный тег)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 2) Отключаем проблемные GnuTLS-оптимизации
ENV GNUTLS_CPUID_OVERRIDE=0x1

# 3) Устанавливаем Python3, pip3 и системные пакеты
RUN apt-get update -o Acquire::ForceIPv4=true \
 && apt-get install -y --no-install-recommends \
      python3 python3-pip \
      apt-transport-https ca-certificates \
      ffmpeg git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 4) Устанавливаем необходимые пакеты
RUN pip3 install --no-cache-dir \
      torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir \
      ctranslate2>=3.0 \
      faster-whisper fastapi uvicorn

# 5) Конвертируем модель в CTranslate2-формат
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

# 6) Копируем сервис
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5085"]
