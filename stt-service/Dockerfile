FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
ENV GNUTLS_CPUID_OVERRIDE=0x1

# 1) Системные зависимости для аудио
RUN apt-get update -o Acquire::ForceIPv4=true \
 && apt-get install -y --no-install-recommends \
      python3 python3-pip \
      ffmpeg git libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Устанавливаем сначала transformers и ctranslate2, 
#    чтобы конвертер смог импортировать AutoConfig
RUN pip3 install --no-cache-dir \
      transformers>=4.23 \
      ctranslate2>=3.0

# 3) Устанавливаем faster-whisper и FastAPI-стек
RUN pip3 install --no-cache-dir \
      faster-whisper \
      fastapi uvicorn python-multipart

# 4) Конвертируем модель в CTranslate2-формат (force + удаляем старую)
RUN rm -rf /models/islomov_navaistt_v1_medium_ct2 \
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
      --quantization float16 \
      --force

# 5) Копируем ваш сервис и запускаем
COPY app.py .

EXPOSE 5085
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5085"]
