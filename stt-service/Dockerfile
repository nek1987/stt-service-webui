FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV GNUTLS_CPUID_OVERRIDE=0x1

# Устанавливаем Python, pip и системные пакеты
RUN apt-get update -o Acquire::ForceIPv4=true \
 && apt-get install -y --no-install-recommends \
      python3 python3-pip \
      apt-transport-https ca-certificates \
      ffmpeg git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Устанавливаем конвертер и сам CTranslate2
RUN pip3 install --no-cache-dir \
      transformers[torch]>=4.23 \
      ctranslate2>=3.0 \
      faster-whisper[all] fastapi uvicorn

# 2) Конвертируем HF-модель в CTranslate2-формат (будет лежать в /models)
RUN mkdir /models \
 && ct2-transformers-converter \
      --model islomov/navaistt_v1_medium \
      --output_dir /models/islomov_navaistt_v1_medium_ct2 \
      --copy_files tokenizer.json preprocessor_config.json \
      --quantization float16

# 3) Копируем приложение
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5085"]
