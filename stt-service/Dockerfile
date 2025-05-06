# 1) Base image with CUDA & cuDNN
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 2) Workaround for apt’s https method crash
ENV GNUTLS_CPUID_OVERRIDE=0x1

# 3) Install system dependencies
RUN apt-get update -o Acquire::ForceIPv4=true \
 && apt-get install -y --no-install-recommends \
      python3 python3-pip \
      ffmpeg git libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 4) Install transformers & CTranslate2 first (so the converter can import them)
RUN pip3 install --no-cache-dir \
      transformers>=4.23 \
      ctranslate2>=3.0

# 5) Install faster-whisper and FastAPI stack
RUN pip3 install --no-cache-dir \
      faster-whisper \
      fastapi uvicorn python-multipart soundfile torchaudio

# 6) Convert the HF Whisper-medium model into CTranslate2 format
#    (remove any previous conversion, force overwrite)
RUN rm -rf /models/islomov_navaistt_v1_medium_ct2 \
 && ct2-transformers-converter \
      --model islomov/navaistt_v1_medium \
      --output_dir /models/islomov_navaistt_v1_medium_ct2 \
      --quantization float16 \
      --force

# 7) Copy the application code
COPY app.py .

# 8) Expose and launch
EXPOSE 5085
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5085"]
