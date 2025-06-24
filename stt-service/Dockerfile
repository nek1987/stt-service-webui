# 1) Base image with CUDA 12.1 + cuDNN 8
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# 2) Workaround for apt https SIGILL
ENV GNUTLS_CPUID_OVERRIDE=0x1

# 3) System deps (audio, Python, pip)
RUN apt-get update -o Acquire::ForceIPv4=true \
 && apt-get install -y --no-install-recommends \
      python3 python3-pip \
      ffmpeg git libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 4) Install transformers & the correct CTranslate2 for CUDA12/cuDNN8
RUN pip3 install --no-cache-dir \
      transformers>=4.23 \
      ctranslate2==4.4.0

# 5) Install faster-whisper and FastAPI stack
RUN pip3 install --no-cache-dir \
      faster-whisper \
      fastapi uvicorn python-multipart soundfile torchaudio

# 6) Convert the Huggingface Whisper-medium model to CTranslate2 format
#    Force-overwrite any existing directory
RUN rm -rf /models/islomov_navaistt_v2_medium_ct2 \
 && ct2-transformers-converter \
      --model islomov/navaistt_v2_medium \
      --output_dir /models/islomov_navaistt_v2_medium_ct2 \
      --quantization float16 \
      --force

# 7) Copy the application code
COPY app.py .

# 8) Expose port & launch Uvicorn
EXPOSE 5085
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5085"]
