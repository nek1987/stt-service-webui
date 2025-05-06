# Base image with CUDA 12.1 and cuDNN8
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Avoid GnuTLS SIGILL inside apt method https
ENV GNUTLS_CPUID_OVERRIDE=0x1

# Install system dependencies
RUN apt-get update -o Acquire::ForceIPv4=true \
 && apt-get install -y --no-install-recommends \
      python3 python3-pip \
      ffmpeg git libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Install PyTorch & torchvision from PyTorch index
RUN pip3 install --no-cache-dir \
      torch torchvision \
      --extra-index-url https://download.pytorch.org/whl/cu121

# 2) Install faster-whisper and FastAPI stack
RUN pip3 install --no-cache-dir \
      faster-whisper \
      ctranslate2 \
      fastapi uvicorn python-multipart soundfile torchaudio

# Convert the Huggingface Whisper-medium model into CTranslate2 format
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


# Copy application code
COPY app.py .

EXPOSE 5085
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5085"]
