# STT Service WebUI

This repository contains two Docker services for deploying an STT (Automatic Speech Recognition) system using the **islomov/navaistt\_v1\_medium** model (Whisper-medium for Uzbek) and a simple Gradio-based web interface.

## Architecture

```text
stt-service/         # FastAPI service using Huggingface Pipeline
  ├── Dockerfile     # Builds image with CUDA, cuDNN, Python3, PyTorch, and Transformers
  └── app.py         # FastAPI application (/healthz, /transcribe)

webui-service/       # Gradio web interface
  ├── Dockerfile     # Builds image with Python3, requests, Gradio dependencies
  └── app.py         # Gradio interface for uploading audio and displaying text

docker-compose.yml   # Stack definition, health checks, webui depends on stt-service
```

---

## Quick Start

1. **Build and run**

   ```bash
   docker-compose down --rmi local  # optional: remove local images
   docker-compose build             # build both services
   docker-compose up -d             # run in detached mode
   ```

2. **Check service health**

   ```bash
   curl http://localhost:5085/healthz
   # {"status":"ok"}
   ```

3. **Web UI**

   Open your browser at `http://localhost:7860`.

   * Upload an audio file (WAV, MP3, OGG/Opus, M4A, FLAC, AMR, etc.)
   * Press **Submit** — your audio will be sent to the STT service
   * View the recognized text

4. **API example**

   ```bash
   curl -X POST "http://localhost:5085/transcribe" \
        -F "file=@/path/to/audio.wav" \
        -H "Accept: application/json"
   # {"text":"...transcribed text..."}
   ```

---

## Configuration and Environment Variables

* `STT_API` (in webui-service): URL of the STT service, e.g., `http://stt-service:5085/transcribe`.
* `NVIDIA_VISIBLE_DEVICES` (in stt-service): GPU device index to use.

---

## Dependencies

* **stt-service**:

  * Python 3, FastAPI, Uvicorn
  * PyTorch + CUDA/cuDNN8
  * Transformers (Huggingface)
  * soundfile, torchaudio, python-multipart

* **webui-service**:

  * Python 3, Gradio, requests

---

## Development and Debugging

* View logs:

  ```bash
  docker-compose logs -f stt-service
  docker-compose logs -f webui-service
  ```
* Check containers and ports:

  ```bash
  docker-compose ps
  ```

---

## Changelog

* **v2**: Switched from vLLM to Huggingface Pipeline (removed CTranslate2 and faster-whisper).
* **v1**: Initial implementation based on vLLM.

---

## Author

**Jamshid Radjabov** - AI / Telecom Expert
