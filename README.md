# STT Service WebUI

A Docker-based STT (Automatic Speech Recognition) system using **faster-whisper** with a Gradio web UI.

## Repository Structure

```text
stt-service/         # FastAPI service with faster‑whisper + CTranslate2
  ├── Dockerfile     # Builds CUDA/cuDNN image, installs dependencies, converts model
  └── app.py         # FastAPI app: /healthz (instant), /transcribe (lazy model load + auth)

webui-service/       # Gradio front-end
  ├── Dockerfile     # Builds Slim Python image with Python libs and ffmpeg
  ├── requirements.txt
  └── app.py         # Gradio UI with basic auth, sends X-API-KEY header

docker-compose.yml   # Defines services, ports, and simple depends_on
README.md            # This documentation
```

---

## Features

* **faster‑whisper** with CTranslate2 for \~2× real-time transcription on GPU
* **Lazy loading**: model loads on first request (20–30 s), then stays in memory
* **Token‑based auth**: secure `/transcribe` with `X-API-KEY`
* **UI Basic Auth**: protect Gradio interface with username/password
* **Multi‑format support**: WAV, MP3, OGG/Opus, M4A, FLAC, AMR, etc.
* **Health check**: `/healthz` returns OK immediately

---

## Quick Start

### 1. Clone & backup

```bash
git clone https://github.com/nek1987/stt-service-webui.git
cd stt-service-webui
# (optional) git archive -o backup.tar HEAD
```

### 2. Define environment variables

In your `.env` or directly in `docker-compose.yml`:

```yaml
services:
  stt-service:
    environment:
      - MODEL_PATH=/models/islomov_navaistt_v2_medium_ct2
      - NVIDIA_VISIBLE_DEVICES=0
      - API_TOKEN=your-secret-token

  webui-service:
    environment:
      - STT_API=http://stt-service:5085/transcribe
      - API_TOKEN=your-secret-token
      - UI_USER=admin
      - UI_PASS=s3cret
```

### 3. Build & run

```bash
docker-compose down --rmi local    # optional: remove old images
docker-compose build               # build both services
docker-compose up -d               # start in detached mode
```

### 4. Check services

```bash
# STT service health
curl http://localhost:5085/healthz
# → {"status":"ok"}

# Gradio UI
open http://localhost:7860
# Will prompt for user/pass (UI_USER/UI_PASS)
```

---

## Service Configuration

### stt-service

* **Port:** 5085
* **Endpoints:**

  * `GET  /healthz` → `{"status":"ok"}`
  * `POST /transcribe` (multipart `file@`, header `X-API-KEY`)
* **Auth:** must include `X-API-KEY: your-secret-token`
* **Model path:** baked in `/models/islomov_navaistt_v1_medium_ct2`
* **Lazy load**: model initializes on first `/transcribe`

### webui-service

* **Port:** 7860
* **Auth:** basic HTTP auth (username/UI\_USER, password/UI\_PASS)
* **UI → Service:** sends `X-API-KEY` header automatically

---

## API Reference

### Health Check

```
GET http://<host>:5085/healthz
→ 200 OK {"status":"ok"}
```

### Transcribe Audio

```
POST http://<host>:5085/transcribe
Headers:
  X-API-KEY: your-secret-token
  Accept: application/json
Body:
  multipart/form-data, field "file" = audio file
```

**Response** `200 OK`:

```json
{ "text": "transcribed text here" }
```

**Errors**:

* `401 Unauthorized` if missing/invalid token
* `400 Bad Request` if no file
* `500 Internal Server Error` on model/load failures

---

## Notes & Tips

* Increase GPU concurrency by running multiple instances behind a load balancer.
* To support streaming partial results, integrate `model.transcribe(..., stream=True)`.
* Tune `beam_size` and `compute_type` in `app.py` for quality vs. speed.

---

## Author

**Jamshid Radjabov** — Telecom expert and AI Enthusias .

*Pull requests and issues are welcome!*
