import os
import logging
import time
import io
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from faster_whisper import WhisperModel

# ────────── BASIC LOGGING ──────────
LOG_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format=LOG_FMT,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("stt-service")

# ────────── CONFIG ──────────
API_TOKEN = os.getenv("API_TOKEN", "")
if not API_TOKEN:
    logger.warning("API_TOKEN not set — endpoint will be unprotected!")

MODEL_PATH_DEFAULT = "/models/islomov_navaistt_v2_medium_ct2"

app = FastAPI(title="STT via faster-whisper")
model: Optional[WhisperModel] = None  # lazy singleton

# ────────── ROUTES ──────────
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    api_key: str | None = Header(None, alias="X-API-KEY"),
):
    """Transcribe posted audio file with Whisper."""

    # ───── Auth
    if API_TOKEN and api_key != API_TOKEN:
        logger.warning("Unauthorized access attempt")
        raise HTTPException(status_code=401, detail="Invalid API key")

    global model

    # ───── Lazy model load
    if model is None:
        logger.info("Loading Whisper model…")
        try:
            model_path = os.getenv("MODEL_PATH", MODEL_PATH_DEFAULT)
            model = WhisperModel(model_path, device="cuda", compute_type="float16")
            logger.info("Whisper model loaded.")
        except Exception as exc:
            logger.exception("Failed to load model")
            raise HTTPException(status_code=500, detail=f"Model load error: {exc}")

    # ───── Read audio & transcribe
    data = await file.read()
    start = time.perf_counter()
    logger.info("Received audio %d bytes, fname=%s", len(data), file.filename)

    try:
        segments, _ = model.transcribe(
            io.BytesIO(data),
            beam_size=5,
            best_of=5,
            language="uz",
        )
        text = "".join(seg.text for seg in segments)
        elapsed = time.perf_counter() - start
        logger.info("Transcribed in %.2fs: %s…", elapsed, text[:60])
        return {"text": text}
    except Exception as exc:
        logger.exception("Transcription error")
        raise HTTPException(status_code=500, detail=f"Transcription error: {exc}")
