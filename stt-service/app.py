import os
import logging
import time
import itertools
import io
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from faster_whisper import WhisperModel

# ────────── LOGGING ──────────
_counter = itertools.count(1)  # incremental request ID

class ReqFilter(logging.Filter):
    """Ensure every LogRecord has `req` attr so formatter never fails."""
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "req"):
            record.req = "-"
        return True

LOG_FMT = "%(asctime)s %(levelname)s [req=%(req)s] %(message)s"
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format=LOG_FMT,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger().addFilter(ReqFilter())  # root logger
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
    if API_TOKEN and api_key != API_TOKEN:
        logger.warning("Unauthorized access attempt")
        raise HTTPException(status_code=401, detail="Invalid API key")

    global model
    req_id = next(_counter)
    log = logging.LoggerAdapter(logger, {"req": req_id})

    # Lazy model load
    if model is None:
        log.info("Loading Whisper model…")
        try:
            model_path = os.getenv("MODEL_PATH", MODEL_PATH_DEFAULT)
            model = WhisperModel(model_path, device="cuda", compute_type="float16")
            log.info("Whisper model loaded.")
        except Exception as exc:
            log.exception("Failed to load model")
            raise HTTPException(status_code=500, detail=f"Model load error: {exc}")

    # Read audio
    data = await file.read()
    t0 = time.perf_counter()
    log.info("Received audio %d bytes, fname=%s", len(data), file.filename)

    try:
        segments, _ = model.transcribe(
            io.BytesIO(data),
            beam_size=5,
            best_of=5,
            language="uz",
        )
        text = "".join(seg.text for seg in segments)
        took = time.perf_counter() - t0
        log.info("Done (%.3fs): %s…", took, text[:60])
        return {"text": text}
    except Exception as exc:
        log.exception("Transcription error")
        raise HTTPException(status_code=500, detail=f"Transcription error: {exc}")
