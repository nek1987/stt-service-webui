import os
import logging
import time
import itertools
import io

from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from faster_whisper import WhisperModel

# ────────── LOGGING ──────────
_counter = itertools.count(1)  # сквозной номер запроса

# Фабрика, чтобы у каждого LogRecord было поле req
_old_factory = logging.getLogRecordFactory()
def _record_factory(*args, **kwargs):
    record = _old_factory(*args, **kwargs)
    if not hasattr(record, "req"):
        record.req = "-"          # дефолт
    return record
logging.setLogRecordFactory(_record_factory)

LOG_FMT = "%(asctime)s %(levelname)s [req=%(req)s] %(message)s"
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

app = FastAPI(
    title="STT via faster-whisper",
    description="Uzbek Whisper-medium transcription with token-based auth",
)

model: WhisperModel | None = None        # ленивый singleton

# ────────── ROUTES ──────────
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    api_key: str = Header(None, alias="X-API-KEY"),
):
    # Проверяем токен
    if API_TOKEN and api_key != API_TOKEN:
        logger.warning("Unauthorized access attempt", extra={"req": "-"})
        raise HTTPException(status_code=401, detail="Invalid API key")

    global model
    req_id = next(_counter)

    # Ленивая инициализация модели
    if model is None:
        logger.info("Loading Whisper model…")
        try:
            model_path = os.getenv("MODEL_PATH", MODEL_PATH_DEFAULT)
            model = WhisperModel(
                model_path,
                device="cuda",
                compute_type="float16",
            )
            logger.info("Whisper model loaded.")
        except Exception as e:
            logger.exception("Failed to load Whisper model.", extra={"req": req_id})
            raise HTTPException(status_code=500, detail=f"Model load error: {e}")

    # Читаем аудио
    data = await file.read()
    t0 = time.perf_counter()
    logger.info(
        "Received audio %d bytes, fname=%s",
        len(data),
        file.filename,
        extra={"req": req_id},
    )

    # Распознаём
    try:
        segments, _ = model.transcribe(
            io.BytesIO(data),
            beam_size=5,
            best_of=5,
            language="uz",
        )
        text = "".join(seg.text for seg in segments)
        took = time.perf_counter() - t0
        logger.info("Done (%.3fs): %s…", took, text[:60], extra={"req": req_id})
        return {"text": text}
    except Exception as e:
        logger.exception("Transcription error.", extra={"req": req_id})
        raise HTTPException(status_code=500, detail=f"Transcription error: {e}")
```
