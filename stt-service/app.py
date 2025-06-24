import os
import logging
import io
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from faster_whisper import WhisperModel

# 1) Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("stt-service")

# 2) Read API token from env
API_TOKEN = os.getenv("API_TOKEN", "")
if not API_TOKEN:
    logger.warning("API_TOKEN not set — endpoint will be unprotected!")

app = FastAPI(
    title="STT via faster-whisper",
    description="Whisper-medium Uzbek transcription with token-based auth",
)

# 3) Lazy model placeholder
model = None  # type: WhisperModel | None

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    api_key: str = Header(None, alias="X-API-KEY"),
):
    # 4) Enforce token auth
    if API_TOKEN and api_key != API_TOKEN:
        logger.warning(f"Unauthorized access attempt with key={api_key}")
        raise HTTPException(status_code=401, detail="Invalid API key")

    global model
    # 5) Lazy-load the model
    if model is None:
        logger.info("Loading Whisper model for the first time…")
        try:
            MODEL_PATH = os.getenv(
                "MODEL_PATH",
                "/models/islomov_navaistt_v2_medium_ct2",   # дефолт на новую модель
            )
            model = WhisperModel(
                MODEL_PATH,
                device="cuda",
                compute_type="float16",
            )
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.exception("Failed to load Whisper model.")
            raise HTTPException(status_code=500, detail=f"Model load error: {e}")

    # 6) Read and transcribe
    data = await file.read()
    logger.info(f"Received audio {len(data)} bytes, fname={file.filename}")
    try:
        segments, _ = model.transcribe(
            io.BytesIO(data),
            beam_size=5,
            best_of=5,
            language="uz"
        )
        text = "".join(seg.text for seg in segments)
        logger.info(f"Transcription successful: {text[:80]}…")
        return {"text": text}
    except Exception as e:
        logger.exception("Error during transcription.")
        raise HTTPException(status_code=500, detail=f"Transcription error: {e}")
