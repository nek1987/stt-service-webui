import logging
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from faster_whisper import WhisperModel

# 1) Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("stt-service")

# 2) Create FastAPI app
app = FastAPI(
    title="STT via faster-whisper",
    description="Whisper-medium Uzbek transcription with lazy model loading",
)

# 3) Placeholder for the model; will load on first request
model: WhisperModel | None = None

@app.get("/healthz")
async def healthz():
    """
    Health check endpoint.
    Returns 200 OK as soon as the server is up (model not yet loaded).
    """
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Transcribe an audio file. On the first call, loads the Whisper model,
    which takes ~20–30s. Subsequent calls are fast.
    """
    global model

    # Lazy-load the model on demand
    if model is None:
        logger.info("Loading Whisper model for the first time...")
        try:
            model = WhisperModel(
                "/models/islomov_navaistt_v1_medium_ct2",
                device="cuda",
                compute_type="float16",
            )
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.exception("Failed to load Whisper model.")
            raise HTTPException(status_code=500, detail=f"Model load error: {e}")

    # Read audio bytes
    data = await file.read()
    logger.info(f"Received audio: {len(data)} bytes, filename={file.filename}")

    # Perform transcription
    try:
        segments, _ = model.transcribe(
            io.BytesIO(data),
            beam_size=5,
            best_of=5,
            language="uz"
        )
        text = "".join([seg.text for seg in segments])
        logger.info(f"Transcription successful: {text[:80]}...")
        return {"text": text}
    except Exception as e:
        logger.exception("Error during transcription.")
        raise HTTPException(status_code=500, detail=f"Transcription error: {e}")
