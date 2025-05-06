import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from faster_whisper import WhisperModel
import io

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("stt-service")

app = FastAPI(
    title="STT via faster-whisper",
    description="Whisper-medium Uzbek transcription with lazy model loading",
)

# Placeholder for the Whisper model; it will be loaded on first request
model = None

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    global model

    # Lazy-load the model on first transcription request
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

    # Read audio bytes and run transcription
    data = await file.read()
    logger.info(f"Received audio: {len(data)} bytes, filename={file.filename}")
    try:
        segments, _ = model.transcribe(
            io.BytesIO(data),
            beam_size=5,
            best_of=5,
            language="uz"
        )
        text = "".join(seg.text for seg in segments)
        logger.info(f"Transcription successful: {text[:80]}...")
        return {"text": text}
    except Exception as e:
        logger.exception("Error during transcription.")
        raise HTTPException(status_code=500, detail=f"Transcription error: {e}")
