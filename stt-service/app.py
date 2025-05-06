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
    description="Whisper-medium Uzbek transcription using faster-whisper",
)

# Load model once
model = WhisperModel(
    "islomov/navaistt_v1_medium",
    device="cuda",
    compute_type="float16",
)

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    data = await file.read()
    logger.info(f"Received {len(data)} bytes file={file.filename}")
    try:
        segments, _ = model.transcribe(
            io.BytesIO(data),
            beam_size=5,
            best_of=5,
            language="uz"
        )
        text = "".join(seg.text for seg in segments)
        logger.info(f"Transcription success: {text[:80]}…")
        return {"text": text}
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(e))
