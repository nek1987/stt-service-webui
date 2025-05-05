import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import pipeline
import io

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("stt-service")

app = FastAPI()

asr = pipeline(
    "automatic-speech-recognition",
    model="islomov/navaistt_v1_medium",
    device=0,
    chunk_length_s=30,
    stride_length_s=(5, 5),
)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    data = await file.read()
    logger.info(f"Got {len(data)} bytes of audio, filename={file.filename}")
    try:
        result = asr(data)
        text = result.get("text", "")
        logger.info(f"Transcribed text: {text[:100]}...")
        return {"text": text}
    except Exception as e:
        # логируем весь стектрейс
        logger.exception("Transcription failed")
        # возвращаем понятный 500
        raise HTTPException(status_code=500, detail=str(e))
