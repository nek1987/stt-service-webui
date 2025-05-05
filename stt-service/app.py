from fastapi import FastAPI, File, UploadFile
from transformers import pipeline
import io

app = FastAPI(
    title="STT via Huggingface pipeline",
    description="Whisper-medium for Uzbek transcription",
)

# Загружаем модель один раз при старте
asr = pipeline(
    "automatic-speech-recognition",
    model="islomov/navaistt_v1_medium",
    device=0,                 # GPU 0
    chunk_length_s=30,        # разбивать длинные аудио на 30s
    stride_length_s=(5, 5),   # наложение контекста
)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    data = await file.read()
    # результат содержит {'text': "...", ...}
    result = asr(data)
    return {"text": result["text"]}
