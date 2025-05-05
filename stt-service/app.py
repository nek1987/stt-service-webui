import io
from fastapi import FastAPI, File, UploadFile
from faster_whisper import WhisperModel

app = FastAPI(title="STT faster-whisper + CTranslate2")

# Загружаем уже сконвертированную модель
model = WhisperModel(
    "/models/islomov_navaistt_v1_medium_ct2",
    device="cuda",
    compute_type="float16",
)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    data = await file.read()
    segments, _ = model.transcribe(io.BytesIO(data), beam_size=5)
    text = "".join(seg.text for seg in segments)
    return {"text": text}
