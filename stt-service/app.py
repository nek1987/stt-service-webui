import io
from fastapi import FastAPI, File, UploadFile
from faster_whisper import WhisperModel

app = FastAPI(
    title="STT faster-whisper",
    description="Transcribe Uzbek (Whisper-medium) via faster-whisper",
)

# Загружаем модель один раз в GPU (float16)
model = WhisperModel(
    "islomov/navaistt_v1_medium",
    device="cuda",
    compute_type="float16",
)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    data = await file.read()
    segments, _ = model.transcribe(
        io.BytesIO(data),
        beam_size=5,
        best_of=5,
        language="uz"          # если нужно жёстко указать язык
    )
    text = "".join(seg.text for seg in segments)
    return {"text": text}
