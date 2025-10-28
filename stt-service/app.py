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

# 2) Read API tokens from env
def _load_api_tokens() -> set[str]:
    tokens_env = os.getenv("API_TOKENS")
    if tokens_env:
        tokens = {token.strip() for token in tokens_env.split(",") if token.strip()}
        if not tokens:
            logger.warning("API_TOKENS provided but no valid entries found")
        return tokens

    single_token = os.getenv("API_TOKEN", "").strip()
    return {single_token} if single_token else set()


API_TOKENS = _load_api_tokens()
if not API_TOKENS:
    logger.warning("No API tokens configured — endpoint will be unprotected!")
else:
    logger.info("Configured %d API token(s)", len(API_TOKENS))

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
    if API_TOKENS and (not api_key or api_key not in API_TOKENS):
        logger.warning("Unauthorized access attempt")
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
