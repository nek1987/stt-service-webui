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

# 2) Read API tokens from env with validation
def _load_api_tokens() -> set[str]:
    """Load and validate API tokens from environment variables."""
    MIN_TOKEN_LENGTH = 8
    tokens_raw = []

    # Try API_TOKENS first (comma-separated list)
    tokens_env = os.getenv("API_TOKENS")
    if tokens_env:
        tokens_raw = [token.strip() for token in tokens_env.split(",") if token.strip()]

    # Fallback to single API_TOKEN if API_TOKENS is empty
    if not tokens_raw:
        single_token = os.getenv("API_TOKEN", "").strip()
        if single_token:
            tokens_raw = [single_token]

    # Validate tokens
    valid_tokens = set()
    for token in tokens_raw:
        if len(token) < MIN_TOKEN_LENGTH:
            logger.warning(
                "Ignoring token with insufficient length (%d chars, min %d required): %s***",
                len(token), MIN_TOKEN_LENGTH, token[:3] if len(token) >= 3 else "***"
            )
            continue
        valid_tokens.add(token)

    # Check for duplicates
    if len(tokens_raw) != len(valid_tokens):
        logger.info("Removed %d duplicate token(s)", len(tokens_raw) - len(valid_tokens))

    return valid_tokens


def _mask_token(token: str) -> str:
    """Mask API token for safe logging (show first 6 and last 3 chars)."""
    if len(token) <= 9:
        return "***" + token[-3:] if len(token) >= 3 else "***"
    return token[:6] + "***" + token[-3:]


API_TOKENS = _load_api_tokens()
if not API_TOKENS:
    logger.warning("No API tokens configured — endpoint will be unprotected!")
else:
    logger.info("Configured %d valid API token(s)", len(API_TOKENS))
    for token in sorted(API_TOKENS):  # sorted for consistent logging
        logger.info("  - Token: %s", _mask_token(token))

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
    if API_TOKENS:
        if not api_key:
            logger.warning("Unauthorized access attempt: No API key provided")
            raise HTTPException(status_code=401, detail="Invalid API key")

        if api_key not in API_TOKENS:
            logger.warning(
                "Unauthorized access attempt: Invalid key %s",
                _mask_token(api_key)
            )
            raise HTTPException(status_code=401, detail="Invalid API key")

        # Log successful authentication
        logger.info("Authenticated with key: %s", _mask_token(api_key))

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
