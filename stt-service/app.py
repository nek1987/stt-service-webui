import os, logging, time, itertools
import io
from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from faster_whisper import WhisperModel

# 1) Logging setup
_counter = itertools.count(1)          # глобальный счётчик

class ReqFilter(logging.Filter):
    def filter(self, record):
        # если handler получил extra={"req": …} ― покажем номер, иначе «–»
        record.req = getattr(record, "req", "-")
        return True

LOG_FMT = "%(asctime)s %(levelname)s [req=%(req)s] %(message)s"
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format=LOG_FMT,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],   # пишем прямиком в stdout
)
logging.getLogger().addFilter(ReqFilter())
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
    req_id = next(_counter)
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
    t0 = time.perf_counter()
    logger.info(
        f"Received audio {len(data)} bytes, fname={file.filename}",
        extra={"req": req_id},
    )
    try:
        segments, _ = model.transcribe(
            io.BytesIO(data),
            beam_size=5,
            best_of=5,
            language="uz"
        )
        text = "".join(seg.text for seg in segments)
        took = time.perf_counter() - t0
        logger.info(
            f"Done ({took:.3f}s): {text[:60]}…",
            extra={"req": req_id},
        )
  
    except Exception as e:
        logger.exception("Error", extra={"req": req_id})
        raise HTTPException(status_code=500, detail=f"Transcription error: {e}")
