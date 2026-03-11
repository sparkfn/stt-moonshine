> **ARCHIVED:** This implementation plan describes the original pipeline with VAD, LID, and denoise components. These were stripped in commit `34150b8` (2026-03-09). See `2026-03-08-strip-vad-lid-denoise.md` for the refactor plan and the current README for up-to-date architecture.

# Moonshine ASR Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a drop-in replacement for qwen3-asr using a Moonshine-based voice pipeline (noisereduce -> Silero VAD -> Silero LID -> Moonshine EN/ZH).

**Architecture:** Single-process FastAPI server. All models (VAD, LID, Moonshine EN, Moonshine ZH) loaded at startup. Priority inference queue for WebSocket vs HTTP. Identical API surface to qwen3-asr.

**Tech Stack:** FastAPI, uvicorn, Moonshine (moonshine-onnx or moonshine), Silero VAD, Silero LID, noisereduce, loguru, PyTorch, numpy, soundfile

---

### Task 1: Project scaffold — logger, errors, config

**Files:**
- Create: `src/logger.py`
- Create: `src/errors.py`
- Create: `src/config.py`

**Step 1: Create `src/logger.py`**

```python
from __future__ import annotations
import contextvars
import json
import logging
import os
import sys
from loguru import logger

_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


def set_request_id(req_id: str) -> contextvars.Token:
    return _request_id_var.set(req_id)


def reset_request_id(token: contextvars.Token) -> None:
    _request_id_var.reset(token)


def get_request_id() -> str | None:
    return _request_id_var.get()


_LEVEL_MAP = {
    "critical": "fatal",
    "warning": "warn",
}


def _json_sink(message: "loguru.Message") -> None:
    record = message.record
    raw_level = record["level"].name.lower()
    entry: dict = {
        "timestamp": record["time"].isoformat(),
        "level": _LEVEL_MAP.get(raw_level, raw_level),
        "message": record["message"],
        "service": "moonshine-asr",
    }
    req_id = _request_id_var.get()
    if req_id:
        entry["requestId"] = req_id
    extra = record.get("extra", {})
    if extra:
        entry.update({k: v for k, v in extra.items()})
    if record["exception"]:
        entry["err"] = str(record["exception"].value)
    sys.stdout.write(json.dumps(entry) + "\n")
    sys.stdout.flush()


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            if frame.f_back:
                frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


_STDLIB_LEVEL_MAP = {"TRACE": "DEBUG"}


def setup_logger() -> "loguru.Logger":
    log_level = os.getenv("LOG_LEVEL", "info").upper()
    logging.root.handlers = [InterceptHandler()]
    stdlib_level = _STDLIB_LEVEL_MAP.get(log_level, log_level)
    logging.root.setLevel(stdlib_level)
    logger.remove()
    logger.add(_json_sink, level=log_level, format="{message}", catch=False)
    for name in logging.root.manager.loggerDict.keys():
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True
    for name in ["uvicorn.access", "uvicorn.error", "uvicorn"]:
        logging_logger = logging.getLogger(name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False
    return logger


log = setup_logger()
```

**Step 2: Create `src/errors.py`**

```python
from fastapi.responses import JSONResponse
from logger import get_request_id


def error_response(code: str, message: str, status_code: int, **context) -> JSONResponse:
    req_id = get_request_id()
    ctx = dict(context) if context else {}
    if req_id:
        ctx["requestId"] = req_id
    body = {
        "code": code,
        "message": message,
        "statusCode": status_code,
    }
    if ctx:
        body["context"] = ctx
    return JSONResponse(status_code=status_code, content=body)
```

**Step 3: Create `src/config.py`**

```python
"""Startup configuration validation."""
import os
import sys
from logger import log

def _safe_float(name: str, default: str) -> float:
    raw = os.getenv(name, default)
    try:
        return float(raw)
    except ValueError:
        log.error("Config error: {} must be a float, got '{}' -- using default {}", name, raw, default)
        return float(default)

def _safe_int(name: str, default: str) -> int:
    raw = os.getenv(name, default)
    try:
        return int(raw)
    except ValueError:
        log.error("Config error: {} must be an integer, got '{}' -- using default {}", name, raw, default)
        return int(default)

SSE_CHUNK_SECONDS = _safe_int("SSE_CHUNK_SECONDS", "5")
SSE_OVERLAP_SECONDS = _safe_int("SSE_OVERLAP_SECONDS", "1")

_VALID_LOG_LEVELS = {"TRACE", "DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL", "FATAL"}
_LOG_LEVEL_ALIASES = {"WARN": "WARNING", "FATAL": "CRITICAL"}


def validate_env() -> None:
    errors = []

    # MOONSHINE_EN_MODEL
    en_model = os.getenv("MOONSHINE_EN_MODEL", "")
    if not en_model:
        errors.append("MOONSHINE_EN_MODEL is required but empty or unset")

    # MOONSHINE_ZH_MODEL
    zh_model = os.getenv("MOONSHINE_ZH_MODEL", "")
    if not zh_model:
        errors.append("MOONSHINE_ZH_MODEL is required but empty or unset")

    # REQUEST_TIMEOUT
    try:
        rt = int(os.getenv("REQUEST_TIMEOUT", "300"))
        if rt <= 0:
            errors.append(f"REQUEST_TIMEOUT must be positive, got {rt}")
    except ValueError as e:
        errors.append(f"REQUEST_TIMEOUT must be an integer: {e}")

    # IDLE_TIMEOUT
    try:
        it = int(os.getenv("IDLE_TIMEOUT", "0"))
        if it < 0:
            errors.append(f"IDLE_TIMEOUT must be non-negative, got {it}")
    except ValueError as e:
        errors.append(f"IDLE_TIMEOUT must be an integer: {e}")

    # LOG_LEVEL
    log_level = os.getenv("LOG_LEVEL", "info").upper()
    log_level = _LOG_LEVEL_ALIASES.get(log_level, log_level)
    if log_level not in _VALID_LOG_LEVELS:
        errors.append(f"LOG_LEVEL must be one of {_VALID_LOG_LEVELS}, got '{log_level}'")

    # STT_DEVICE
    stt_device = os.getenv("STT_DEVICE", "auto").lower()
    if stt_device not in ("auto", "cpu", "cuda"):
        errors.append(f"STT_DEVICE must be auto/cpu/cuda, got '{stt_device}'")

    # WS_WINDOW_MAX_S
    try:
        ws = float(os.getenv("WS_WINDOW_MAX_S", "6.0"))
        if ws <= 0:
            errors.append(f"WS_WINDOW_MAX_S must be positive, got {ws}")
    except ValueError as e:
        errors.append(f"WS_WINDOW_MAX_S must be a float: {e}")

    if errors:
        for err in errors:
            log.error("Config validation failed: {}", err)
        sys.exit(1)

    log.info("Config validation passed")
```

**Step 4: Verify files exist**

Run: `ls -la /volume3/docker/moonshine-asr/src/`
Expected: logger.py, errors.py, config.py

---

### Task 2: Schemas — identical to qwen3-asr

**Files:**
- Create: `src/schemas.py`

**Step 1: Create `src/schemas.py`**

```python
from pydantic import BaseModel, Field
from typing import Optional


class ErrorResponse(BaseModel):
    code: str = Field(..., description="Machine-readable error identifier (e.g. AUDIO_DECODE_FAILED)")
    message: str = Field(..., description="Human-readable error description")
    context: Optional[dict] = Field(None, description="Debug data (requestId, input params)")
    statusCode: int = Field(..., description="HTTP status code")

    model_config = {"json_schema_extra": {"examples": [{"code": "AUDIO_DECODE_FAILED", "message": "Could not decode audio: unknown format", "context": {"fileSize": 1024}, "statusCode": 422}]}}


class HealthResponse(BaseModel):
    status: str = Field(..., description="Status of the service", examples=["ok"])
    mode: Optional[str] = Field(None, description="Running mode: 'server' (standalone)", examples=["server"])
    model_loaded: bool = Field(..., description="Whether the ASR models are currently loaded")
    model_id: Optional[str] = Field(None, description="Model identifier", examples=["moonshine-tiny-en + moonshine-tiny-zh"])
    cuda: Optional[bool] = Field(None, description="Whether CUDA is available")
    gpu_name: Optional[str] = Field(None, description="Name of the GPU", examples=["NVIDIA GeForce RTX 4060"])
    gpu_allocated_mb: Optional[int] = Field(None, description="GPU memory currently allocated (MB)")
    gpu_reserved_mb: Optional[int] = Field(None, description="GPU memory currently reserved by PyTorch (MB)")
    worker_alive: Optional[bool] = Field(None, description="Not used (compatibility with qwen3-asr)")


class TranscriptionResponse(BaseModel):
    text: str = Field(..., description="The transcribed text")
    language: str = Field(..., description="The detected or requested language code", examples=["en"])

    model_config = {"json_schema_extra": {"examples": [{"text": "Hello, how are you today?", "language": "en"}]}}


class TranslationResponse(BaseModel):
    text: str = Field(..., description="The translated text")
    language: str = Field(..., description="The target language code used", examples=["en"])

    model_config = {"json_schema_extra": {"examples": [{"text": "Hello, how are you?", "language": "en"}]}}


class SSEChunkEvent(BaseModel):
    text: str = Field(..., description="Transcribed text for this chunk")
    chunk_index: int = Field(..., description="Zero-based index of this chunk")
    is_final: bool = Field(..., description="Whether this is the last chunk")
    language: str = Field(..., description="Detected language code")

    model_config = {"json_schema_extra": {"examples": [{"text": "This is the first part", "chunk_index": 0, "is_final": False, "language": "en"}]}}


class WebSocketHandshake(BaseModel):
    status: str = Field(..., description="Connection status", examples=["connected"])
    buffer_size: int = Field(..., description="Audio buffer size in bytes before inference triggers")
    window_max_s: float = Field(..., description="Maximum sliding window duration in seconds")
    use_server_vad: bool = Field(..., description="Whether server-side VAD is enabled for this connection")
    sample_rate: int = Field(..., description="Expected input sample rate in Hz", examples=[16000])

    model_config = {"json_schema_extra": {"examples": [{"status": "connected", "buffer_size": 14400, "window_max_s": 6.0, "use_server_vad": True, "sample_rate": 16000}]}}


class WebSocketPartial(BaseModel):
    partial: str = Field(..., description="Cumulative transcript of the current sliding window")
    language: str = Field(..., description="Detected language code")

    model_config = {"json_schema_extra": {"examples": [{"partial": "Hello how are you", "language": "en"}]}}


API_TAGS = [
    {
        "name": "Transcription",
        "description": "Speech-to-text transcription endpoints. Upload audio files (WAV, FLAC, MP3, OGG) to get text back.",
    },
    {
        "name": "Translation",
        "description": "Transcribe and translate audio (not implemented -- returns 501).",
    },
    {
        "name": "Subtitles",
        "description": "Generate SRT subtitle files from audio (not implemented -- returns 501).",
    },
    {
        "name": "Streaming",
        "description": "Real-time and SSE streaming transcription for low-latency use cases.",
    },
    {
        "name": "System",
        "description": "Health checks, model status, and server diagnostics.",
    },
]

API_DESCRIPTION = """\
Speech-to-text API powered by Moonshine with Silero VAD/LID pipeline.

## Features
- **OpenAI-compatible** `/v1/audio/transcriptions` endpoint
- **Dual-language**: English and Chinese via language-routed Moonshine models
- **Real-time WebSocket** streaming with sliding window and VAD
- **SSE streaming** for chunked transcription of long files
- **Noise suppression** via noisereduce
- **Drop-in replacement** for qwen3-asr API

## Audio Formats
Supported: WAV, FLAC, MP3, OGG, AIFF, CAF, AU, W64, RF64.

## WebSocket Protocol
Connect to `/ws/transcribe` and send raw PCM audio (16-bit LE, 16kHz mono).
"""
```

---

### Task 3: Pipeline — models, denoise, VAD, LID, STT

**Files:**
- Create: `src/pipeline.py`
- Create: `src/models.py`

**Step 1: Create `src/models.py`**

This handles loading all models with per-component device control.

```python
"""Model loading and management for the Moonshine ASR pipeline."""
from __future__ import annotations
import os
import gc
import time
import numpy as np
from logger import log

# Model instances
_moonshine_en = None
_moonshine_zh = None
_vad_model = None
_lid_model = None
_models_loaded = False
_last_used = 0.0

TARGET_SR = 16000


def _resolve_device(env_var: str, default: str = "cpu") -> str:
    device = os.getenv(env_var, default).lower()
    if device == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_all_models():
    """Load all pipeline models at startup."""
    global _moonshine_en, _moonshine_zh, _vad_model, _lid_model, _models_loaded, _last_used

    if _models_loaded:
        return

    stt_device = _resolve_device("STT_DEVICE", "auto")
    vad_device = _resolve_device("VAD_DEVICE", "cpu")
    lid_device = _resolve_device("LID_DEVICE", "cpu")

    # Load Silero VAD
    log.info("Loading Silero VAD on {}...", vad_device)
    try:
        import torch
        _vad_model = torch.jit.load(
            torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False, trust_repo=True),
        ) if False else None
        # Use silero-vad package directly
        from silero_vad import load_silero_vad
        _vad_model = load_silero_vad()
        _vad_model.eval()
        log.info("Silero VAD loaded")
    except Exception as e:
        log.error("Failed to load Silero VAD: {}", e)

    # Load Silero LID
    log.info("Loading Silero LID on {}...", lid_device)
    try:
        import torch
        _lid_model = torch.jit.load(
            os.getenv("LID_MODEL_PATH", ""),
        ) if os.getenv("LID_MODEL_PATH") else None
        if _lid_model is None:
            # Download from torch hub
            _lid_model, _lid_utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_lang_detector',
                force_reload=False,
                trust_repo=True,
            )
        log.info("Silero LID loaded")
    except Exception as e:
        log.error("Failed to load Silero LID: {}", e)
        _lid_model = None

    # Load Moonshine EN
    en_model_id = os.getenv("MOONSHINE_EN_MODEL", "")
    log.info("Loading Moonshine EN ({}) on {}...", en_model_id, stt_device)
    try:
        from moonshine import load_model
        _moonshine_en = load_model(en_model_id, device=stt_device)
        log.info("Moonshine EN loaded")
    except Exception as e:
        log.error("Failed to load Moonshine EN: {}", e)
        raise

    # Load Moonshine ZH
    zh_model_id = os.getenv("MOONSHINE_ZH_MODEL", "")
    log.info("Loading Moonshine ZH ({}) on {}...", zh_model_id, stt_device)
    try:
        from moonshine import load_model
        _moonshine_zh = load_model(zh_model_id, device=stt_device)
        log.info("Moonshine ZH loaded")
    except Exception as e:
        log.error("Failed to load Moonshine ZH: {}", e)
        raise

    # Warmup
    if stt_device == "cuda":
        log.info("Warming up GPU...")
        import torch
        rng = np.random.default_rng(seed=42)
        dummy = rng.standard_normal(TARGET_SR).astype(np.float32) * 0.01
        try:
            transcribe_en(dummy, TARGET_SR)
            transcribe_zh(dummy, TARGET_SR)
        except Exception:
            pass
        release_gpu_memory()

    _models_loaded = True
    _last_used = time.time()
    log.info("All models loaded!")

    import torch
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        log.info("GPU memory: Allocated={:.0f}MB Reserved={:.0f}MB", allocated, reserved)


def unload_all_models():
    """Unload all models to free memory."""
    global _moonshine_en, _moonshine_zh, _vad_model, _lid_model, _models_loaded
    log.info("Unloading all models (idle timeout)...")
    _moonshine_en = None
    _moonshine_zh = None
    _vad_model = None
    _lid_model = None
    _models_loaded = False
    release_gpu_memory()


def release_gpu_memory():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def is_loaded() -> bool:
    return _models_loaded


def get_last_used() -> float:
    global _last_used
    return _last_used


def touch():
    global _last_used
    _last_used = time.time()


def is_speech(audio_float32: np.ndarray, threshold: float = 0.5) -> bool:
    if _vad_model is None:
        return True
    try:
        import torch
        tensor = torch.from_numpy(audio_float32).unsqueeze(0)
        with torch.no_grad():
            confidence = _vad_model(tensor, TARGET_SR).item()
        return confidence >= threshold
    except Exception:
        return True


def detect_language(audio_float32: np.ndarray) -> str:
    """Detect language using Silero LID. Returns 'en' or 'zh'."""
    if _lid_model is None:
        return "en"
    try:
        import torch
        tensor = torch.from_numpy(audio_float32)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            output = _lid_model(tensor)
        # Silero LID returns language probabilities
        # Map to our two supported languages
        if hasattr(output, 'item'):
            # Simple binary output
            return "zh" if output.item() > 0.5 else "en"
        elif isinstance(output, dict):
            zh_score = output.get("zh", 0) + output.get("cmn", 0)
            en_score = output.get("en", 0)
            return "zh" if zh_score > en_score else "en"
        else:
            return "en"
    except Exception as e:
        log.debug("LID failed, defaulting to en: {}", e)
        return "en"


def transcribe_en(audio: np.ndarray, sr: int) -> str:
    """Transcribe audio using Moonshine English model."""
    from moonshine import transcribe
    result = transcribe(audio, sr, model=_moonshine_en)
    return result if isinstance(result, str) else result[0] if result else ""


def transcribe_zh(audio: np.ndarray, sr: int) -> str:
    """Transcribe audio using Moonshine Chinese model."""
    from moonshine import transcribe
    result = transcribe(audio, sr, model=_moonshine_zh)
    return result if isinstance(result, str) else result[0] if result else ""
```

**Step 2: Create `src/pipeline.py`**

```python
"""Voice pipeline: denoise -> VAD -> LID -> route -> STT."""
from __future__ import annotations
import os
import numpy as np
from logger import log
import models

TARGET_SR = 16000
DENOISE_ENABLED = os.getenv("DENOISE_ENABLED", "true").lower() == "true"


def denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply noise reduction if enabled."""
    if not DENOISE_ENABLED:
        return audio
    try:
        import noisereduce as nr
        return nr.reduce_noise(y=audio, sr=sr, stationary=True)
    except Exception as e:
        log.debug("Denoise failed, using raw audio: {}", e)
        return audio


def process_audio(audio: np.ndarray, sr: int, lang_code: str | None = None) -> tuple[str, str]:
    """
    Full pipeline: denoise -> transcribe with language routing.

    Args:
        audio: float32 audio array
        sr: sample rate
        lang_code: language override (None = auto-detect)

    Returns:
        (transcribed_text, detected_language)
    """
    # Step 1: Denoise
    audio = denoise(audio, sr)

    # Step 2: Determine language
    if lang_code is None or lang_code == "auto":
        detected_lang = models.detect_language(audio)
    else:
        detected_lang = lang_code

    # Step 3: Route to correct model
    if detected_lang == "zh":
        text = models.transcribe_zh(audio, sr)
    else:
        text = models.transcribe_en(audio, sr)
        detected_lang = "en"

    return text, detected_lang


def process_audio_vad(audio: np.ndarray, sr: int, lang_code: str | None = None) -> tuple[str, str]:
    """
    Full pipeline with VAD segmentation: denoise -> VAD -> chunk -> LID -> route -> STT.

    Used for file uploads where we want to process speech segments only.
    """
    from silero_vad import get_speech_timestamps, read_audio
    import torch

    # Step 1: Denoise
    audio = denoise(audio, sr)

    # Step 2: VAD segmentation
    tensor = torch.from_numpy(audio)
    try:
        timestamps = get_speech_timestamps(tensor, models._vad_model, sampling_rate=sr)
    except Exception:
        # Fallback: process entire audio
        return process_audio(audio, sr, lang_code)

    if not timestamps:
        return "", lang_code or "en"

    # Step 3: Process each segment
    texts = []
    detected_lang = lang_code
    for ts in timestamps:
        start = ts['start']
        end = ts['end']
        chunk = audio[start:end]

        if len(chunk) < sr * 0.1:  # skip chunks < 100ms
            continue

        if detected_lang is None or detected_lang == "auto":
            detected_lang = models.detect_language(chunk)

        if detected_lang == "zh":
            text = models.transcribe_zh(chunk, sr)
        else:
            text = models.transcribe_en(chunk, sr)

        if text:
            texts.append(text)

    return " ".join(texts), detected_lang or "en"
```

---

### Task 4: Server — all endpoints matching qwen3-asr

**Files:**
- Create: `src/server.py`

**Step 1: Create `src/server.py`**

```python
from __future__ import annotations
from logger import log, set_request_id, reset_request_id
from errors import error_response
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
import sys
import io
import os
import uuid as _uuid_module
import gc
import json
import re
import asyncio
import concurrent.futures
import time
import heapq
import dataclasses
import numpy as np
from scipy.signal import butter, sosfilt

import models
import pipeline


def _telephony_bandpass(audio: np.ndarray, sr: int) -> np.ndarray:
    sos = butter(4, [300, 3400], btype="bandpass", fs=sr, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def _resample_pcm_bytes(pcm_bytes: bytes, orig_sr: int) -> bytes:
    if orig_sr == TARGET_SR:
        return pcm_bytes
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    import librosa
    resampled = librosa.resample(samples, orig_sr=orig_sr, target_sr=TARGET_SR)
    return resampled.astype(np.int16).tobytes()


TARGET_SR = 16000

_infer_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="moonshine-infer",
)


@dataclasses.dataclass(order=True)
class _InferJob:
    priority: int
    submit_time: float
    future: asyncio.Future = dataclasses.field(compare=False)
    fn: object = dataclasses.field(compare=False)


class PriorityInferQueue:
    def __init__(self):
        self._heap: list[_InferJob] = []
        self._lock = asyncio.Lock()
        self._has_work = asyncio.Event()
        self._worker_task: asyncio.Task | None = None

    def start(self):
        self._worker_task = asyncio.create_task(self._worker())

    def stop(self):
        if self._worker_task:
            self._worker_task.cancel()

    async def _worker(self):
        loop = asyncio.get_event_loop()
        while True:
            await self._has_work.wait()
            async with self._lock:
                if not self._heap:
                    self._has_work.clear()
                    continue
                job = heapq.heappop(self._heap)
                if not self._heap:
                    self._has_work.clear()
            try:
                result = await loop.run_in_executor(_infer_executor, job.fn)
                job.future.set_result(result)
            except Exception as e:
                job.future.set_exception(e)

    async def submit(self, fn, priority: int = 1):
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        job = _InferJob(priority=priority, submit_time=time.time(), future=future, fn=fn)
        async with self._lock:
            heapq.heappush(self._heap, job)
            self._has_work.set()
        return await future


_infer_queue = PriorityInferQueue()
_model_lock = asyncio.Lock()

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "0"))

WS_BUFFER_SIZE = int(os.getenv("WS_BUFFER_SIZE", str(int(TARGET_SR * 2 * 0.45))))
WS_FLUSH_SILENCE_MS = int(os.getenv("WS_FLUSH_SILENCE_MS", "600"))
WS_WINDOW_MAX_S = float(os.getenv("WS_WINDOW_MAX_S", "6.0"))
WS_WINDOW_MAX_BYTES = int(WS_WINDOW_MAX_S * TARGET_SR * 2)
ASR_USE_SERVER_VAD = os.getenv("ASR_USE_SERVER_VAD", "true").lower() == "true"


def detect_and_fix_repetitions(text: str, max_repeats: int = 2) -> str:
    if not text or len(text) < 10:
        return text
    text = re.sub(r'\b(\w+)( \1){2,}\b', r'\1', text)
    words = text.split()
    for phrase_len in range(3, min(9, len(words) // 3 + 1)):
        i = 0
        result = []
        while i < len(words):
            phrase = words[i:i + phrase_len]
            count = 1
            j = i + phrase_len
            while j + phrase_len <= len(words) and words[j:j + phrase_len] == phrase:
                count += 1
                j += phrase_len
            result.extend(phrase)
            if count > max_repeats:
                i = j
            else:
                i += phrase_len
            words = result
    return ' '.join(words)


def _decode_audio(audio_bytes: bytes) -> tuple:
    import soundfile as sf
    return sf.read(io.BytesIO(audio_bytes))


async def _ensure_model_loaded():
    if models.is_loaded():
        models.touch()
        return
    async with _model_lock:
        if models.is_loaded():
            models.touch()
            return
        await asyncio.get_event_loop().run_in_executor(_infer_executor, models.load_all_models)
        models.touch()


async def _idle_watchdog():
    while True:
        await asyncio.sleep(30)
        if IDLE_TIMEOUT <= 0 or not models.is_loaded():
            continue
        if time.time() - models.get_last_used() > IDLE_TIMEOUT:
            async with _model_lock:
                if models.is_loaded() and time.time() - models.get_last_used() > IDLE_TIMEOUT:
                    await asyncio.get_event_loop().run_in_executor(_infer_executor, models.unload_all_models)


@asynccontextmanager
async def lifespan(the_app):
    from config import validate_env
    validate_env()
    _infer_queue.start()
    asyncio.create_task(_idle_watchdog())
    yield
    _infer_executor.shutdown(wait=False)


from schemas import (
    HealthResponse, TranscriptionResponse, TranslationResponse,
    ErrorResponse, API_TAGS, API_DESCRIPTION,
)

app = FastAPI(
    title="Moonshine-ASR",
    version="0.1.0",
    description=API_DESCRIPTION,
    openapi_tags=API_TAGS,
    lifespan=lifespan,
    responses={
        422: {"model": ErrorResponse, "description": "Audio decode or validation error"},
        504: {"model": ErrorResponse, "description": "Inference timed out"},
    },
)


@app.middleware("http")
async def _request_id_middleware(request: Request, call_next):
    req_id = request.headers.get("x-request-id") or str(_uuid_module.uuid4())
    token = set_request_id(req_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response
    finally:
        reset_request_id(token)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
    description="Returns service status, model loading state, and GPU info.",
)
async def health():
    gpu_info = {}
    torch = sys.modules.get("torch")
    cuda_available = torch.cuda.is_available() if torch is not None else False
    if cuda_available and models.is_loaded():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2),
            "gpu_reserved_mb": round(torch.cuda.memory_reserved() / 1024**2),
        }
    return {
        "status": "ok",
        "mode": "server",
        "model_loaded": models.is_loaded(),
        "model_id": f"{os.getenv('MOONSHINE_EN_MODEL', '')} + {os.getenv('MOONSHINE_ZH_MODEL', '')}" if models.is_loaded() else None,
        "cuda": cuda_available,
        **gpu_info,
    }


def _do_transcribe(audio, sr, lang_code) -> tuple[str, str]:
    """Run pipeline transcription. Returns (text, language)."""
    return pipeline.process_audio(audio, sr, lang_code)


@app.post(
    "/v1/audio/transcriptions",
    response_model=TranscriptionResponse,
    tags=["Transcription"],
    summary="Transcribe audio file",
    description="Upload an audio file and get the transcribed text back. Supports WAV, FLAC, MP3, OGG, and other formats. Language is auto-detected by default.",
)
async def transcribe(
    file: UploadFile = File(..., description="Audio file (WAV, FLAC, MP3, OGG, AIFF, etc.)"),
    language: str = Form("auto", description="Language code (e.g. 'en', 'zh'). Use 'auto' for detection."),
    return_timestamps: bool = Form(False, description="Include word-level timestamps in the output text"),
):
    await _ensure_model_loaded()
    audio_bytes = await file.read()
    log.info("POST /v1/audio/transcriptions | file={} size={} language={}", file.filename, len(audio_bytes), language)
    t0 = time.time()
    try:
        audio, sr = _decode_audio(audio_bytes)
    except Exception as e:
        log.error("POST /v1/audio/transcriptions | audio decode failed: {}", e)
        return error_response("AUDIO_DECODE_FAILED", f"Could not decode audio: {e}", 422, fileSize=len(audio_bytes))

    lang_code = None if language == "auto" else language

    try:
        text, detected_lang = await asyncio.wait_for(
            _infer_queue.submit(
                lambda: _do_transcribe(audio, sr, lang_code),
                priority=1,
            ),
            timeout=REQUEST_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return error_response("TRANSCRIPTION_TIMEOUT", "Transcription timed out", 504, elapsed=round(time.time() - t0, 2))

    text = detect_and_fix_repetitions(text)
    log.info("POST /v1/audio/transcriptions | completed in {:.2f}s text_len={} lang={}", time.time() - t0, len(text), detected_lang)
    return {"text": text, "language": detected_lang}


@app.post(
    "/v1/audio/translations",
    tags=["Translation"],
    summary="Translate audio file",
    description="Not implemented. Returns 501.",
)
async def translate_endpoint(
    file: UploadFile = File(..., description="Audio file to transcribe and translate"),
    language: str = Form("en", description="Target language: 'en' (English) or 'zh' (Chinese)"),
    response_format: str = Form("json", description="Response format: 'json' (text) or 'srt' (subtitles)"),
):
    return error_response("NOT_IMPLEMENTED", "Translation is not available in moonshine-asr", 501)


@app.post(
    "/v1/audio/subtitles",
    tags=["Subtitles"],
    summary="Generate SRT subtitles",
    description="Not implemented. Returns 501.",
)
async def generate_subtitles(
    file: UploadFile = File(..., description="Audio file to generate subtitles from"),
    language: str = Form("auto", description="Language code (e.g. 'en', 'zh'). Use 'auto' for detection."),
    mode: str = Form("accurate", description="Subtitle mode: 'fast' (heuristic) or 'accurate' (ForcedAligner)"),
    max_line_chars: int = Form(42, description="Maximum characters per subtitle line before wrapping"),
):
    return error_response("NOT_IMPLEMENTED", "Subtitles are not available in moonshine-asr", 501)


async def sse_transcribe_generator(audio, sr, lang_code):
    audio_duration = len(audio) / sr
    t0 = time.time()
    chunk_count = 0
    log.info("SSE stream | audio={:.2f}s lang={}", audio_duration, lang_code or "auto")
    try:
        from config import SSE_CHUNK_SECONDS, SSE_OVERLAP_SECONDS
        CHUNK_SAMPLES = TARGET_SR * SSE_CHUNK_SECONDS
        OVERLAP_SAMPLES = TARGET_SR * SSE_OVERLAP_SECONDS

        if len(audio) <= CHUNK_SAMPLES:
            text, detected_lang = await _infer_queue.submit(
                lambda: _do_transcribe(audio, sr, lang_code),
                priority=1,
            )
            text = detect_and_fix_repetitions(text)
            data = {"text": text, "language": detected_lang, "is_final": True, "chunk_index": 0}
            chunk_count += 1
            yield f"data: {json.dumps(data)}\n\n"
        else:
            start = 0
            chunk_index = 0
            while start < len(audio):
                end = min(start + CHUNK_SAMPLES, len(audio))
                chunk = audio[start:end]
                is_last = (end >= len(audio))

                text, detected_lang = await _infer_queue.submit(
                    lambda c=chunk: _do_transcribe(c, sr, lang_code),
                    priority=1,
                )
                text = detect_and_fix_repetitions(text)
                data = {
                    "text": text,
                    "language": detected_lang,
                    "is_final": is_last,
                    "chunk_index": chunk_index,
                }
                chunk_count += 1
                yield f"data: {json.dumps(data)}\n\n"
                chunk_index += 1

                if is_last:
                    break
                start = end - OVERLAP_SAMPLES

        log.info("SSE stream | done chunks={} elapsed={:.2f}s", chunk_count, time.time() - t0)
        yield f"data: {json.dumps({'done': True})}\n\n"

    except Exception as e:
        log.error("SSE stream | error after {:.2f}s: {}", time.time() - t0, e)
        yield f"data: {json.dumps({'code': 'SSE_STREAM_ERROR', 'message': str(e), 'statusCode': 500})}\n\n"


@app.post(
    "/v1/audio/transcriptions/stream",
    tags=["Streaming"],
    summary="Stream transcription (SSE)",
    description="Upload a long audio file and receive transcription results as Server-Sent Events.",
    responses={200: {"content": {"text/event-stream": {}}, "description": "SSE stream of transcription chunks"}},
)
async def transcribe_stream(
    file: UploadFile = File(..., description="Audio file to transcribe in streaming mode"),
    language: str = Form("auto", description="Language code (e.g. 'en', 'zh'). Use 'auto' for detection."),
    return_timestamps: bool = Form(False, description="Include word-level timestamps in each chunk"),
):
    await _ensure_model_loaded()
    audio_bytes = await file.read()
    log.info("POST /v1/audio/transcriptions/stream | file={} size={} language={}", file.filename, len(audio_bytes), language)
    try:
        audio, sr = _decode_audio(audio_bytes)
    except Exception as e:
        return error_response("AUDIO_DECODE_FAILED", f"Could not decode audio: {e}", 422, fileSize=len(audio_bytes))

    lang_code = None if language == "auto" else language
    return StreamingResponse(
        sse_transcribe_generator(audio, sr, lang_code),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


async def _transcribe_with_context(
    audio_bytes: bytes | bytearray,
    overlap: bytes | bytearray,
    pad_silence: bool = False,
    lang_code: str | None = None,
    use_vad: bool = True,
) -> str:
    """Transcribe PCM audio with optional silence padding. Returns transcribed text."""
    full_audio = bytearray()
    if overlap:
        full_audio.extend(overlap)
    full_audio.extend(audio_bytes)

    if pad_silence:
        silence_bytes = int((WS_FLUSH_SILENCE_MS / 1000) * TARGET_SR * 2)
        full_audio.extend(bytes(silence_bytes))

    if len(full_audio) == 0:
        return ""

    audio = np.frombuffer(full_audio, dtype=np.int16)
    audio = audio.astype(np.float32) / 32768.0
    audio = _telephony_bandpass(audio, TARGET_SR)
    sr = TARGET_SR

    if use_vad and not models.is_speech(audio):
        return ""

    try:
        text, lang = await asyncio.wait_for(
            _infer_queue.submit(
                lambda: _do_transcribe(audio, sr, lang_code),
                priority=0,
            ),
            timeout=REQUEST_TIMEOUT,
        )
        return detect_and_fix_repetitions(text)
    except asyncio.TimeoutError:
        return "[timeout]"
    except Exception as e:
        log.error("_transcribe_with_context error: {}", e)
        return f"[error: {e}]"


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()

    ws_req_id = websocket.query_params.get("request_id") or str(_uuid_module.uuid4())
    token = set_request_id(ws_req_id)
    log.info("[WS] Client connected")

    audio_buffer = bytearray()
    audio_window = bytearray()
    lang_code: str | None = None
    use_vad: bool = ASR_USE_SERVER_VAD
    vad_param = websocket.query_params.get("use_server_vad")
    if vad_param is not None:
        use_vad = vad_param.lower() in ("true", "1", "yes")
    client_sr = int(websocket.query_params.get("sample_rate", str(TARGET_SR)))
    if client_sr not in (8000, 16000):
        await websocket.send_json({
            "code": "UNSUPPORTED_SAMPLE_RATE",
            "message": f"sample_rate must be 8000 or 16000, got {client_sr}",
            "statusCode": 400,
        })
        await websocket.close()
        return

    chunk_count = 0
    _ws_prev_had_speech: bool = False

    try:
        await _ensure_model_loaded()

        await websocket.send_json({
            "status": "connected",
            "sample_rate": client_sr,
            "format": "pcm_s16le",
            "buffer_size": WS_BUFFER_SIZE,
            "window_max_s": WS_WINDOW_MAX_S,
            "use_server_vad": use_vad,
        })

        while True:
            try:
                data = await websocket.receive()

                if data.get("type") == "websocket.disconnect":
                    break

                if "text" in data:
                    try:
                        msg = json.loads(data["text"])
                        action = msg.get("action", "")

                        if action == "flush":
                            if audio_buffer:
                                audio_window.extend(audio_buffer)
                                audio_buffer.clear()

                            if len(audio_window) > 0:
                                text = await _transcribe_with_context(
                                    bytes(audio_window), b"", pad_silence=True,
                                    lang_code=lang_code, use_vad=use_vad,
                                )
                                chunk_count += 1
                                await websocket.send_json({
                                    "text": text,
                                    "is_partial": False,
                                    "is_final": True,
                                })
                            else:
                                await websocket.send_json({
                                    "text": "",
                                    "is_partial": False,
                                    "is_final": True,
                                })
                            audio_window.clear()

                        elif action == "reset":
                            audio_buffer.clear()
                            audio_window.clear()
                            await websocket.send_json({"status": "buffer_reset"})

                        elif action == "config":
                            new_lang = msg.get("language")
                            if new_lang == "auto":
                                lang_code = None
                            elif new_lang:
                                lang_code = new_lang
                            if "use_server_vad" in msg:
                                use_vad = bool(msg["use_server_vad"])
                            await websocket.send_json({
                                "status": "configured",
                                "language": lang_code or "auto",
                                "use_server_vad": use_vad,
                            })

                        else:
                            await websocket.send_json({
                                "code": "UNKNOWN_ACTION",
                                "message": f"Unknown action: {action!r}",
                                "statusCode": 400,
                            })

                    except json.JSONDecodeError:
                        await websocket.send_json({
                            "code": "INVALID_JSON",
                            "message": "Invalid JSON command",
                            "statusCode": 400,
                        })

                elif "bytes" in data:
                    incoming = data["bytes"]
                    if client_sr != TARGET_SR:
                        incoming = _resample_pcm_bytes(incoming, client_sr)
                    audio_buffer.extend(incoming)

                    if len(audio_buffer) >= WS_BUFFER_SIZE:
                        audio_window.extend(audio_buffer)
                        audio_buffer.clear()

                        if len(audio_window) > WS_WINDOW_MAX_BYTES:
                            trim = len(audio_window) - WS_WINDOW_MAX_BYTES
                            trim = (trim // 2) * 2
                            audio_window = audio_window[trim:]

                        _vad_flushed = False
                        if use_vad:
                            _tail_bytes = bytes(audio_window[-WS_BUFFER_SIZE:]) if len(audio_window) >= WS_BUFFER_SIZE else bytes(audio_window)
                            _tail_float = np.frombuffer(_tail_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                            _current_has_speech = models.is_speech(_tail_float)

                            if not _current_has_speech and _ws_prev_had_speech:
                                _ws_prev_had_speech = False
                                _vad_flushed = True
                                text = await _transcribe_with_context(
                                    bytes(audio_window), b"", pad_silence=True,
                                    lang_code=lang_code, use_vad=use_vad,
                                )
                                chunk_count += 1
                                if text:
                                    await websocket.send_json({
                                        "text": text,
                                        "is_partial": False,
                                        "is_final": True,
                                    })
                                audio_window.clear()
                            else:
                                _ws_prev_had_speech = _current_has_speech

                        if not _vad_flushed:
                            text = await _transcribe_with_context(
                                bytes(audio_window), b"", pad_silence=False,
                                lang_code=lang_code, use_vad=use_vad,
                            )
                            chunk_count += 1
                            if text:
                                await websocket.send_json({
                                    "text": text,
                                    "is_partial": True,
                                    "is_final": False,
                                })

            except WebSocketDisconnect:
                if audio_buffer:
                    audio_window.extend(audio_buffer)
                if len(audio_window) > 0:
                    try:
                        text = await _transcribe_with_context(
                            bytes(audio_window), b"", pad_silence=True,
                            lang_code=lang_code, use_vad=use_vad,
                        )
                        chunk_count += 1
                        if text:
                            log.info("[WS] Final transcription on disconnect: {}", text)
                    except Exception:
                        pass
                log.info("[WS] Client disconnected | chunks_processed={}", chunk_count)
                break

    except Exception as e:
        log.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"code": "WEBSOCKET_ERROR", "message": str(e), "statusCode": 500})
        except Exception:
            pass
    finally:
        reset_request_id(token)
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

### Task 5: Docker — Dockerfile, compose.yaml, .env.example, requirements.txt

**Files:**
- Create: `Dockerfile`
- Create: `compose.yaml`
- Create: `.env.example`
- Create: `requirements.txt`

**Step 1: Create `requirements.txt`**

```
fastapi==0.133.1
uvicorn==0.41.0
python-multipart==0.0.22
websockets==16.0
soundfile==0.13.1
numpy
scipy
noisereduce
silero-vad==6.2.1
loguru==0.7.3
librosa
torch
torchaudio
moonshine
```

**Step 2: Create `Dockerfile`**

```dockerfile
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY src/*.py /app/

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--ws", "websockets"]
```

**Step 3: Create `compose.yaml`**

```yaml
services:
  moonshine-asr:
    build:
      context: .
    image: moonshine-asr-local:cuda124
    container_name: moonshine-asr
    restart: unless-stopped
    ports:
      - "${PORT:-8200}:8000"
    volumes:
      - ./models:/root/.cache/huggingface
    env_file:
      - .env
    networks:
      - moonshine-asr-net
      - freepbx-net
    logging:
      driver: json-file
      options:
        max-size: "20m"
        max-file: "3"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [ gpu ]
    healthcheck:
      test: [ "CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" ]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s

networks:
  moonshine-asr-net:
    driver: bridge
    name: moonshine-asr-net
  freepbx-net:
    external: true
    name: freepbx-net
```

**Step 4: Create `.env.example`**

```bash
# ==========================================
# Moonshine ASR Configuration
# ==========================================
# Copy to .env and adjust as needed.

# -----------------
# Host / Network
# -----------------
PORT=8200

# -----------------
# Models
# -----------------
MOONSHINE_EN_MODEL=moonshine/tiny
MOONSHINE_ZH_MODEL=moonshine/tiny-zh

# -----------------
# Devices (auto / cuda / cpu)
# -----------------
STT_DEVICE=auto
VAD_DEVICE=cpu
LID_DEVICE=cpu
DENOISE_DEVICE=cpu

# -----------------
# Pipeline
# -----------------
DENOISE_ENABLED=true
REQUEST_TIMEOUT=300
IDLE_TIMEOUT=0

# -----------------
# Logging
# -----------------
LOG_LEVEL=info

# -----------------
# WebSocket Streaming
# -----------------
# WS_BUFFER_SIZE=14400
# WS_FLUSH_SILENCE_MS=600
WS_WINDOW_MAX_S=6.0
ASR_USE_SERVER_VAD=true

# -----------------
# SSE Streaming
# -----------------
# SSE_CHUNK_SECONDS=5
# SSE_OVERLAP_SECONDS=1

# -----------------
# Docker / CUDA Internals
# -----------------
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

### Task 6: Verify and adjust model integration

**Note:** This task requires verifying the actual Moonshine Python API. The `models.py` file uses placeholder imports (`from moonshine import load_model, transcribe`). After creating all files, we need to:

1. Check the actual Moonshine package API
2. Check the actual Silero LID API
3. Adjust `models.py` imports and calls to match

**Step 1: Research Moonshine API**

Run: `pip install moonshine && python -c "import moonshine; help(moonshine)"`

**Step 2: Research Silero LID API**

Run: `python -c "import torch; model = torch.hub.load('snakers4/silero-vad', 'silero_lang_detector'); print(type(model))"`

**Step 3: Update `models.py` based on actual APIs**

Adjust load/transcribe calls to match the real package interfaces.

**Step 4: Smoke test**

Run: `cd /volume3/docker/moonshine-asr && python -c "from server import app; print('import ok')"`

---

### Task 7: Integration test

**Step 1: Start server locally**

Run: `cd /volume3/docker/moonshine-asr/src && uvicorn server:app --host 0.0.0.0 --port 8200`

**Step 2: Test health endpoint**

Run: `curl http://localhost:8200/health`
Expected: `{"status": "ok", "mode": "server", "model_loaded": ...}`

**Step 3: Test transcription endpoint**

Run: `curl -X POST http://localhost:8200/v1/audio/transcriptions -F file=@test.wav -F language=auto`
Expected: `{"text": "...", "language": "en"}`

**Step 4: Test stub endpoints return 501**

Run: `curl -X POST http://localhost:8200/v1/audio/subtitles -F file=@test.wav`
Expected: `{"code": "NOT_IMPLEMENTED", ...}`

Run: `curl -X POST http://localhost:8200/v1/audio/translations -F file=@test.wav`
Expected: `{"code": "NOT_IMPLEMENTED", ...}`

---
