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

TARGET_SR = 16000

# Language name ↔ code mapping for qwen3-asr compatibility
_LANG_NAME_TO_CODE = {
    "english": "en", "chinese": "zh", "mandarin": "zh",
    "en": "en", "zh": "zh",
}
_LANG_CODE_TO_NAME = {"en": "English", "zh": "Chinese"}


def _normalize_lang_input(language: str) -> str | None:
    """Convert client language param to pipeline code. Returns None for 'auto'."""
    if language.lower() in ("auto", ""):
        return None
    return _LANG_NAME_TO_CODE.get(language.lower(), language.lower())


def _format_lang_output(detected: str, requested: str) -> str:
    """Format language for response. Echo back client's format if explicit."""
    if requested.lower() not in ("auto", ""):
        # Client specified a language — echo it back as-is
        return requested
    # Auto-detected: return full name
    return _LANG_CODE_TO_NAME.get(detected, detected)


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
_model_lock = None

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
    global _model_lock
    _model_lock = asyncio.Lock()
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

    lang_code = _normalize_lang_input(language)

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
    out_lang = _format_lang_output(detected_lang, language)
    log.info("POST /v1/audio/transcriptions | completed in {:.2f}s text_len={} lang={}", time.time() - t0, len(text), out_lang)
    return {"text": text, "language": out_lang}


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


async def sse_transcribe_generator(audio, sr, lang_code, requested_language: str = "auto"):
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
            data = {"text": text, "language": _format_lang_output(detected_lang, requested_language), "is_final": True, "chunk_index": 0}
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
                    "language": _format_lang_output(detected_lang, requested_language),
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

    lang_code = _normalize_lang_input(language)
    return StreamingResponse(
        sse_transcribe_generator(audio, sr, lang_code, requested_language=language),
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
) -> tuple[str, str]:
    """Transcribe PCM audio with optional silence padding. Returns (text, language)."""
    full_audio = bytearray()
    if overlap:
        full_audio.extend(overlap)
    full_audio.extend(audio_bytes)

    if pad_silence:
        silence_bytes = int((WS_FLUSH_SILENCE_MS / 1000) * TARGET_SR * 2)
        full_audio.extend(bytes(silence_bytes))

    if len(full_audio) == 0:
        return "", ""

    audio = np.frombuffer(full_audio, dtype=np.int16)
    audio = audio.astype(np.float32) / 32768.0
    audio = _telephony_bandpass(audio, TARGET_SR)
    sr = TARGET_SR

    if use_vad and not models.is_speech(audio):
        return "", ""

    try:
        text, lang = await asyncio.wait_for(
            _infer_queue.submit(
                lambda: _do_transcribe(audio, sr, lang_code),
                priority=0,
            ),
            timeout=REQUEST_TIMEOUT,
        )
        return detect_and_fix_repetitions(text), lang
    except asyncio.TimeoutError:
        return "[timeout]", ""
    except Exception as e:
        log.error("_transcribe_with_context error: {}", e)
        return f"[error: {e}]", ""


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()

    ws_req_id = websocket.query_params.get("request_id") or str(_uuid_module.uuid4())
    token = set_request_id(ws_req_id)
    log.info("[WS] Client connected")

    audio_buffer = bytearray()
    audio_window = bytearray()
    lang_code: str | None = "en"
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
                                text, _ = await _transcribe_with_context(
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
                                text, _ = await _transcribe_with_context(
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
                            text, _ = await _transcribe_with_context(
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
                        text, _ = await _transcribe_with_context(
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
