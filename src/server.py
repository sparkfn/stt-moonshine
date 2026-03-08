from __future__ import annotations
from logger import log, set_request_id, reset_request_id
from errors import error_response, ws_error, INVALID_REQUEST, SERVER_ERROR
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
import sys
import io
import os
import uuid as _uuid_module
import re
import asyncio
import concurrent.futures
import time
import heapq
import dataclasses
import numpy as np
try:
    import librosa as _librosa
except ImportError:
    _librosa = None

try:
    import orjson
    def _json_encode(obj) -> str:
        return orjson.dumps(obj).decode()
    def _json_decode(s):
        return orjson.loads(s)
except ImportError:
    import json
    def _json_encode(obj) -> str:
        return json.dumps(obj)
    def _json_decode(s):
        return json.loads(s)

import soundfile as sf
import models
import pipeline

TARGET_SR = 16000

# ── Pre-computed constants (avoid recomputation on every call) ──────────

# Multiplier for int16 → float32 normalization (avoids division per call)
_INV_32768 = np.float32(1.0 / 32768.0)

# Pre-compiled regex for repetition detection
_WORD_REPEAT_RE = re.compile(r'\b(\w+)( \1){2,}\b')

# Pre-serialized static WS responses
_WS_EMPTY_FINAL = _json_encode({"text": "", "language": None, "is_partial": False, "is_final": True})
_WS_BUFFER_RESET = _json_encode({"status": "buffer_reset"})
_SSE_DONE = f"data: {_json_encode({'done': True})}\n\n"

# Language name ↔ code mapping for qwen3-asr compatibility
_LANG_NAME_TO_CODE = {
    "english": "en", "chinese": "zh", "mandarin": "zh",
    "en": "en", "zh": "zh",
}
_LANG_CODE_TO_NAME = {"en": "English", "zh": "Chinese"}


def _normalize_lang_input(language: str) -> str:
    """Convert client language param to pipeline code. Defaults to 'en'."""
    if language is None or language.lower() in ("auto", ""):
        return "en"
    return _LANG_NAME_TO_CODE.get(language.lower(), "en")


def _format_lang_output(detected: str, requested: str) -> str:
    """Format language for response. Echo back client's format if explicit."""
    if requested is None or requested.lower() in ("auto", ""):
        return _LANG_CODE_TO_NAME.get(detected, detected)
    return requested


def _resample_pcm_bytes(pcm_bytes: bytes, orig_sr: int) -> bytes:
    if orig_sr == TARGET_SR:
        return pcm_bytes
    if _librosa is None:
        raise RuntimeError("librosa is required for resampling but is not installed")
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    resampled = _librosa.resample(samples, orig_sr=orig_sr, target_sr=TARGET_SR)
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()


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
                if not job.future.done():
                    job.future.set_result(result)
            except Exception as e:
                if not job.future.done():
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
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "0"))

WS_BUFFER_SIZE = int(os.getenv("WS_BUFFER_SIZE", str(int(TARGET_SR * 2 * 0.2))))
WS_FLUSH_SILENCE_MS = int(os.getenv("WS_FLUSH_SILENCE_MS", "600"))
WS_WINDOW_MAX_S = float(os.getenv("WS_WINDOW_MAX_S", "6.0"))
WS_WINDOW_MAX_BYTES = int(WS_WINDOW_MAX_S * TARGET_SR * 2)
# Pre-computed silence padding length in bytes (int16 samples)
_SILENCE_PAD_BYTES_LEN = int(WS_FLUSH_SILENCE_MS / 1000 * TARGET_SR) * 2


def detect_and_fix_repetitions(text: str, max_repeats: int = 2) -> str:
    if not text or len(text) < 10:
        return text
    text = _WORD_REPEAT_RE.sub(r'\1', text)
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
    """Decode audio file bytes. Returns (float32_array, sample_rate)."""
    return sf.read(io.BytesIO(audio_bytes), dtype='float32')


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
        try:
            await asyncio.sleep(30)
            if IDLE_TIMEOUT <= 0 or not models.is_loaded():
                continue
            if time.time() - models.get_last_used() > IDLE_TIMEOUT:
                async with _model_lock:
                    if models.is_loaded() and time.time() - models.get_last_used() > IDLE_TIMEOUT:
                        await asyncio.get_event_loop().run_in_executor(_infer_executor, models.unload_all_models)
        except Exception as e:
            log.bind(error=str(e)).error("idle_watchdog_error")


@asynccontextmanager
async def lifespan(the_app):
    from config import validate_env
    validate_env()
    _infer_queue.start()
    asyncio.create_task(_idle_watchdog())
    log.bind(
        request_timeout=REQUEST_TIMEOUT,
        idle_timeout=IDLE_TIMEOUT,
        ws_buffer_size=WS_BUFFER_SIZE,
        ws_window_max_s=WS_WINDOW_MAX_S,
    ).info("server_started")
    yield
    t_shutdown = time.time()
    _infer_executor.shutdown(wait=False)
    _infer_queue.stop()
    log.bind(elapsed_ms=round((time.time() - t_shutdown) * 1000)).info("server_shutdown")


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
    subsystems = models.subsystem_status() if models.is_loaded() else {}
    return {
        "status": "ok",
        "mode": "server",
        "model_loaded": models.is_loaded(),
        "model_id": f"{os.getenv('MOONSHINE_EN_MODEL', '')} + {os.getenv('MOONSHINE_ZH_MODEL', '')}" if models.is_loaded() else None,
        "cuda": cuda_available,
        **gpu_info,
        **({"subsystems": subsystems} if subsystems else {}),
    }


def _do_transcribe(audio, sr, lang_code) -> tuple[str, str]:
    """Run pipeline transcription. Returns (text, language)."""
    models.touch()
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
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        return error_response("file_too_large", f"Upload exceeds {MAX_UPLOAD_BYTES // (1024*1024)}MB limit", 413, param="file")
    log.bind(endpoint="/v1/audio/transcriptions", filename=file.filename, size_bytes=len(audio_bytes), language=language).info("request_received")
    t0 = time.time()
    try:
        audio, sr = _decode_audio(audio_bytes)
    except Exception as e:
        log.bind(error=str(e), filename=file.filename).warning("audio_decode_failed")
        return error_response("audio_decode_failed", "Could not decode the uploaded audio file", 422, param="file")

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
        elapsed = round(time.time() - t0, 2)
        log.bind(endpoint="/v1/audio/transcriptions", elapsed_s=elapsed).error("transcription_timeout")
        return error_response("transcription_timeout", "Transcription timed out", 504)

    text = detect_and_fix_repetitions(text)
    out_lang = _format_lang_output(detected_lang, language)
    elapsed_ms = round((time.time() - t0) * 1000)
    log.bind(endpoint="/v1/audio/transcriptions", elapsed_ms=elapsed_ms, text_len=len(text), lang=out_lang).info("request_completed")
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
    return error_response("not_implemented", "Translation is not available in moonshine-asr", 501)


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
    return error_response("not_implemented", "Subtitles are not available in moonshine-asr", 501)


async def sse_transcribe_generator(audio, sr, lang_code, requested_language: str = "auto"):
    audio_duration = len(audio) / sr
    t0 = time.time()
    chunk_count = 0
    log.bind(endpoint="/v1/audio/transcriptions/stream", audio_s=round(audio_duration, 2), lang=lang_code or "auto").info("sse_stream_started")
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
            yield f"data: {_json_encode(data)}\n\n"
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
                yield f"data: {_json_encode(data)}\n\n"
                chunk_index += 1

                if is_last:
                    break
                start = end - OVERLAP_SAMPLES

        elapsed_ms = round((time.time() - t0) * 1000)
        log.bind(endpoint="/v1/audio/transcriptions/stream", chunks=chunk_count, elapsed_ms=elapsed_ms).info("sse_stream_completed")
        yield _SSE_DONE

    except Exception as e:
        elapsed_ms = round((time.time() - t0) * 1000)
        log.bind(endpoint="/v1/audio/transcriptions/stream", error=str(e), elapsed_ms=elapsed_ms, chunks=chunk_count).error("sse_stream_error")
        yield f"data: {_json_encode({'error': {'message': 'Internal streaming error', 'type': 'server_error', 'param': None, 'code': 'sse_stream_error'}})}\n\n"


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
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        return error_response("file_too_large", f"Upload exceeds {MAX_UPLOAD_BYTES // (1024*1024)}MB limit", 413, param="file")
    log.bind(endpoint="/v1/audio/transcriptions/stream", filename=file.filename, size_bytes=len(audio_bytes), language=language).info("request_received")
    try:
        audio, sr = _decode_audio(audio_bytes)
    except Exception as e:
        log.bind(error=str(e), filename=file.filename).warning("audio_decode_failed")
        return error_response("audio_decode_failed", "Could not decode the uploaded audio file", 422, param="file")

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
    pad_silence: bool = False,
    lang_code: str | None = None,
) -> tuple[str, str, str | None]:
    n_audio = len(audio_bytes)
    n_total = n_audio + (_SILENCE_PAD_BYTES_LEN if pad_silence else 0)
    if n_total == 0:
        return "", "", None
    buf = bytearray(n_total)
    buf[:n_audio] = audio_bytes
    audio = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
    audio *= _INV_32768
    try:
        text, lang = await asyncio.wait_for(
            _infer_queue.submit(lambda: _do_transcribe(audio, TARGET_SR, lang_code), priority=0),
            timeout=REQUEST_TIMEOUT,
        )
        return detect_and_fix_repetitions(text), lang, None
    except asyncio.TimeoutError:
        log.error("ws_transcribe_timeout")
        return "", "", "transcription_timeout"
    except Exception as e:
        log.bind(error=str(e)).error("ws_transcribe_context_error")
        return "", "", "transcription_error"


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()

    ws_req_id = websocket.query_params.get("request_id") or str(_uuid_module.uuid4())
    ws_id = ws_req_id[:12]
    token = set_request_id(ws_req_id)
    ws_log = log.bind(ws_id=ws_id)

    t_connect = time.time()
    msg_count = 0
    audio_frame_count = 0

    ws_log.info("ws_connected")

    audio_buffer = bytearray()
    audio_window = bytearray()
    lang_code: str | None = None
    client_sr = int(websocket.query_params.get("sample_rate", str(TARGET_SR)))
    if client_sr not in (8000, 16000):
        ws_log.bind(sample_rate=client_sr).error("ws_unsupported_sample_rate")
        await websocket.send_json(ws_error(
            "unsupported_sample_rate",
            f"sample_rate must be 8000 or 16000, got {client_sr}",
            INVALID_REQUEST,
        ))
        await websocket.close()
        return

    chunk_count = 0

    # Local references for hot-loop performance
    _frombuffer = np.frombuffer
    _ws_send_text = websocket.send_text
    _needs_resample = client_sr != TARGET_SR

    try:
        await _ensure_model_loaded()

        await _ws_send_text(_json_encode({
            "status": "connected",
            "sample_rate": client_sr,
            "format": "pcm_s16le",
            "buffer_size": WS_BUFFER_SIZE,
            "window_max_s": WS_WINDOW_MAX_S,
        }))

        while True:
            try:
                data = await websocket.receive()

                msg_count += 1

                if data.get("type") == "websocket.disconnect":
                    break

                if "text" in data:
                    try:
                        msg = _json_decode(data["text"])
                        action = msg.get("action", "")

                        if action == "flush":
                            window_s = round(len(audio_window) / TARGET_SR / 2, 2) if len(audio_window) > 0 else 0
                            ws_log.bind(window_s=window_s).info("ws_flush")
                            if audio_buffer:
                                audio_window.extend(audio_buffer)
                                audio_buffer.clear()

                            if len(audio_window) > 0:
                                t_infer = time.time()
                                text, lang, err = await _transcribe_with_context(
                                    audio_window, pad_silence=True,
                                    lang_code=lang_code,
                                )
                                inference_ms = round((time.time() - t_infer) * 1000)
                                chunk_count += 1
                                if err:
                                    err_msg = ws_error(err, f"Transcription failed: {err}")
                                    err_msg["is_final"] = True
                                    await _ws_send_text(_json_encode(err_msg))
                                else:
                                    if text.strip():
                                        ws_log.bind(text=text[:80], text_len=len(text.strip()), lang=lang, is_final=True, inference_ms=inference_ms).debug("ws_transcript")
                                    await _ws_send_text(_json_encode({
                                        "text": text,
                                        "language": lang or None,
                                        "is_partial": False,
                                        "is_final": True,
                                    }))
                            else:
                                await _ws_send_text(_WS_EMPTY_FINAL)
                            audio_window.clear()

                        elif action == "reset":
                            ws_log.info("ws_reset")
                            audio_buffer.clear()
                            audio_window.clear()
                            await _ws_send_text(_WS_BUFFER_RESET)

                        elif action == "config":
                            new_lang = msg.get("language")
                            if new_lang == "auto":
                                lang_code = None
                            elif new_lang:
                                lang_code = new_lang
                            ws_log.bind(language=lang_code or "auto").info("ws_configured")
                            await _ws_send_text(_json_encode({
                                "status": "configured",
                                "language": lang_code or "auto",
                            }))

                        else:
                            ws_log.bind(action=action).warning("ws_unknown_action")
                            await _ws_send_text(_json_encode(ws_error(
                                "unknown_action",
                                f"Unknown action: {action!r}",
                                INVALID_REQUEST,
                            )))

                    except ValueError as e:
                        ws_log.bind(error=str(e)).warning("ws_invalid_json")
                        await _ws_send_text(_json_encode(ws_error(
                            "invalid_json",
                            "Invalid JSON command",
                            INVALID_REQUEST,
                        )))
                    except KeyError as e:
                        ws_log.bind(error=str(e)).warning("ws_missing_field")
                        await _ws_send_text(_json_encode(ws_error(
                            "missing_field",
                            f"Missing required field: {e}",
                            INVALID_REQUEST,
                        )))

                elif "bytes" in data:
                    audio_frame_count += 1
                    incoming = data["bytes"]
                    if len(incoming) % 2 != 0:
                        incoming = incoming[:-1]
                    if not incoming:
                        continue
                    if _needs_resample:
                        incoming = _resample_pcm_bytes(incoming, client_sr)

                    audio_buffer.extend(incoming)

                    # Buffer-based partial transcription
                    if len(audio_buffer) >= WS_BUFFER_SIZE:
                        audio_window.extend(audio_buffer)
                        audio_buffer.clear()

                        if len(audio_window) > WS_WINDOW_MAX_BYTES:
                            trim = len(audio_window) - WS_WINDOW_MAX_BYTES
                            trim = (trim // 2) * 2
                            del audio_window[:trim]

                        t_infer = time.time()
                        text, lang, err = await _transcribe_with_context(
                            audio_window, pad_silence=False,
                            lang_code=lang_code,
                        )
                        inference_ms = round((time.time() - t_infer) * 1000)
                        chunk_count += 1
                        if err:
                            ws_log.bind(error=err, inference_ms=inference_ms).warning("ws_partial_transcribe_error")
                        elif text:
                            if text.strip():
                                ws_log.bind(text=text[:80], text_len=len(text.strip()), lang=lang, is_final=False, inference_ms=inference_ms).debug("ws_transcript")
                            await _ws_send_text(_json_encode({
                                "text": text,
                                "language": lang or None,
                                "is_partial": True,
                                "is_final": False,
                            }))

            except WebSocketDisconnect:
                discarded_bytes = len(audio_buffer) + len(audio_window)
                if discarded_bytes > 0:
                    ws_log.bind(discarded_bytes=discarded_bytes).info("ws_disconnect_audio_discarded")
                break

    except Exception as e:
        ws_log.bind(error=str(e)).error("ws_error")
        try:
            await _ws_send_text(_json_encode(ws_error("websocket_error", "Internal server error")))
        except Exception as send_err:
            ws_log.bind(error=str(send_err)).warning("ws_error_send_failed")
    finally:
        duration_s = round(time.time() - t_connect, 1)
        ws_log.bind(
            messages=msg_count,
            audio_frames=audio_frame_count,
            transcriptions=chunk_count,
            duration_s=duration_s,
        ).info("ws_disconnected")
        reset_request_id(token)
        try:
            await websocket.close()
        except Exception as close_err:
            ws_log.bind(error=str(close_err)).warning("ws_close_failed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
