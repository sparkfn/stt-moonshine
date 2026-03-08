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
import re
import asyncio
import concurrent.futures
import time
import heapq
import dataclasses
import numpy as np
from scipy.signal import butter, sosfilt

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

# Bandpass filter coefficients: computed once, reused for all WS transcriptions
_BANDPASS_SOS = butter(4, [300, 3400], btype="bandpass", fs=TARGET_SR, output="sos")

# Multiplier for int16 → float32 normalization (avoids division per call)
_INV_32768 = np.float32(1.0 / 32768.0)

# Pre-compiled regex for repetition detection
_WORD_REPEAT_RE = re.compile(r'\b(\w+)( \1){2,}\b')

# Pre-serialized static WS responses
_WS_EMPTY_FINAL = _json_encode({"text": "", "is_partial": False, "is_final": True})
_WS_BUFFER_RESET = _json_encode({"status": "buffer_reset"})
_SSE_DONE = f"data: {_json_encode({'done': True})}\n\n"

# Language name ↔ code mapping for qwen3-asr compatibility
_LANG_NAME_TO_CODE = {
    "english": "en", "chinese": "zh", "mandarin": "zh",
    "en": "en", "zh": "zh",
}
_LANG_CODE_TO_NAME = {"en": "English", "zh": "Chinese"}


def _normalize_lang_input(language: str) -> str | None:
    """Convert client language param to pipeline code. Returns None for 'auto'."""
    if language is None or language.lower() in ("auto", ""):
        return None
    return _LANG_NAME_TO_CODE.get(language.lower(), language.lower())


def _format_lang_output(detected: str, requested: str) -> str:
    """Format language for response. Echo back client's format if explicit."""
    if requested is None or requested.lower() in ("auto", ""):
        return _LANG_CODE_TO_NAME.get(detected, detected)
    return requested


def _telephony_bandpass(audio: np.ndarray) -> np.ndarray:
    """Apply pre-computed bandpass filter. Returns float32."""
    return sosfilt(_BANDPASS_SOS, audio).astype(np.float32)


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
ASR_USE_SERVER_VAD = os.getenv("ASR_USE_SERVER_VAD", "true").lower() == "true"
WS_SKIP_DENOISE = os.getenv("WS_SKIP_DENOISE", "true").lower() == "true"

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
        use_server_vad=ASR_USE_SERVER_VAD,
        ws_skip_denoise=WS_SKIP_DENOISE,
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
    return {
        "status": "ok",
        "mode": "server",
        "model_loaded": models.is_loaded(),
        "model_id": f"{os.getenv('MOONSHINE_EN_MODEL', '')} + {os.getenv('MOONSHINE_ZH_MODEL', '')}" if models.is_loaded() else None,
        "cuda": cuda_available,
        **gpu_info,
    }


def _do_transcribe(audio, sr, lang_code, skip_denoise=False) -> tuple[str, str]:
    """Run pipeline transcription. Returns (text, language)."""
    return pipeline.process_audio(audio, sr, lang_code, skip_denoise=skip_denoise)


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
        return error_response("FILE_TOO_LARGE", f"Upload exceeds {MAX_UPLOAD_BYTES // (1024*1024)}MB limit", 413)
    log.bind(endpoint="/v1/audio/transcriptions", filename=file.filename, size_bytes=len(audio_bytes), language=language).info("request_received")
    t0 = time.time()
    try:
        audio, sr = _decode_audio(audio_bytes)
    except Exception as e:
        return error_response("AUDIO_DECODE_FAILED", "Could not decode the uploaded audio file", 422, endpoint="/v1/audio/transcriptions", fileSize=len(audio_bytes))

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
        return error_response("TRANSCRIPTION_TIMEOUT", "Transcription timed out", 504, elapsed=elapsed)

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
        yield f"data: {_json_encode({'code': 'SSE_STREAM_ERROR', 'message': 'Internal streaming error', 'statusCode': 500})}\n\n"


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
        return error_response("FILE_TOO_LARGE", f"Upload exceeds {MAX_UPLOAD_BYTES // (1024*1024)}MB limit", 413)
    log.bind(endpoint="/v1/audio/transcriptions/stream", filename=file.filename, size_bytes=len(audio_bytes), language=language).info("request_received")
    try:
        audio, sr = _decode_audio(audio_bytes)
    except Exception as e:
        return error_response("AUDIO_DECODE_FAILED", "Could not decode the uploaded audio file", 422, endpoint="/v1/audio/transcriptions/stream", fileSize=len(audio_bytes))

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
    n_audio = len(audio_bytes)
    n_overlap = len(overlap)
    n_total = n_audio + n_overlap
    if pad_silence:
        n_total += _SILENCE_PAD_BYTES_LEN

    if n_total == 0:
        return "", ""

    # Single allocation; trailing zeros serve as silence padding (bytearray inits to 0)
    buf = bytearray(n_total)
    if n_overlap:
        buf[:n_overlap] = overlap
    buf[n_overlap:n_overlap + n_audio] = audio_bytes
    # Silence: remaining bytes already zero from bytearray init

    # int16 → float32 with in-place multiply (avoids extra array from division)
    audio = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
    audio *= _INV_32768
    audio = _telephony_bandpass(audio)

    if use_vad and not models.is_speech(audio):
        return "", ""

    try:
        text, lang = await asyncio.wait_for(
            _infer_queue.submit(
                lambda: _do_transcribe(audio, TARGET_SR, lang_code, skip_denoise=WS_SKIP_DENOISE),
                priority=0,
            ),
            timeout=REQUEST_TIMEOUT,
        )
        return detect_and_fix_repetitions(text), lang
    except asyncio.TimeoutError:
        return "[timeout]", ""
    except Exception as e:
        log.bind(error=str(e)).error("ws_transcribe_context_error")
        return "[transcription error]", ""


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
    use_vad: bool = ASR_USE_SERVER_VAD
    vad_param = websocket.query_params.get("use_server_vad")
    if vad_param is not None:
        use_vad = vad_param.lower() in ("true", "1", "yes")
    client_sr = int(websocket.query_params.get("sample_rate", str(TARGET_SR)))
    if client_sr not in (8000, 16000):
        ws_log.bind(sample_rate=client_sr).error("ws_unsupported_sample_rate")
        await websocket.send_json({
            "code": "UNSUPPORTED_SAMPLE_RATE",
            "message": f"sample_rate must be 8000 or 16000, got {client_sr}",
            "statusCode": 400,
        })
        await websocket.close()
        return

    chunk_count = 0

    # Timeout for receive() when VAD speech is active — if no audio arrives
    # within this window, assume the client stopped sending and force speech_end.
    # Many clients stop forwarding audio when their own VAD detects silence,
    # so the server never sees silence frames to trigger speech_end normally.
    _VAD_RECV_TIMEOUT = 0.6  # seconds (slightly above redemption_ms=400)

    # Local references for hot-loop performance
    _frombuffer = np.frombuffer
    _ws_send_text = websocket.send_text
    _needs_resample = client_sr != TARGET_SR

    try:
        await _ensure_model_loaded()

        # Create StreamingVAD AFTER models are loaded — deepcopy needs _vad_model
        streaming_vad = models.StreamingVAD(
            positive_threshold=0.3, negative_threshold=0.25,
            redemption_ms=400, min_speech_ms=250,
        ) if use_vad else None

        await _ws_send_text(_json_encode({
            "status": "connected",
            "sample_rate": client_sr,
            "format": "pcm_s16le",
            "buffer_size": WS_BUFFER_SIZE,
            "window_max_s": WS_WINDOW_MAX_S,
            "use_server_vad": use_vad,
        }))

        while True:
            try:
                # When VAD detects active speech, use a short receive timeout.
                # If the client stops sending audio (common when client-side VAD
                # detects silence), we force speech_end rather than waiting for
                # the client's flush watchdog (typically 3s).
                _recv_timeout = _VAD_RECV_TIMEOUT if (streaming_vad and streaming_vad.speech_active) else None
                try:
                    if _recv_timeout is not None:
                        data = await asyncio.wait_for(websocket.receive(), timeout=_recv_timeout)
                    else:
                        data = await websocket.receive()
                except asyncio.TimeoutError:
                    # No audio arrived while speech was active — client likely
                    # stopped sending. Force speech_end to avoid waiting for
                    # the client's silence watchdog (saves ~2-3s latency).
                    if streaming_vad and streaming_vad.speech_active:
                        ws_log.bind(
                            frame=audio_frame_count,
                            timeout_s=_VAD_RECV_TIMEOUT,
                            window_bytes=len(audio_window) + len(audio_buffer),
                        ).info("vad_speech_end_timeout")
                        audio_window.extend(audio_buffer)
                        audio_buffer.clear()

                        if len(audio_window) > 0:
                            t_infer = time.time()
                            text, _ = await _transcribe_with_context(
                                audio_window, b"", pad_silence=True,
                                lang_code=lang_code, use_vad=False,
                            )
                            inference_ms = round((time.time() - t_infer) * 1000)
                            chunk_count += 1
                            if text and text.strip():
                                ws_log.bind(text=text[:80], text_len=len(text.strip()), is_final=True, inference_ms=inference_ms).info("vad_transcript_final_timeout")
                            await _ws_send_text(_json_encode({
                                "text": text,
                                "is_partial": False,
                                "is_final": True,
                            }))
                        audio_window.clear()
                        streaming_vad.reset()
                    continue

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
                                text, _ = await _transcribe_with_context(
                                    audio_window, b"", pad_silence=True,
                                    lang_code=lang_code, use_vad=use_vad,
                                )
                                inference_ms = round((time.time() - t_infer) * 1000)
                                chunk_count += 1
                                if text.strip():
                                    ws_log.bind(text=text[:80], text_len=len(text.strip()), is_final=True, inference_ms=inference_ms).debug("ws_transcript")
                                await _ws_send_text(_json_encode({
                                    "text": text,
                                    "is_partial": False,
                                    "is_final": True,
                                }))
                            else:
                                await _ws_send_text(_WS_EMPTY_FINAL)
                            audio_window.clear()
                            if streaming_vad:
                                streaming_vad.reset()

                        elif action == "reset":
                            ws_log.info("ws_reset")
                            audio_buffer.clear()
                            audio_window.clear()
                            if streaming_vad:
                                streaming_vad.reset()
                            await _ws_send_text(_WS_BUFFER_RESET)

                        elif action == "config":
                            new_lang = msg.get("language")
                            if new_lang == "auto":
                                lang_code = None
                            elif new_lang:
                                lang_code = new_lang
                            if "use_server_vad" in msg:
                                use_vad = bool(msg["use_server_vad"])
                                if use_vad and streaming_vad is None:
                                    streaming_vad = models.StreamingVAD(
                                        positive_threshold=0.3, negative_threshold=0.25,
                                        redemption_ms=400, min_speech_ms=250,
                                    )
                                elif not use_vad and streaming_vad is not None:
                                    streaming_vad = None
                            ws_log.bind(language=lang_code or "auto", use_server_vad=use_vad).info("ws_configured")
                            await _ws_send_text(_json_encode({
                                "status": "configured",
                                "language": lang_code or "auto",
                                "use_server_vad": use_vad,
                            }))

                        else:
                            ws_log.bind(action=action).warning("ws_unknown_action")
                            await _ws_send_text(_json_encode({
                                "code": "UNKNOWN_ACTION",
                                "message": f"Unknown action: {action!r}",
                                "statusCode": 400,
                            }))

                    except (ValueError, KeyError):
                        ws_log.warning("ws_invalid_json")
                        await _ws_send_text(_json_encode({
                            "code": "INVALID_JSON",
                            "message": "Invalid JSON command",
                            "statusCode": 400,
                        }))

                elif "bytes" in data:
                    audio_frame_count += 1
                    incoming = data["bytes"]
                    if _needs_resample:
                        incoming = _resample_pcm_bytes(incoming, client_sr)

                    # Feed every frame through streaming VAD
                    _vad_event = None
                    if streaming_vad:
                        # int16 → float32 with in-place multiply (one less array copy)
                        _incoming_float = _frombuffer(incoming, dtype=np.int16).astype(np.float32)
                        _incoming_float *= _INV_32768
                        _vad_event = streaming_vad.process(_incoming_float)
                        if _vad_event == "speech_start":
                            ws_log.bind(frame=audio_frame_count, window_bytes=len(audio_window) + len(audio_buffer)).debug("vad_speech_start")
                        elif _vad_event == "speech_end":
                            ws_log.bind(frame=audio_frame_count, window_bytes=len(audio_window) + len(audio_buffer)).info("vad_speech_end")

                    audio_buffer.extend(incoming)

                    # VAD detected end of speech — transcribe immediately
                    if _vad_event == "speech_end":
                        audio_window.extend(audio_buffer)
                        audio_buffer.clear()

                        if len(audio_window) > 0:
                            t_infer = time.time()
                            text, _ = await _transcribe_with_context(
                                audio_window, b"", pad_silence=True,
                                lang_code=lang_code, use_vad=False,
                            )
                            inference_ms = round((time.time() - t_infer) * 1000)
                            chunk_count += 1
                            if text and text.strip():
                                ws_log.bind(text=text[:80], text_len=len(text.strip()), is_final=True, inference_ms=inference_ms).info("vad_transcript_final")
                            await _ws_send_text(_json_encode({
                                "text": text,
                                "is_partial": False,
                                "is_final": True,
                            }))
                        audio_window.clear()
                        streaming_vad.reset()

                    # Normal buffer-based partial transcription
                    elif len(audio_buffer) >= WS_BUFFER_SIZE:
                        audio_window.extend(audio_buffer)
                        audio_buffer.clear()

                        if len(audio_window) > WS_WINDOW_MAX_BYTES:
                            trim = len(audio_window) - WS_WINDOW_MAX_BYTES
                            trim = (trim // 2) * 2
                            del audio_window[:trim]

                        t_infer = time.time()
                        text, _ = await _transcribe_with_context(
                            audio_window, b"", pad_silence=False,
                            lang_code=lang_code, use_vad=use_vad,
                        )
                        inference_ms = round((time.time() - t_infer) * 1000)
                        chunk_count += 1
                        if text:
                            if text.strip():
                                ws_log.bind(text=text[:80], text_len=len(text.strip()), is_final=False, inference_ms=inference_ms).debug("ws_transcript")
                            await _ws_send_text(_json_encode({
                                "text": text,
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
            await _ws_send_text(_json_encode({"code": "WEBSOCKET_ERROR", "message": "Internal server error", "statusCode": 500}))
        except Exception:
            pass
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
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
