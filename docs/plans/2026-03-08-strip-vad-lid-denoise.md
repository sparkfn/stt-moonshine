> **COMPLETED:** This refactor was implemented in commit `34150b8` (2026-03-09). VAD, LID, and denoise are fully removed. WebSocket VAD-gated mode added in `10cd45d`.

# Strip VAD/LID/Denoise — Transcription-Only Refactor

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Simplify moonshine-asr to a pure transcription service — no VAD, no LID, no denoise. A Go audio-preprocessor sidecar will handle all upstream processing.

**Architecture:** Remove Silero VAD, SpeechBrain LID, and noisereduce from all code paths. WebSocket keeps buffer/window/partial/flush mechanics but drops the VAD state machine. HTTP endpoints require explicit language (default "en"). Pipeline becomes a thin route-by-language wrapper.

**Tech Stack:** FastAPI, Moonshine ONNX (unchanged), numpy, soundfile

---

## Task 1: Strip models.py — remove VAD and LID

**Files:**
- Modify: `src/models.py`

**Step 1: Remove VAD and LID imports, globals, and loading**

Remove these globals (lines 15-16):
```python
_vad_model = None
_lid_classifier = None
```

Remove `_EMPTY_F32` (line 24), `_torch` (line 27).

Remove `import copy` (line 6), `import threading` (unused after lock removal check).

In `load_all_models()`:
- Remove `_vad_model`, `_lid_classifier` from global declarations (line 56)
- Remove `_torch` from global declarations (line 57)
- Remove `import torch` and `_torch = torch` (lines 63-64)
- Remove entire Silero VAD loading block (lines 75-83)
- Remove entire SpeechBrain LID loading block (lines 85-100)
- Remove torch-based GPU memory reporting at end (lines 143-146) — replace with simple log
- Keep: Moonshine EN/ZH loading, warmup, `_models_loaded`, `_last_used`, `_numpy_input`

**Step 2: Remove VAD functions and StreamingVAD class**

Delete entirely:
- `is_speech()` (lines 195-226)
- `vad_confidence()` (lines 229-245)
- `reset_vad_state()` (lines 248-254)
- `VAD_FRAME_SAMPLES` constant (line 258)
- `StreamingVAD` class (lines 261-406)
- `_lid_unavailable_warned` (line 409)
- `detect_language()` (lines 412-429)

**Step 3: Simplify unload_all_models()**

Remove `_vad_model` and `_lid_classifier` from unload (lines 151, 159-160). Remove torch cuda cache clear (lines 163-168) — ONNX RT manages its own memory.

**Step 4: Simplify subsystem_status()**

Remove `vad` and `lid` keys. Only report `stt_en` and `stt_zh`.

```python
def subsystem_status() -> dict:
    return {
        "stt_en": _moonshine_en is not None,
        "stt_zh": _moonshine_zh is not None,
    }
```

**Step 5: Remove torch dependency**

After removing all VAD/LID code, `torch` is no longer needed in models.py. Remove `import torch` from `load_all_models()`. The `_torch` global goes away entirely. GPU info for health endpoint will use a try/import pattern in server.py instead.

**Step 6: Verify syntax**

```bash
python3 -c "import py_compile; py_compile.compile('src/models.py', doraise=True); print('OK')"
```

**Step 7: Commit**

```bash
git add src/models.py
git commit -m "refactor: strip VAD and LID from models.py — transcription only"
```

---

## Task 2: Strip pipeline.py — remove denoise and LID

**Files:**
- Modify: `src/pipeline.py`

**Step 1: Rewrite pipeline.py**

Replace entire file with:

```python
"""Transcription pipeline: route by language and transcribe."""
from __future__ import annotations
import time
import numpy as np
from logger import log
import models

TARGET_SR = 16000


def process_audio(
    audio: np.ndarray,
    sr: int,
    lang_code: str | None = None,
) -> tuple[str, str]:
    """
    Route to correct model and transcribe.

    Args:
        audio: float32 audio array
        sr: sample rate
        lang_code: 'en', 'zh', or None (defaults to 'en')

    Returns:
        (transcribed_text, language)
    """
    t0 = time.time()

    lang = lang_code if lang_code in ("en", "zh") else "en"

    if lang == "zh":
        text = models.transcribe_zh(audio, sr)
    else:
        text = models.transcribe_en(audio, sr)

    elapsed_ms = round((time.time() - t0) * 1000)
    log.bind(
        lang=lang,
        audio_s=round(len(audio) / sr, 2),
        text_len=len(text),
        elapsed_ms=elapsed_ms,
    ).debug("pipeline_transcribed")

    return text, lang
```

**Step 2: Verify syntax**

```bash
python3 -c "import py_compile; py_compile.compile('src/pipeline.py', doraise=True); print('OK')"
```

**Step 3: Commit**

```bash
git add src/pipeline.py
git commit -m "refactor: strip denoise and LID from pipeline — route-and-transcribe only"
```

---

## Task 3: Strip server.py — remove VAD state machine and bandpass

**Files:**
- Modify: `src/server.py`

**Step 1: Remove bandpass filter and VAD imports**

Remove:
- `from scipy.signal import butter, sosfilt` (line 18)
- `_BANDPASS_SOS` constant (line 47)
- `_telephony_bandpass()` function (lines 82-84)

**Step 2: Remove VAD-related config variables**

Remove these lines:
- `ASR_USE_SERVER_VAD = ...` (line 165)
- `WS_SKIP_DENOISE = ...` (line 166)

Remove from `lifespan()` log: `use_server_vad=ASR_USE_SERVER_VAD`, `ws_skip_denoise=WS_SKIP_DENOISE` (lines 238-239).

**Step 3: Simplify _do_transcribe()**

Remove `skip_denoise` parameter:

```python
def _do_transcribe(audio, sr, lang_code) -> tuple[str, str]:
    """Run pipeline transcription. Returns (text, language)."""
    models.touch()
    return pipeline.process_audio(audio, sr, lang_code)
```

**Step 4: Simplify _transcribe_with_context()**

Remove `use_vad` parameter, bandpass filter, and VAD check:

```python
async def _transcribe_with_context(
    audio_bytes: bytes | bytearray,
    pad_silence: bool = False,
    lang_code: str | None = None,
) -> tuple[str, str, str | None]:
    """Transcribe PCM audio with optional silence padding.

    Returns (text, language, error_code).
    error_code is None on success.
    """
    n_audio = len(audio_bytes)
    n_total = n_audio
    if pad_silence:
        n_total += _SILENCE_PAD_BYTES_LEN

    if n_total == 0:
        return "", "", None

    buf = bytearray(n_total)
    buf[:n_audio] = audio_bytes

    audio = np.frombuffer(buf, dtype=np.int16).astype(np.float32)
    audio *= _INV_32768

    try:
        text, lang = await asyncio.wait_for(
            _infer_queue.submit(
                lambda: _do_transcribe(audio, TARGET_SR, lang_code),
                priority=0,
            ),
            timeout=REQUEST_TIMEOUT,
        )
        return detect_and_fix_repetitions(text), lang, None
    except asyncio.TimeoutError:
        log.error("ws_transcribe_timeout")
        return "", "", "transcription_timeout"
    except Exception as e:
        log.bind(error=str(e)).error("ws_transcribe_context_error")
        return "", "", "transcription_error"
```

**Step 5: Simplify HTTP transcribe endpoint**

In `_normalize_lang_input()`: change so `None`/`"auto"`/`""` returns `"en"` instead of `None`:

```python
def _normalize_lang_input(language: str) -> str:
    """Convert client language param to pipeline code. Defaults to 'en'."""
    if language is None or language.lower() in ("auto", ""):
        return "en"
    return _LANG_NAME_TO_CODE.get(language.lower(), "en")
```

Update `_do_transcribe` calls in HTTP endpoints — remove `skip_denoise` kwarg from the lambda in `transcribe()` and `sse_transcribe_generator()`.

**Step 6: Rewrite WebSocket handler — strip VAD**

The WebSocket handler (`websocket_transcribe`) needs these changes:

1. Remove `use_vad` variable and `vad_param` query param handling (lines 545-548)
2. Remove `_VAD_RECV_TIMEOUT` constant (line 566)
3. Remove `streaming_vad` creation block (lines 577-586)
4. Remove `use_server_vad` from handshake response (line 594) — or set it to `false` always
5. Remove the entire receive timeout block (lines 599-643) — simplify to just `data = await websocket.receive()`
6. In flush handler: remove `streaming_vad.reset()` (line 687), remove `use_vad` from `_transcribe_with_context` call (line 666)
7. In reset handler: remove `streaming_vad.reset()` (line 694)
8. In config handler: remove entire `use_server_vad` handling block (lines 703-711), remove from config response (line 716)
9. Remove entire VAD audio processing block (lines 752-791): the `streaming_vad` feeding, `_vad_event`, `speech_start`/`speech_end` handling, and VAD-triggered transcription
10. Keep: buffer accumulation (line 764), partial transcription trigger on buffer full (lines 793-822)

The simplified audio frame handling becomes:

```python
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
```

**Step 7: Verify syntax**

```bash
python3 -c "import py_compile; py_compile.compile('src/server.py', doraise=True); print('OK')"
```

**Step 8: Commit**

```bash
git add src/server.py
git commit -m "refactor: strip VAD state machine and bandpass from server — client controls flush"
```

---

## Task 4: Strip requirements and Dockerfile

**Files:**
- Modify: `requirements.txt`
- Modify: `Dockerfile`

**Step 1: Remove unused dependencies from requirements.txt**

Remove these lines:
- `noisereduce`
- `silero-vad==6.2.1`
- `speechbrain==1.0.3`
- `huggingface_hub<1.0`
- `scipy<2.0` (was only used for bandpass filter)

Keep everything else.

**Step 2: Update Dockerfile**

Remove `SPEECHBRAIN_CACHE` env var (line 16). Keep `HF_HOME` and `MOONSHINE_VOICE_CACHE`.

**Step 3: Verify requirements parse**

```bash
pip install --dry-run -r requirements.txt 2>&1 | head -5
```

**Step 4: Commit**

```bash
git add requirements.txt Dockerfile
git commit -m "refactor: remove VAD/LID/denoise dependencies — smaller image"
```

---

## Task 5: Update schemas and config

**Files:**
- Modify: `src/schemas.py`
- Modify: `src/config.py`

**Step 1: Update HealthResponse in schemas.py**

The `subsystems` field now only has `stt_en` and `stt_zh`. No schema change needed (it's `Optional[dict]`), but update the docstring if present.

**Step 2: Clean up config.py**

Remove any VAD/denoise-related env var validation if present. Check for `DENOISE_ENABLED`, `LID_DEVICE`, `ASR_USE_SERVER_VAD` references and remove.

**Step 3: Verify syntax**

```bash
python3 -c "import py_compile; py_compile.compile('src/schemas.py', doraise=True); py_compile.compile('src/config.py', doraise=True); print('OK')"
```

**Step 4: Commit**

```bash
git add src/schemas.py src/config.py
git commit -m "refactor: clean up schemas and config after VAD/LID removal"
```

---

## Task 6: Update E2E tests

**Files:**
- Delete: `E2Etest/test_vad.py`
- Modify: `E2Etest/test_websocket.py`
- Modify: `E2Etest/conftest.py`

**Step 1: Delete test_vad.py**

The entire file tests server-side VAD which no longer exists.

```bash
git rm E2Etest/test_vad.py
```

**Step 2: Update test_websocket.py**

- Any test that checks `use_server_vad` in handshake: update to expect `false` or remove the field check
- Any test that sends `use_server_vad` in config: remove or update
- Keep all other tests (flush, reset, config language, streaming, etc.)

**Step 3: Remove vad marker from conftest.py**

Remove the `vad` marker registration if it exists in `conftest.py`.

**Step 4: Verify tests parse**

```bash
python3 -m pytest E2Etest/test_websocket.py --collect-only 2>&1 | tail -5
```

**Step 5: Commit**

```bash
git add -A E2Etest/
git commit -m "test: remove VAD tests, update WebSocket tests for no-VAD server"
```

---

## Task 7: Build, restart, run full test suite

**Step 1: Build Docker image**

```bash
docker compose build
```

**Step 2: Restart container**

```bash
docker compose down && docker compose up -d
```

**Step 3: Wait for health**

```bash
sleep 15 && curl -s http://localhost:8200/health | python3 -m json.tool
```

Expected: `status: ok`, no `vad`/`lid` in subsystems.

**Step 4: Run E2E tests**

```bash
python3 -m pytest E2Etest/test_websocket.py -v --timeout=120
```

Expected: all tests pass.

**Step 5: Run HTTP endpoint tests**

```bash
python3 -m pytest E2Etest/ -v --timeout=120 -k "not realtime"
```

Expected: all pass (except pre-existing WER benchmarks).

**Step 6: Commit final state if any fixups needed**

```bash
git add -A && git commit -m "fix: test fixups after VAD/LID/denoise removal"
```

---

## Summary of removals

| Component | Files affected | What's removed |
|-----------|---------------|----------------|
| Silero VAD | models.py, server.py, requirements.txt | `StreamingVAD`, `is_speech()`, `vad_confidence()`, `reset_vad_state()`, VAD state machine in WS handler |
| SpeechBrain LID | models.py, pipeline.py, requirements.txt | `detect_language()`, `_lid_classifier`, auto-detect in pipeline |
| noisereduce | pipeline.py, requirements.txt | `denoise()`, `_nr` import |
| scipy bandpass | server.py, requirements.txt | `_telephony_bandpass()`, `_BANDPASS_SOS`, `butter`/`sosfilt` |
| torch (in models.py) | models.py | `_torch`, `_vad_model`, CUDA memory reporting |
| Test coverage | test_vad.py | Entire file deleted |
