> **ARCHIVED:** This design doc describes the original pipeline with VAD, LID, and denoise components. These were stripped in commit `34150b8` (2026-03-09). See `2026-03-08-strip-vad-lid-denoise.md` for the refactor plan and the current README for up-to-date architecture.

# Moonshine ASR Service Design

## Goal

Drop-in replacement for qwen3-asr. Identical API surface, same request/response schemas, same WebSocket protocol. Clients switch by changing the URL only.

## Pipeline

```
audio_in -> noisereduce -> silero_vad -> speech_chunks -> silero_lid -> router -> moonshine_en / moonshine_zh -> text
```

### Components

| Stage | Model/Tool | Size | Purpose |
|-------|-----------|------|---------|
| Noise Suppression | noisereduce (Python) | minimal | Remove background noise |
| VAD | Silero VAD | ~2 MB | Detect speech segments |
| Language Detection | Silero LID | ~20 MB | Detect spoken language |
| STT (English) | Moonshine Tiny EN | ~27M params | English speech-to-text |
| STT (Chinese) | Moonshine Tiny ZH | ~27M params | Chinese speech-to-text |

### Routing Logic

```python
audio = noisereduce(audio)
segments = silero_vad(audio)
for chunk in segments:
    lang = detect_language(chunk)  # skipped if client specifies language
    if lang == "zh":
        text = moonshine_zh(chunk)
    else:
        text = moonshine_en(chunk)
```

### Expected Latency

| Stage | Latency |
|-------|---------|
| noisereduce | ~1-5 ms |
| VAD | ~1-5 ms |
| Language detection | ~10-30 ms |
| Moonshine STT | ~100-200 ms |
| **Total** | **~120-250 ms** |

## API Surface (identical to qwen3-asr)

| Endpoint | Method | Status |
|----------|--------|--------|
| `/health` | GET | Full implementation |
| `/v1/audio/transcriptions` | POST | Full implementation |
| `/v1/audio/transcriptions/stream` | POST | Full implementation (SSE) |
| `/v1/audio/subtitles` | POST | Stub — returns 501 |
| `/v1/audio/translations` | POST | Stub — returns 501 |
| `/ws/transcribe` | WebSocket | Full implementation |

### Schemas (match qwen3-asr exactly)

**HealthResponse:**
```json
{
  "status": "ok",
  "mode": "server",
  "model_loaded": true,
  "model_id": "moonshine-tiny-en + moonshine-tiny-zh",
  "cuda": true,
  "gpu_name": "NVIDIA ...",
  "gpu_allocated_mb": 128,
  "gpu_reserved_mb": 256
}
```

**TranscriptionResponse:**
```json
{"text": "transcribed text", "language": "en"}
```

**SSE events:**
```json
data: {"text": "chunk", "language": "en", "is_final": false, "chunk_index": 0}
data: {"done": true}
```

**WebSocket handshake:**
```json
{"status": "connected", "sample_rate": 16000, "format": "pcm_s16le", "buffer_size": 14400, "window_max_s": 6.0, "use_server_vad": true}
```

**WebSocket responses:**
```json
{"text": "partial", "is_partial": true, "is_final": false}
{"text": "final", "is_partial": false, "is_final": true}
```

**WebSocket commands:**
```json
{"action": "flush"}
{"action": "reset"}
{"action": "config", "language": "en", "use_server_vad": false}
```

**ErrorResponse:**
```json
{"code": "AUDIO_DECODE_FAILED", "message": "...", "statusCode": 422, "context": {}}
```

## Architecture

Single-process FastAPI server. No gateway/worker split. All models loaded at startup.

### Device Configuration (per-component)

| Component | Default | Env Var |
|-----------|---------|---------|
| noisereduce | cpu | `DENOISE_DEVICE` |
| Silero VAD | cpu | `VAD_DEVICE` |
| Silero LID | cpu | `LID_DEVICE` |
| Moonshine EN/ZH | auto (cuda if available) | `STT_DEVICE` |

### Concurrency

- Priority inference queue (WebSocket=0, HTTP=1), single inference worker
- AsyncIO for I/O (file upload, WebSocket, HTTP)
- Thread pool for blocking inference calls

### Project Structure

```
moonshine-asr/
├── src/
│   ├── server.py       # FastAPI app, all endpoints, WebSocket handler
│   ├── pipeline.py     # VoicePipeline: denoise -> VAD -> LID -> route -> STT
│   ├── models.py       # Model loading/management
│   ├── schemas.py      # Pydantic models (copied from qwen3-asr shape)
│   ├── config.py       # Env var validation
│   ├── logger.py       # Loguru structured logging
│   └── errors.py       # Error response utilities
├── Dockerfile
├── compose.yaml
├── .env.example
└── requirements.txt
```

## Configuration (.env)

```
# Models
MOONSHINE_EN_MODEL=<path or HF id>
MOONSHINE_ZH_MODEL=<path or HF id>

# Devices
STT_DEVICE=auto
VAD_DEVICE=cpu
LID_DEVICE=cpu
DENOISE_DEVICE=cpu

# Server
PORT=8000
LOG_LEVEL=info

# Pipeline
DENOISE_ENABLED=true
REQUEST_TIMEOUT=300
IDLE_TIMEOUT=0

# WebSocket
WS_BUFFER_SIZE=14400
WS_FLUSH_SILENCE_MS=600
WS_WINDOW_MAX_S=6.0
ASR_USE_SERVER_VAD=true

# SSE
SSE_CHUNK_SECONDS=5
SSE_OVERLAP_SECONDS=1
```

## Decisions

1. **noisereduce over RNNoise** — pure Python, GPU-capable, no C compilation needed
2. **Single process** — models are tiny (~27M params each), no need for gateway/worker isolation
3. **Per-component device control** — VAD/LID stay on CPU (tiny), STT uses GPU when available
4. **Language auto-detect + client override** — Silero LID per VAD segment by default, skip if client specifies language
5. **Stub endpoints for subtitles/translations** — return 501 so clients get meaningful error, not 404
