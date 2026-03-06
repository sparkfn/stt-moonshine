# Moonshine-ASR Docker Server

A Dockerized REST API server for [Moonshine](https://github.com/moonshine-ai/moonshine-voice) (Automatic Speech Recognition) with GPU acceleration and OpenAI-compatible endpoints.

Drop-in replacement for [qwen3-asr](https://github.com/jaaacki/qwen3-asr) — same API surface, same WebSocket protocol, same request/response schemas. Swap the URL and everything works.

## Features

### Speech-to-Text
- **OpenAI-compatible transcription** — `POST /v1/audio/transcriptions` accepts WAV, MP3, FLAC, OGG, and more
- **SSE streaming** — `POST /v1/audio/transcriptions/stream` returns chunked results as Server-Sent Events
- **Real-time WebSocket** — `WS /ws/transcribe` accepts raw PCM audio for live transcription with sliding window (up to 6s context) and cumulative partials
- **Dual-language** — English and Chinese via language-routed Moonshine ONNX models

### Voice Pipeline
```
audio -> noisereduce -> silero_vad -> speech_chunk -> language_detection -> router -> moonshine_en / moonshine_zh -> transcript
```

| Component | Model | Purpose |
|-----------|-------|---------|
| Noise suppression | noisereduce | Stationary noise reduction |
| Voice activity | Silero VAD v6 | Speech/silence detection, auto-flush |
| Language ID | SpeechBrain VoxLingua107 (ECAPA-TDNN) | 107-language classifier, routes to en/zh |
| STT (English) | Moonshine medium-streaming-en (quantized) | ONNX Runtime inference |
| STT (Chinese) | Moonshine base-zh (quantized) | ONNX Runtime inference |

### Real-Time WebSocket
- **Sliding window** — re-transcribes up to 6s of accumulated audio each trigger for full context
- **Server-side VAD** — Silero VAD auto-flushes on speech-to-silence transitions; skips inference for silence
- **Per-connection VAD toggle** — clients can disable via query param (`?use_server_vad=false`) or mid-session config action
- **Silence padding** — 600ms silence appended on flush to commit trailing words
- **Control commands** — `flush`, `reset`, `config` (set language, toggle VAD)

### Performance
- **Priority scheduling** — WebSocket requests (priority 0) preempt HTTP uploads (priority 1) via min-heap queue
- **On-demand model loading** — models load on first request, unload after idle timeout
- **ONNX Runtime** — Moonshine uses ONNX backend, automatically selects CUDA/CPU execution provider
- **Repetition detection** — post-processing removes repeated words/phrases from transcriptions

### Observability
- **Structured JSON logging** — loguru-based with configurable `LOG_LEVEL`
- **Request tracing** — `X-Request-ID` header on every request
- **Swagger UI** — interactive API docs at `/docs`

## Quick Start

```bash
# Clone and start
git clone https://github.com/jaaacki/moonshine-asr.git
cd moonshine-asr
cp .env.example .env   # Adjust settings as needed
docker compose up -d

# Check health
curl http://localhost:8200/health

# Transcribe audio file
curl -X POST http://localhost:8200/v1/audio/transcriptions \
  -F "file=@audio.wav"

# Transcribe with explicit language
curl -X POST http://localhost:8200/v1/audio/transcriptions \
  -F "file=@audio.wav" -F "language=English"

# Streaming transcription (SSE)
curl -X POST http://localhost:8200/v1/audio/transcriptions/stream \
  -F "file=@audio.wav" -N
```

### WebSocket Example (Python)

```python
import asyncio, json, websockets, numpy as np

async def stream():
    async with websockets.connect("ws://localhost:8200/ws/transcribe") as ws:
        info = json.loads(await ws.recv())
        print("Connected:", info)

        # Send PCM audio (16-bit LE, 16kHz mono)
        audio = np.zeros(16000, dtype=np.int16)  # 1s silence
        await ws.send(audio.tobytes())

        # Flush to get final result
        await ws.send(json.dumps({"action": "flush"}))
        result = json.loads(await ws.recv())
        print("Result:", result)

asyncio.run(stream())
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check with GPU info and model status |
| `POST` | `/v1/audio/transcriptions` | Transcribe uploaded audio file |
| `POST` | `/v1/audio/transcriptions/stream` | SSE streaming transcription |
| `POST` | `/v1/audio/translations` | Stub (returns 501) |
| `POST` | `/v1/audio/subtitles` | Stub (returns 501) |
| `WS` | `/ws/transcribe` | Real-time WebSocket streaming |

### WebSocket Protocol

**Connect:** `ws://host:port/ws/transcribe`

Query params: `sample_rate` (8000 or 16000), `use_server_vad` (true/false), `request_id`

**Send:**
- Binary frames: raw PCM audio (16-bit LE, mono)
- JSON commands: `{"action": "flush"}`, `{"action": "reset"}`, `{"action": "config", "language": "en"}`

**Receive:**
- Handshake: `{"status": "connected", "sample_rate": 16000, "format": "pcm_s16le", "buffer_size": ..., "window_max_s": 6.0}`
- Partial: `{"text": "...", "is_partial": true, "is_final": false}`
- Final: `{"text": "...", "is_partial": false, "is_final": true}`
- Reset ack: `{"status": "buffer_reset"}`
- Config ack: `{"status": "configured", "language": "en", "use_server_vad": true}`

## Configuration

Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8200` | Host port mapping |
| `MOONSHINE_EN_MODEL` | `en` | English model: language code to auto-download, or explicit path |
| `MOONSHINE_ZH_MODEL` | `zh` | Chinese model: language code to auto-download, or explicit path |
| `STT_DEVICE` | `auto` | ONNX Runtime device (auto/cpu/cuda) — informational only |
| `VAD_DEVICE` | `cpu` | Silero VAD device |
| `LID_DEVICE` | `cpu` | SpeechBrain LID device (cpu/cuda/auto) |
| `DENOISE_DEVICE` | `cpu` | Noise reduction device |
| `DENOISE_ENABLED` | `true` | Enable/disable noise reduction |
| `REQUEST_TIMEOUT` | `300` | Max seconds per inference request |
| `IDLE_TIMEOUT` | `0` | Seconds before unloading idle models (0 = never) |
| `LOG_LEVEL` | `info` | Logging level (trace/debug/info/warning/error) |
| `WS_WINDOW_MAX_S` | `6.0` | WebSocket sliding window max duration (seconds) |
| `ASR_USE_SERVER_VAD` | `true` | Default server-side VAD for WebSocket connections |
| `SSE_CHUNK_SECONDS` | `5` | SSE chunk duration for long audio |
| `SSE_OVERLAP_SECONDS` | `1` | SSE chunk overlap for context continuity |

## Project Structure

```
moonshine-asr/
  Dockerfile          # pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
  compose.yaml        # Docker Compose with GPU reservation
  requirements.txt    # Python dependencies
  .env.example        # Configuration template
  src/
    server.py         # FastAPI app, all endpoints, WebSocket handler, priority queue
    models.py         # Model loading: Silero VAD, SpeechBrain LID, Moonshine EN/ZH
    pipeline.py       # Voice pipeline: denoise -> VAD -> LID -> route -> STT
    schemas.py        # Pydantic request/response models
    config.py         # Environment variable validation
    logger.py         # Loguru structured JSON logging
    errors.py         # Error response helper
  models/             # HuggingFace cache (auto-populated on first run)
```

## Differences from qwen3-asr

| Feature | qwen3-asr | moonshine-asr |
|---------|-----------|---------------|
| STT model | Qwen3-ASR-1.7B (PyTorch) | Moonshine EN + ZH (ONNX Runtime) |
| Language detection | Built-in (Qwen3) | SpeechBrain VoxLingua107 |
| Languages | en, zh, ja, yue, hi, th | en, zh (others routed to en) |
| Subtitles | Full SRT generation | Stub (501) |
| Translation | LLM-backed translation | Stub (501) |
| Quantization | INT8/FP8/ONNX/TRT | ONNX quantized by default |
| VRAM usage | ~3-6 GB | ~1-2 GB |
| API compatibility | Reference | Identical endpoints and schemas |

## Requirements

- Docker with NVIDIA Container Toolkit (for GPU)
- NVIDIA GPU with CUDA 12.4+ (or CPU-only with reduced performance)
- ~4 GB disk for model downloads on first run

## License

MIT
