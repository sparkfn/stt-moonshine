# E2E Test Suite for Moonshine-ASR

End-to-end tests for the Moonshine-ASR speech-to-text server.

## Quick Start

```bash
# Install test dependencies
pip install -r E2Etest/requirements.txt

# Run all tests (requires server running on localhost:8200)
pytest E2Etest/ -v

# Or use the helper script
./E2Etest/run_tests.sh
```

## Test Structure

```
E2Etest/
├── conftest.py              # Pytest configuration and fixtures
├── requirements.txt         # Test dependencies
├── test_api_http.py         # HTTP API tests
├── test_websocket.py        # WebSocket streaming tests
├── test_performance.py      # Latency and throughput tests
├── test_integration.py      # Multi-feature integration tests
├── test_accuracy.py         # Transcription quality tests
├── utils/
│   ├── client.py            # HTTP/WebSocket client wrappers
│   └── audio.py             # Audio generation helpers
└── data/
    ├── audio/               # Test audio files
    └── expected/            # Expected transcripts (optional)
```

## Running Tests

### With Server Running

```bash
# Start server if not already running
docker compose up -d

# Run smoke tests
pytest E2Etest/ -v -m smoke

# Run HTTP tests only
pytest E2Etest/test_api_http.py -v

# Run WebSocket tests only
pytest E2Etest/test_websocket.py -v

# Run performance tests
pytest E2Etest/test_performance.py -v --benchmark

# Run without slow tests
pytest E2Etest/ -v -m "not slow"

# Run with server management (start/stop container)
./E2Etest/run_tests.sh --with-server
```

### Environment Variables

```bash
# Custom server URL
E2E_BASE_URL=http://localhost:8200 pytest E2Etest/

# Custom WebSocket URL
E2E_WS_URL=ws://localhost:8200/ws/transcribe pytest E2Etest/

# Run with API key (if authentication enabled)
E2E_API_KEY=your_key pytest E2Etest/
```

## Test Categories

### Smoke Tests (`-m smoke`)
- Health endpoint returns 200
- Basic transcription works
- Model loads on first request

### HTTP Tests (`test_api_http.py`)
- Health check with GPU info
- File upload transcription
- Streaming transcription (SSE)
- Error handling (empty file, invalid format)

### WebSocket Tests (`test_websocket.py`)
- Connection and handshake
- Audio streaming and partial transcriptions
- Control commands (flush, reset, config)
- Auto-reset on audio gap
- Duplicate suppression
- Buffer-only mode (`WS_PARTIAL=false`)

### Performance Tests (`test_performance.py`)
- Cold start latency
- Warm inference latency
- Throughput benchmarks
- GPU memory stability

### Integration Tests (`test_integration.py`)
- Priority queue (WS vs HTTP)
- Dual model (EN + ZH) routing
- Idle timeout behavior
- Feature combinations

### Accuracy Tests (`test_accuracy.py`)
- Word Error Rate (if reference data available)
- Repetition detection
- Output format validation

## Test Audio

Test audio files (sine tones, speech-like signals, noise) are **auto-generated** on first run via a session-scoped autouse fixture. No manual setup needed. Generated files are gitignored.

To regenerate manually: `python -c "import sys; sys.path.insert(0, 'E2Etest'); from utils.audio import generate_test_audio_files; generate_test_audio_files('E2Etest/data/audio', force=True)"`

## Troubleshooting

### Tests skipped
- Ensure server is running: `docker compose up -d`
- Check health endpoint: `curl http://localhost:8200/health`
- Verify port mapping in `compose.yaml`

### Model not loading
- First transcription request triggers model load
- Can take 30-120s depending on model size
- Tests with `ensure_model_loaded` fixture wait for load

### WebSocket tests failing
- Check server supports WebSocket: `/ws/transcribe`
- Verify no proxy/firewall blocking WebSocket
