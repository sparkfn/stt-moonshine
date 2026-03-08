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


class WebSocketTranscript(BaseModel):
    """WebSocket transcription result (partial or final)."""
    text: str = Field(..., description="Transcribed text")
    is_partial: bool = Field(..., description="True if this is an interim result")
    is_final: bool = Field(..., description="True if this is the final result for this segment")


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
