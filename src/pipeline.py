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
        if detected_lang not in ("en", "zh"):
            detected_lang = "en"

    return text, detected_lang


def process_audio_vad(audio: np.ndarray, sr: int, lang_code: str | None = None) -> tuple[str, str]:
    """
    Full pipeline with VAD segmentation: denoise -> VAD -> chunk -> LID -> route -> STT.
    Used for file uploads where we want to process speech segments only.
    """
    import torch
    from silero_vad import get_speech_timestamps

    # Step 1: Denoise
    audio = denoise(audio, sr)

    # Step 2: VAD segmentation
    tensor = torch.from_numpy(audio)
    vad_model = models.get_vad_model()
    if vad_model is None:
        return process_audio(audio, sr, lang_code)

    try:
        timestamps = get_speech_timestamps(tensor, vad_model, sampling_rate=sr)
    except Exception:
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
