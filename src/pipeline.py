"""Voice pipeline: denoise -> VAD -> LID -> route -> STT."""
from __future__ import annotations
import os
import time
import numpy as np
from logger import log
import models

TARGET_SR = 16000
DENOISE_ENABLED = os.getenv("DENOISE_ENABLED", "true").lower() == "true"

# Pre-import noisereduce at module level (avoids per-call import overhead)
_nr = None
if DENOISE_ENABLED:
    try:
        import noisereduce as _nr
    except ImportError:
        pass


def denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply noise reduction if enabled."""
    if _nr is None:
        return audio
    try:
        return _nr.reduce_noise(y=audio, sr=sr, stationary=True)
    except Exception as e:
        log.bind(error=str(e)).warning("denoise_failed_using_original")
        return audio


def process_audio(
    audio: np.ndarray,
    sr: int,
    lang_code: str | None = None,
    skip_denoise: bool = False,
) -> tuple[str, str]:
    """
    Full pipeline: denoise -> transcribe with language routing.

    Args:
        audio: float32 audio array
        sr: sample rate
        lang_code: language override (None = auto-detect)
        skip_denoise: skip noisereduce (e.g. WS path already bandpass-filtered)

    Returns:
        (transcribed_text, detected_language)
    """
    t0 = time.time()

    # Step 1: Denoise (skip for pre-filtered audio)
    if skip_denoise:
        log.debug("denoise_skipped_bandpass")
    else:
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

    elapsed_ms = round((time.time() - t0) * 1000)
    log.bind(
        lang=detected_lang,
        audio_s=round(len(audio) / sr, 2),
        text_len=len(text),
        elapsed_ms=elapsed_ms,
    ).debug("pipeline_transcribed")

    return text, detected_lang
