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
