"""Model loading and management for the Moonshine ASR pipeline."""
from __future__ import annotations
import os
import gc
import time
import threading
import numpy as np
from logger import log

_model_rw_lock = threading.Lock()

_moonshine_en = None
_moonshine_zh = None
_models_loaded = False
_last_used = 0.0
_numpy_input = False  # detected at warmup: can moonshine accept numpy directly?

TARGET_SR = 16000


def _resolve_model(env_var: str, fallback_lang: str):
    """Resolve a Moonshine model path and architecture."""
    from moonshine_voice import get_model_for_language, ModelArch

    raw = os.getenv(env_var, "").strip()

    if raw == "" or raw.lower() in ("en", "zh", "english", "chinese"):
        lang = raw.lower() if raw.lower() in ("en", "zh") else fallback_lang
        log.bind(env_var=env_var, language=lang).info("model_resolve_default")
        model_path, model_arch = get_model_for_language(lang)
        return model_path, model_arch

    _, model_arch = get_model_for_language(fallback_lang)
    log.bind(env_var=env_var, path=raw).info("model_resolve_explicit")
    return raw, model_arch


def _prepare_audio(audio: np.ndarray) -> object:
    """Prepare audio input for moonshine. Uses numpy if supported, else .tolist()."""
    a = audio if audio.dtype == np.float32 else audio.astype(np.float32)
    return a if _numpy_input else a.tolist()


def load_all_models():
    """Load all pipeline models at startup."""
    global _moonshine_en, _moonshine_zh
    global _models_loaded, _last_used, _numpy_input

    with _model_rw_lock:
        if _models_loaded:
            return

        t_total = time.time()

        # -- STT_DEVICE note ------------------------------------------------
        stt_device_val = os.getenv("STT_DEVICE", "")
        if stt_device_val:
            log.bind(stt_device=stt_device_val).info(
                "stt_device_ignored_onnx_rt_selects_provider"
            )

        # -- Moonshine EN ---------------------------------------------------
        t0 = time.time()
        try:
            from moonshine_voice import Transcriber
            en_path, en_arch = _resolve_model("MOONSHINE_EN_MODEL", "en")
            _moonshine_en = Transcriber(en_path, en_arch)
            log.bind(path=en_path, elapsed_ms=round((time.time() - t0) * 1000)).info("model_loaded_en")
        except Exception as e:
            log.bind(error=str(e)).error("model_load_failed_en")
            raise

        # -- Moonshine ZH ---------------------------------------------------
        t0 = time.time()
        try:
            from moonshine_voice import Transcriber
            zh_path, zh_arch = _resolve_model("MOONSHINE_ZH_MODEL", "zh")
            _moonshine_zh = Transcriber(zh_path, zh_arch)
            log.bind(path=zh_path, elapsed_ms=round((time.time() - t0) * 1000)).info("model_loaded_zh")
        except Exception as e:
            log.bind(error=str(e)).error("model_load_failed_zh")
            raise

        # -- Warmup + detect numpy input support ----------------------------
        t0 = time.time()
        rng = np.random.default_rng(seed=42)
        dummy = rng.standard_normal(TARGET_SR).astype(np.float32) * 0.01
        try:
            # Try numpy array directly (avoids expensive .tolist() copy)
            _moonshine_en.transcribe_without_streaming(dummy, TARGET_SR)
            _moonshine_zh.transcribe_without_streaming(dummy, TARGET_SR)
            _numpy_input = True
        except (TypeError, ValueError, AttributeError):
            # Fall back to Python list
            _moonshine_en.transcribe_without_streaming(dummy.tolist(), TARGET_SR)
            _moonshine_zh.transcribe_without_streaming(dummy.tolist(), TARGET_SR)
            _numpy_input = False
        log.bind(elapsed_ms=round((time.time() - t0) * 1000), numpy_input=_numpy_input).info("model_warmup_complete")

        _models_loaded = True
        _last_used = time.time()
        total_ms = round((time.time() - t_total) * 1000)
        log.bind(total_elapsed_ms=total_ms).info("all_models_ready")


def unload_all_models():
    """Unload all models to free memory."""
    global _moonshine_en, _moonshine_zh
    global _models_loaded

    with _model_rw_lock:
        t0 = time.time()
        log.info("models_unloading")
        _moonshine_en = None
        _moonshine_zh = None
        _models_loaded = False
        gc.collect()
        log.bind(elapsed_ms=round((time.time() - t0) * 1000)).info("models_unloaded")


def is_loaded() -> bool:
    return _models_loaded


def subsystem_status() -> dict:
    """Return status of each model subsystem. Useful for health checks."""
    return {
        "stt_en": _moonshine_en is not None,
        "stt_zh": _moonshine_zh is not None,
    }


def get_last_used() -> float:
    return _last_used


def touch():
    global _last_used
    _last_used = time.time()


def transcribe_en(audio: np.ndarray, sr: int) -> str:
    """Transcribe audio using Moonshine English model."""
    if _moonshine_en is None:
        raise RuntimeError("Moonshine EN model not loaded")
    transcript = _moonshine_en.transcribe_without_streaming(_prepare_audio(audio), sr)
    return " ".join(line.text for line in transcript.lines).strip()


def transcribe_zh(audio: np.ndarray, sr: int) -> str:
    """Transcribe audio using Moonshine Chinese model."""
    if _moonshine_zh is None:
        raise RuntimeError("Moonshine ZH model not loaded")
    transcript = _moonshine_zh.transcribe_without_streaming(_prepare_audio(audio), sr)
    return " ".join(line.text for line in transcript.lines).strip()
