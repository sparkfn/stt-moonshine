"""Model loading and management for the Moonshine ASR pipeline."""
from __future__ import annotations
import os
import gc
import time
import numpy as np
from logger import log

_moonshine_en = None
_moonshine_zh = None
_vad_model = None
_lid_classifier = None
_models_loaded = False
_last_used = 0.0

TARGET_SR = 16000


def _resolve_model(env_var: str, fallback_lang: str):
    """Resolve a Moonshine model path and architecture.

    If the env var is unset, empty, or looks like a bare language code (e.g.
    "en", "zh"), fall back to ``get_model_for_language()``.  Otherwise treat
    the value as an explicit *model_path* and pair it with the default arch
    returned by ``get_model_for_language()`` for the fallback language.
    """
    from moonshine_voice import get_model_for_language, ModelArch

    raw = os.getenv(env_var, "").strip()

    # Bare language codes -- just use the downloader
    if raw == "" or raw.lower() in ("en", "zh", "english", "chinese"):
        lang = raw.lower() if raw.lower() in ("en", "zh") else fallback_lang
        log.info("Env {} not set or is a language code; downloading default model for '{}'", env_var, lang)
        model_path, model_arch = get_model_for_language(lang)
        return model_path, model_arch

    # Explicit path -- still need an arch; derive from the fallback language
    _, model_arch = get_model_for_language(fallback_lang)
    log.info("Using explicit model path from {}: {}", env_var, raw)
    return raw, model_arch


def load_all_models():
    """Load all pipeline models at startup."""
    global _moonshine_en, _moonshine_zh
    global _vad_model, _lid_classifier
    global _models_loaded, _last_used

    if _models_loaded:
        return

    t_load_start = time.time()

    # -- STT_DEVICE note ------------------------------------------------
    stt_device_val = os.getenv("STT_DEVICE", "")
    if stt_device_val:
        log.info(
            "STT_DEVICE='{}' is set but moonshine-voice uses ONNX Runtime "
            "which selects the execution provider automatically. "
            "This env var is kept for documentation only.",
            stt_device_val,
        )

    # -- Silero VAD -----------------------------------------------------
    log.info("Loading Silero VAD...")
    try:
        from silero_vad import load_silero_vad
        _vad_model = load_silero_vad()
        _vad_model.eval()
        log.info("Silero VAD loaded")
    except Exception as e:
        log.error("Failed to load Silero VAD: {}", e)

    # -- SpeechBrain LID (VoxLingua107 ECAPA-TDNN) ----------------------
    lid_device = os.getenv("LID_DEVICE", "cpu").lower()
    if lid_device == "auto":
        import torch as _torch
        lid_device = "cuda" if _torch.cuda.is_available() else "cpu"
    log.info("Loading SpeechBrain LID on {}...", lid_device)
    try:
        from speechbrain.inference.classifiers import EncoderClassifier
        _lid_classifier = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            savedir=os.getenv("SPEECHBRAIN_CACHE", "/data/cache/speechbrain"),
            run_opts={"device": lid_device},
        )
        log.info("SpeechBrain LID loaded (107 languages)")
    except Exception as e:
        log.error("Failed to load SpeechBrain LID: {}", e)
        _lid_classifier = None

    # -- Moonshine EN ---------------------------------------------------
    log.info("Loading Moonshine EN transcriber...")
    try:
        from moonshine_voice import Transcriber
        en_path, en_arch = _resolve_model("MOONSHINE_EN_MODEL", "en")
        _moonshine_en = Transcriber(en_path, en_arch)
        log.info("Moonshine EN loaded (path={})", en_path)
    except Exception as e:
        log.error("Failed to load Moonshine EN: {}", e)
        raise

    # -- Moonshine ZH ---------------------------------------------------
    log.info("Loading Moonshine ZH transcriber...")
    try:
        from moonshine_voice import Transcriber
        zh_path, zh_arch = _resolve_model("MOONSHINE_ZH_MODEL", "zh")
        _moonshine_zh = Transcriber(zh_path, zh_arch)
        log.info("Moonshine ZH loaded (path={})", zh_path)
    except Exception as e:
        log.error("Failed to load Moonshine ZH: {}", e)
        raise

    # -- Warmup ---------------------------------------------------------
    log.info("Warming up transcribers...")
    rng = np.random.default_rng(seed=42)
    dummy = rng.standard_normal(TARGET_SR).astype(np.float32) * 0.01
    try:
        transcribe_en(dummy, TARGET_SR)
        transcribe_zh(dummy, TARGET_SR)
    except Exception:
        pass

    _models_loaded = True
    _last_used = time.time()
    elapsed_ms = round((time.time() - t_load_start) * 1000)
    import torch
    gpu_mb = 0
    if torch.cuda.is_available():
        gpu_mb = round(torch.cuda.memory_allocated(0) / 1024 / 1024)
    log.bind(elapsed_ms=elapsed_ms, gpu_allocated_mb=gpu_mb).info("models_loaded")


def unload_all_models():
    """Unload all models to free memory."""
    global _moonshine_en, _moonshine_zh, _vad_model, _lid_classifier
    global _models_loaded

    log.info("Unloading all models (idle timeout)...")
    _moonshine_en = None
    _moonshine_zh = None
    _vad_model = None
    _lid_classifier = None
    _models_loaded = False
    gc.collect()


def is_loaded() -> bool:
    return _models_loaded


def get_last_used() -> float:
    return _last_used


def touch():
    global _last_used
    _last_used = time.time()


def get_vad_model():
    return _vad_model


def is_speech(audio_float32: np.ndarray, threshold: float = 0.5) -> bool:
    """Check if audio contains speech using Silero VAD."""
    if _vad_model is None:
        return True
    try:
        import torch
        tensor = torch.from_numpy(audio_float32).unsqueeze(0)
        with torch.no_grad():
            confidence = _vad_model(tensor, TARGET_SR).item()
        return confidence >= threshold
    except Exception:
        return True


def detect_language(audio_float32: np.ndarray) -> str:
    """Detect language using SpeechBrain VoxLingua107.

    Returns ``'en'`` or ``'zh'``.
    """
    if _lid_classifier is None:
        return "en"
    try:
        import torch
        tensor = torch.from_numpy(audio_float32).unsqueeze(0)
        out_prob, score, index, text_lab = _lid_classifier.classify_batch(tensor)
        # text_lab is e.g. ['zh: Chinese'] or ['en: English']
        label = text_lab[0] if text_lab else ""
        lang_code = label.split(":")[0].strip().lower() if ":" in label else label.strip().lower()
        log.debug("LID result: {} (score={:.3f})", label, score.item())
        return "zh" if lang_code == "zh" else "en"
    except Exception as e:
        log.debug("LID failed, defaulting to en: {}", e)
        return "en"


def transcribe_en(audio: np.ndarray, sr: int) -> str:
    """Transcribe audio using Moonshine English model."""
    if _moonshine_en is None:
        raise RuntimeError("Moonshine EN model not loaded")
    audio_f32 = audio.astype(np.float32) if audio.dtype != np.float32 else audio
    transcript = _moonshine_en.transcribe_without_streaming(audio_f32.tolist(), sr)
    return " ".join(line.text for line in transcript.lines).strip()


def transcribe_zh(audio: np.ndarray, sr: int) -> str:
    """Transcribe audio using Moonshine Chinese model."""
    if _moonshine_zh is None:
        raise RuntimeError("Moonshine ZH model not loaded")
    audio_f32 = audio.astype(np.float32) if audio.dtype != np.float32 else audio
    transcript = _moonshine_zh.transcribe_without_streaming(audio_f32.tolist(), sr)
    return " ".join(line.text for line in transcript.lines).strip()
