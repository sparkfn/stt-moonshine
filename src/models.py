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
_vad_model = None
_lid_classifier = None
_models_loaded = False
_last_used = 0.0
_numpy_input = False  # detected at warmup: can moonshine accept numpy directly?

TARGET_SR = 16000

# Pre-allocated empty array reused across StreamingVAD resets
_EMPTY_F32 = np.array([], dtype=np.float32)

# Lazy-loaded torch reference (set during model loading)
_torch = None


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
    global _vad_model, _lid_classifier
    global _models_loaded, _last_used, _numpy_input, _torch

    with _model_rw_lock:
        if _models_loaded:
            return

        import torch
        _torch = torch

        t_total = time.time()

        # -- STT_DEVICE note ------------------------------------------------
        stt_device_val = os.getenv("STT_DEVICE", "")
        if stt_device_val:
            log.bind(stt_device=stt_device_val).info(
                "stt_device_ignored_onnx_rt_selects_provider"
            )

        # -- Silero VAD -----------------------------------------------------
        t0 = time.time()
        try:
            from silero_vad import load_silero_vad
            _vad_model = load_silero_vad()
            _vad_model.eval()
            log.bind(elapsed_ms=round((time.time() - t0) * 1000)).info("model_loaded_vad")
        except Exception as e:
            log.bind(error=str(e)).error("model_load_failed_vad")

        # -- SpeechBrain LID (VoxLingua107 ECAPA-TDNN) ----------------------
        lid_device = os.getenv("LID_DEVICE", "cpu").lower()
        if lid_device == "auto":
            lid_device = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.time()
        try:
            from speechbrain.inference.classifiers import EncoderClassifier
            _lid_classifier = EncoderClassifier.from_hparams(
                source="speechbrain/lang-id-voxlingua107-ecapa",
                savedir=os.getenv("SPEECHBRAIN_CACHE", "/data/cache/speechbrain"),
                run_opts={"device": lid_device},
            )
            log.bind(device=lid_device, languages=107, elapsed_ms=round((time.time() - t0) * 1000)).info("model_loaded_lid")
        except Exception as e:
            log.bind(device=lid_device, error=str(e)).error("model_load_failed_lid")
            _lid_classifier = None

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
        gpu_mb = 0
        if torch.cuda.is_available():
            gpu_mb = round(torch.cuda.memory_allocated(0) / 1024 / 1024)
        log.bind(total_elapsed_ms=total_ms, gpu_allocated_mb=gpu_mb).info("all_models_ready")


def unload_all_models():
    """Unload all models to free memory."""
    global _moonshine_en, _moonshine_zh, _vad_model, _lid_classifier
    global _models_loaded

    with _model_rw_lock:
        t0 = time.time()
        log.info("models_unloading")
        _moonshine_en = None
        _moonshine_zh = None
        _vad_model = None
        _lid_classifier = None
        _models_loaded = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        log.bind(elapsed_ms=round((time.time() - t0) * 1000)).info("models_unloaded")


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
    """Check if audio contains speech using Silero VAD.

    Silero expects exactly VAD_FRAME_SAMPLES (512) samples at 16kHz.
    We check multiple frames and return True if any frame exceeds threshold.
    """
    if _vad_model is None or _torch is None:
        return True
    if len(audio_float32) < VAD_FRAME_SAMPLES:
        return True
    try:
        # Reset hidden state — is_speech is called on independent audio clips,
        # not a continuous stream, so stale RNN state would corrupt results
        _vad_model.reset_states()
        # Check up to 4 evenly-spaced frames across the audio
        n = len(audio_float32)
        num_frames = min(4, n // VAD_FRAME_SAMPLES)
        step = max(1, (n - VAD_FRAME_SAMPLES) // max(1, num_frames - 1))
        frame_buf = _torch.zeros(1, VAD_FRAME_SAMPLES)
        for i in range(num_frames):
            offset = min(i * step, n - VAD_FRAME_SAMPLES)
            frame_buf[0].copy_(_torch.from_numpy(
                audio_float32[offset:offset + VAD_FRAME_SAMPLES]
            ))
            with _torch.inference_mode():
                if _vad_model(frame_buf, TARGET_SR).item() >= threshold:
                    return True
        return False
    except Exception as e:
        log.bind(error=str(e), samples=len(audio_float32)).warning("vad_is_speech_error")
        return True


def vad_confidence(audio_float32: np.ndarray) -> float:
    """Return max Silero VAD confidence across sampled frames. Returns 1.0 if VAD unavailable."""
    if _vad_model is None or _torch is None:
        return 1.0
    if len(audio_float32) < VAD_FRAME_SAMPLES:
        return 1.0
    try:
        _vad_model.reset_states()
        # Check last frame for a quick confidence estimate
        frame = audio_float32[-VAD_FRAME_SAMPLES:]
        frame_buf = _torch.zeros(1, VAD_FRAME_SAMPLES)
        frame_buf[0].copy_(_torch.from_numpy(frame))
        with _torch.inference_mode():
            return _vad_model(frame_buf, TARGET_SR).item()
    except Exception as e:
        log.bind(error=str(e), samples=len(audio_float32)).warning("vad_confidence_error")
        return 1.0


def reset_vad_state() -> None:
    """Reset Silero VAD internal hidden state (call at start of new audio stream)."""
    if _vad_model is not None:
        try:
            _vad_model.reset_states()
        except Exception as e:
            log.bind(error=str(e)).warning("vad_reset_error")


# Silero VAD expects 512 samples at 16kHz (32ms per frame)
VAD_FRAME_SAMPLES = 512


class StreamingVAD:
    """Streaming VAD using Silero V5 algorithm (matches ricky0123/vad).

    Processes audio in 512-sample frames (32ms at 16kHz). Uses dual thresholds:
    - positiveSpeechThreshold: probability to START speech (default 0.3)
    - negativeSpeechThreshold: probability to consider silence (default 0.25)
    - redemptionMs: ms of silence before ending speech (default 600 for telephony)
    - minSpeechMs: minimum speech duration to emit (default 250)
    """

    def __init__(
        self,
        positive_threshold: float = 0.3,
        negative_threshold: float = 0.25,
        redemption_ms: int = 600,
        min_speech_ms: int = 250,
    ):
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self._frame_ms = VAD_FRAME_SAMPLES / TARGET_SR * 1000  # 32ms
        self._redemption_frames = int(redemption_ms / self._frame_ms)
        self._min_speech_frames = int(min_speech_ms / self._frame_ms)
        self._speech_active = False
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._leftover = _EMPTY_F32
        self._total_frames = 0
        # Pre-allocated tensor buffer for single-frame inference (avoids per-frame allocation)
        self._frame_tensor = None  # lazily created when _torch is available
        # Per-instance VAD model to avoid shared hidden state across sessions
        self._vad = None
        try:
            from silero_vad import load_silero_vad
            self._vad = load_silero_vad()
            self._vad.eval()
        except Exception as e:
            log.bind(error=str(e)).warning("streaming_vad_load_failed_using_global")

    def reset(self) -> None:
        """Reset VAD state for a new utterance."""
        self._speech_active = False
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._leftover = _EMPTY_F32
        self._total_frames = 0
        if self._vad is not None:
            try:
                self._vad.reset_states()
            except Exception:
                pass
        else:
            reset_vad_state()

    def process(self, audio_float32: np.ndarray) -> str | None:
        """Feed audio through VAD frame-by-frame.

        Note: Only the *last* state transition is returned. Callers should feed
        small chunks (~3200 samples / 200ms) to ensure at most one transition
        per call. The WebSocket path already satisfies this constraint.

        Returns:
            "speech_start" — speech detected (above positive threshold)
            "speech_end"   — silence after speech exceeded redemption period
            None           — no state transition
        """
        vad = self._vad or _vad_model
        if vad is None or _torch is None:
            return None

        # Lazily create reusable tensor buffer
        if self._frame_tensor is None:
            self._frame_tensor = _torch.zeros(1, VAD_FRAME_SAMPLES)

        # Prepend leftover from previous call
        if len(self._leftover) > 0:
            audio_float32 = np.concatenate([self._leftover, audio_float32])
            self._leftover = _EMPTY_F32

        result = None
        offset = 0
        n = len(audio_float32)
        pos_thresh = self.positive_threshold
        neg_thresh = self.negative_threshold
        frame_buf = self._frame_tensor

        while offset + VAD_FRAME_SAMPLES <= n:
            # Copy frame data into pre-allocated tensor (avoids from_numpy + unsqueeze per frame)
            frame_buf[0].copy_(_torch.from_numpy(audio_float32[offset:offset + VAD_FRAME_SAMPLES]))
            offset += VAD_FRAME_SAMPLES
            self._total_frames += 1

            try:
                with _torch.inference_mode():
                    prob = vad(frame_buf, TARGET_SR).item()
            except Exception:
                continue

            if not self._speech_active:
                if prob >= pos_thresh:
                    self._speech_active = True
                    self._speech_frame_count = 1
                    self._silence_frame_count = 0
                    result = "speech_start"
            else:
                if prob >= pos_thresh:
                    self._speech_frame_count += 1
                    self._silence_frame_count = 0
                elif prob < neg_thresh:
                    self._silence_frame_count += 1
                    if self._silence_frame_count >= self._redemption_frames:
                        if self._speech_frame_count >= self._min_speech_frames:
                            result = "speech_end"
                        self._speech_active = False
                        self._speech_frame_count = 0
                        self._silence_frame_count = 0

        # Save remaining samples for next call
        if offset < n:
            self._leftover = audio_float32[offset:]

        return result

    @property
    def speech_active(self) -> bool:
        return self._speech_active


def detect_language(audio_float32: np.ndarray) -> str:
    """Detect language using SpeechBrain VoxLingua107. Returns 'en' or 'zh'."""
    if _lid_classifier is None:
        return "en"
    try:
        tensor = _torch.from_numpy(audio_float32).unsqueeze(0)
        out_prob, score, index, text_lab = _lid_classifier.classify_batch(tensor)
        label = text_lab[0] if text_lab else ""
        lang_code = label.split(":")[0].strip().lower() if ":" in label else label.strip().lower()
        log.bind(label=label, score=round(score.item(), 3)).debug("lid_result")
        return "zh" if lang_code == "zh" else "en"
    except Exception as e:
        log.bind(error=str(e)).debug("lid_failed_default_en")
        return "en"


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
