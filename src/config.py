"""Startup configuration validation."""
import os
import sys
from logger import log


def _safe_float(name: str, default: str) -> float:
    raw = os.getenv(name, default)
    try:
        return float(raw)
    except ValueError:
        log.bind(key=name, value=raw, default=default).error("config_invalid_float")
        return float(default)


def _safe_int(name: str, default: str) -> int:
    raw = os.getenv(name, default)
    try:
        return int(raw)
    except ValueError:
        log.bind(key=name, value=raw, default=default).error("config_invalid_int")
        return int(default)


SSE_CHUNK_SECONDS = _safe_int("SSE_CHUNK_SECONDS", "5")
SSE_OVERLAP_SECONDS = _safe_int("SSE_OVERLAP_SECONDS", "1")

_VALID_LOG_LEVELS = {"TRACE", "DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL", "FATAL"}
_LOG_LEVEL_ALIASES = {"WARN": "WARNING", "FATAL": "CRITICAL"}


def validate_env() -> None:
    errors = []

    en_model = os.getenv("MOONSHINE_EN_MODEL", "")
    if not en_model:
        errors.append("MOONSHINE_EN_MODEL is required but empty or unset")

    zh_model = os.getenv("MOONSHINE_ZH_MODEL", "")
    if not zh_model:
        errors.append("MOONSHINE_ZH_MODEL is required but empty or unset")

    try:
        rt = int(os.getenv("REQUEST_TIMEOUT", "300"))
        if rt <= 0:
            errors.append(f"REQUEST_TIMEOUT must be positive, got {rt}")
    except ValueError as e:
        errors.append(f"REQUEST_TIMEOUT must be an integer: {e}")

    try:
        it = int(os.getenv("IDLE_TIMEOUT", "0"))
        if it < 0:
            errors.append(f"IDLE_TIMEOUT must be non-negative, got {it}")
    except ValueError as e:
        errors.append(f"IDLE_TIMEOUT must be an integer: {e}")

    log_level = os.getenv("LOG_LEVEL", "info").upper()
    log_level = _LOG_LEVEL_ALIASES.get(log_level, log_level)
    if log_level not in _VALID_LOG_LEVELS:
        errors.append(f"LOG_LEVEL must be one of {_VALID_LOG_LEVELS}, got '{log_level}'")

    stt_device = os.getenv("STT_DEVICE", "auto").lower()
    if stt_device not in ("auto", "cpu", "cuda"):
        errors.append(f"STT_DEVICE must be auto/cpu/cuda, got '{stt_device}'")

    try:
        ws = float(os.getenv("WS_WINDOW_MAX_S", "6.0"))
        if ws <= 0:
            errors.append(f"WS_WINDOW_MAX_S must be positive, got {ws}")
    except ValueError as e:
        errors.append(f"WS_WINDOW_MAX_S must be a float: {e}")

    if errors:
        for err in errors:
            log.bind(detail=err).error("config_validation_failed")
        sys.exit(1)

    log.info("config_validated")
