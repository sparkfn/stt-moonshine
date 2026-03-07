from __future__ import annotations
import contextvars
import logging
import os
import sys

try:
    import orjson

    def _dumps(obj: dict) -> bytes:
        return orjson.dumps(obj, default=str)
except ImportError:
    import json

    def _dumps(obj: dict) -> bytes:
        return json.dumps(obj, default=str).encode()

from loguru import logger

_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


def set_request_id(req_id: str) -> contextvars.Token:
    return _request_id_var.set(req_id)


def reset_request_id(token: contextvars.Token) -> None:
    _request_id_var.reset(token)


def get_request_id() -> str | None:
    return _request_id_var.get()


_LEVEL_MAP = {
    "critical": "fatal",
    "warning": "warn",
}

_NL = b"\n"
_stdout_buf = sys.stdout.buffer


def _json_sink(message: "loguru.Message") -> None:
    record = message.record
    raw_level = record["level"].name.lower()
    entry: dict = {
        "timestamp": record["time"].isoformat(),
        "level": _LEVEL_MAP.get(raw_level, raw_level),
        "message": record["message"],
        "service": "moonshine-asr",
    }
    req_id = _request_id_var.get()
    if req_id:
        entry["requestId"] = req_id
    extra = record.get("extra", {})
    if extra:
        entry.update(extra)
    exc = record["exception"]
    if exc:
        entry["err"] = str(exc.value)
    _stdout_buf.write(_dumps(entry) + _NL)
    _stdout_buf.flush()


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            if frame.f_back:
                frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


_STDLIB_LEVEL_MAP = {"TRACE": "DEBUG"}


def setup_logger() -> "loguru.Logger":
    log_level = os.getenv("LOG_LEVEL", "info").upper()
    logging.root.handlers = [InterceptHandler()]
    stdlib_level = _STDLIB_LEVEL_MAP.get(log_level, log_level)
    logging.root.setLevel(stdlib_level)
    logger.remove()
    logger.add(_json_sink, level=log_level, format="{message}", catch=False)
    for name in logging.root.manager.loggerDict.keys():
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True
    for name in ["uvicorn.access", "uvicorn.error", "uvicorn"]:
        logging_logger = logging.getLogger(name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False
    return logger


log = setup_logger()
