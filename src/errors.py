from fastapi.responses import JSONResponse
from logger import log, get_request_id


# OpenAI error type categories
INVALID_REQUEST = "invalid_request_error"
SERVER_ERROR = "server_error"
NOT_FOUND = "not_found_error"

# Map HTTP status ranges to OpenAI error types
_STATUS_TO_TYPE = {
    400: INVALID_REQUEST,
    413: INVALID_REQUEST,
    422: INVALID_REQUEST,
    404: NOT_FOUND,
    501: SERVER_ERROR,
    500: SERVER_ERROR,
    504: SERVER_ERROR,
}


def error_response(
    code: str,
    message: str,
    status_code: int,
    param: str | None = None,
    error_type: str | None = None,
    **log_context,
) -> JSONResponse:
    """Build an OpenAI-compatible JSON error response and log it.

    Response shape:
        {"error": {"message": "...", "type": "...", "param": ..., "code": "..."}}

    Status code is conveyed via HTTP status only, not in the body.
    """
    resolved_type = error_type or _STATUS_TO_TYPE.get(status_code, SERVER_ERROR)

    body = {
        "error": {
            "message": message,
            "type": resolved_type,
            "param": param,
            "code": code,
        }
    }

    if status_code >= 500:
        log.bind(code=code, type=resolved_type, status=status_code, **log_context).error("error_response")
    else:
        log.bind(code=code, type=resolved_type, status=status_code, **log_context).warning("error_response")

    return JSONResponse(status_code=status_code, content=body)


def ws_error(code: str, message: str, error_type: str = SERVER_ERROR) -> dict:
    """Build an OpenAI-shaped error dict for WebSocket responses.

    Same shape as HTTP errors but sent as JSON over the WebSocket.
    """
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code,
        }
    }
