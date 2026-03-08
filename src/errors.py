from fastapi.responses import JSONResponse
from logger import log, get_request_id


def error_response(code: str, message: str, status_code: int, **context) -> JSONResponse:
    """Build a JSON error response and log it automatically."""
    req_id = get_request_id()
    ctx = dict(context) if context else {}
    if req_id:
        ctx["requestId"] = req_id
    body = {
        "code": code,
        "message": message,
        "statusCode": status_code,
    }
    if ctx:
        body["context"] = ctx

    # Self-log — callers no longer need a separate log statement
    if status_code >= 500:
        log.bind(code=code, status=status_code, **context).error("error_response")
    else:
        log.bind(code=code, status=status_code, **context).warning("error_response")

    return JSONResponse(status_code=status_code, content=body)
