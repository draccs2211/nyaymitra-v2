"""
Logging setup: one place for format, level, request IDs and PII redaction.

    LOG_LEVEL   INFO (default) | DEBUG | WARNING ...
    LOG_FORMAT  json (default in production) | text

Every record gets `request_id` (set per HTTP request by the middleware in
main.py; asyncio.to_thread copies the context, so agent logs inside the graph
carry it too). Handlers redact emails / phones / Aadhaar / PAN.
"""
import contextvars
import json
import logging
import os
import sys
from datetime import datetime, timezone

from backend.utils.redact import RedactingFilter

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_STD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime", "request_id"}


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "msg": record.getMessage(),
        }
        # structured extras passed via logger.info(..., extra={...})
        for key, value in vars(record).items():
            if key not in _STD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: str = None, fmt: str = None) -> None:
    """Idempotent: replaces root handlers with one configured stream handler."""
    from backend.config import IS_PRODUCTION
    level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    fmt = (fmt or os.getenv("LOG_FORMAT", "json" if IS_PRODUCTION else "text")).lower()

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())
    handler.addFilter(RedactingFilter())
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"))

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)
    # quieten chatty libraries
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers", "mcp", "fastmcp"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))
