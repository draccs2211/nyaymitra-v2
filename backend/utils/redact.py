"""
PII redaction for logs (DPDP Act 2023: minimise personal data outside the
system of record). Pattern masking of identifiers only — not classification.

Masks: email addresses, Aadhaar numbers, PAN, Indian mobile numbers.
"""
import hashlib
import hmac
import logging
import re

_PATTERNS = [
    ("[EMAIL]", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    # Aadhaar before phone: 12 digits, optionally grouped 4-4-4
    ("[AADHAAR]", re.compile(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)")),
    ("[PAN]", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    # optional +91 / leading 0, then a 10-digit mobile starting 6-9 (landlines like 0522-... don't match)
    ("[PHONE]", re.compile(r"(?<!\d)(?:\+91[\s-]?|0)?[6-9]\d{9}(?!\d)")),
]


def redact(text: str) -> str:
    if not text:
        return text
    for token, pattern in _PATTERNS:
        text = pattern.sub(token, text)
    return text


def keyed_hash(value: str, key: str) -> str:
    """HMAC-SHA256: lets us correlate identical values without storing them
    (a plain hash of short text could be reversed by guessing)."""
    return hmac.new(key.encode("utf-8"), (value or "").encode("utf-8"), hashlib.sha256).hexdigest()


class RedactingFilter(logging.Filter):
    """Logging filter that masks PII in the fully formatted message."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        redacted = redact(message)
        if redacted != message:
            record.msg, record.args = redacted, ()
        return True


def install_redaction(logger: logging.Logger = None) -> None:
    """Attach the filter to every handler of `logger` (root by default).
    Handler-level so it also covers records propagated from child loggers."""
    logger = logger or logging.getLogger()
    for handler in logger.handlers:
        if not any(isinstance(f, RedactingFilter) for f in handler.filters):
            handler.addFilter(RedactingFilter())
