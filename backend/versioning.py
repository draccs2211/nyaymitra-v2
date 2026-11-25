"""
Version fingerprints for cache keys (CLAUDE.md issue #11).

A cached answer is only valid for the prompts/model and the corpus that
produced it. Both fingerprints are derived automatically, so changing a
prompt or reloading the corpus invalidates old cached answers without any
manual version bump.
"""
import hashlib
import threading
import time

from backend.config import LLM_MODEL

_CORPUS_TTL = 60.0

_lock = threading.Lock()
_corpus_cache = {"value": None, "at": 0.0}


def _sha(*parts: str, n: int = 12) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:n]


def pipeline_fingerprint() -> str:
    """Hash of every agent prompt + model name."""
    from backend.agents import analysis, drafting, guardian, orchestrator
    return _sha(
        LLM_MODEL,
        guardian.GUARDIAN_PROMPT,
        orchestrator.CLASSIFICATION_PROMPT,
        analysis.SYSTEM_PROMPT,
        drafting.EXTRACTION_PROMPT,
        drafting.DRAFTING_PROMPT,
    )


def corpus_fingerprint() -> str:
    """Hash of the sorted canonical doc IDs (content hashes). Cached briefly."""
    now = time.monotonic()
    with _lock:
        if _corpus_cache["value"] and now - _corpus_cache["at"] < _CORPUS_TTL:
            return _corpus_cache["value"]
    from backend.rag.hybrid_search import hybrid_rag
    try:
        ids = sorted(hybrid_rag.collection.get(include=[])["ids"])
        value = _sha(*ids)
    except Exception:
        value = "unknown"
    with _lock:
        _corpus_cache.update(value=value, at=now)
    return value


def reset_corpus_fingerprint() -> None:
    with _lock:
        _corpus_cache.update(value=None, at=0.0)


def cache_version() -> str:
    return f"p{pipeline_fingerprint()}-c{corpus_fingerprint()}"
