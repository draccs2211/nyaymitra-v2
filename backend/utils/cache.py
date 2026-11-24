from backend.database.connection import get_redis
from backend.config import QUERY_CACHE_TTL, EMBEDDING_CACHE_TTL, LLM_CACHE_TTL
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
import hashlib
import json
import time
from typing import Optional


redis_client = get_redis()

# Bump to invalidate all cached entries (e.g. after prompt/corpus changes)
CACHE_VERSION = "v2"


def make_key(prefix: str, *parts: str) -> str:
    """SHA-256 key over all parts; unit separator prevents ambiguous joins."""
    data = "".join(str(p) for p in parts)
    digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
    return f"nyaymitra:{CACHE_VERSION}:{prefix}:{digest}"


# ─── Redis access with circuit breaker ────────────────────────────────────────
# When Redis is down every call would block for the connect timeout. After a
# connection failure, skip Redis entirely for REDIS_RETRY_AFTER seconds.
REDIS_RETRY_AFTER = 30.0
_redis_down_until = 0.0


def redis_available() -> bool:
    return time.monotonic() >= _redis_down_until


def _redis(op: str, *args):
    """Run redis_client.<op>(*args). Returns None if Redis is down or errors."""
    global _redis_down_until
    if not redis_available():
        return None
    try:
        return getattr(redis_client, op)(*args)
    except (RedisConnectionError, RedisTimeoutError):
        _redis_down_until = time.monotonic() + REDIS_RETRY_AFTER
        return None
    except Exception:
        return None


def _get_json(key: str):
    cached = _redis("get", key)
    if not cached:
        return None
    try:
        return json.loads(cached)
    except (TypeError, ValueError):
        return None


def _set_json(key: str, value, ttl: int) -> None:
    _redis("setex", key, ttl, json.dumps(value, default=str))


def cache_get_json(key: str):
    """Safe JSON get: returns None on miss or if Redis is unavailable."""
    return _get_json(key)


def cache_set_json(key: str, value, ttl: int) -> None:
    """Safe JSON set: silently skips if Redis is unavailable."""
    _set_json(key, value, ttl)


def _query_key(query: str, scope: str, language: str, domain: str) -> str:
    if not scope:
        # Never share query results across tenants/users
        raise ValueError("cache scope (user id) is required")
    return make_key("query", scope, language, domain, query)


def get_cached_query(query: str, *, scope: str, language: str = "", domain: str = "") -> Optional[dict]:
    return _get_json(_query_key(query, scope, language, domain))


def set_cached_query(query: str, response: dict, *, scope: str, language: str = "",
                     domain: str = "", ttl: int = QUERY_CACHE_TTL):
    _set_json(_query_key(query, scope, language, domain), response, ttl)


def get_cached_embedding(text: str) -> Optional[list]:
    return _get_json(make_key("embed", text))


def set_cached_embedding(text: str, embedding: list, ttl: int = EMBEDDING_CACHE_TTL):
    _set_json(make_key("embed", text), embedding, ttl)


def get_cached_llm(prompt: str) -> Optional[str]:
    return _redis("get", make_key("llm", prompt))


def set_cached_llm(prompt: str, response: str, ttl: int = LLM_CACHE_TTL):
    _redis("setex", make_key("llm", prompt), ttl, response)


def _session_key(session_id: str, user_id: str) -> str:
    if not user_id:
        raise ValueError("user_id is required for session access")
    return make_key("session", user_id, session_id)


def store_session(session_id: str, data: dict, *, user_id: str, ttl: int = 3600):
    _set_json(_session_key(session_id, user_id), data, ttl)


def get_session(session_id: str, *, user_id: str) -> Optional[dict]:
    return _get_json(_session_key(session_id, user_id))


def clear_session(session_id: str, *, user_id: str):
    _redis("delete", _session_key(session_id, user_id))
