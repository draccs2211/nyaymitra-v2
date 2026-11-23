from backend.database.connection import get_redis
from backend.config import QUERY_CACHE_TTL, EMBEDDING_CACHE_TTL, LLM_CACHE_TTL
import hashlib
import json
from typing import Optional


redis_client = get_redis()

# Bump to invalidate all cached entries (e.g. after prompt/corpus changes)
CACHE_VERSION = "v2"


def make_key(prefix: str, *parts: str) -> str:
    """SHA-256 key over all parts; unit separator prevents ambiguous joins."""
    data = "".join(str(p) for p in parts)
    digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
    return f"nyaymitra:{CACHE_VERSION}:{prefix}:{digest}"


def _query_key(query: str, scope: str, language: str, domain: str) -> str:
    if not scope:
        # Never share query results across tenants/users
        raise ValueError("cache scope (user id) is required")
    return make_key("query", scope, language, domain, query)


def get_cached_query(query: str, *, scope: str, language: str = "", domain: str = "") -> Optional[dict]:
    try:
        cached = redis_client.get(_query_key(query, scope, language, domain))
        if cached:
            return json.loads(cached)
    except ValueError:
        raise
    except Exception:
        pass
    return None


def set_cached_query(query: str, response: dict, *, scope: str, language: str = "",
                     domain: str = "", ttl: int = QUERY_CACHE_TTL):
    key = _query_key(query, scope, language, domain)
    try:
        redis_client.setex(key, ttl, json.dumps(response, default=str))
    except Exception:
        pass


def get_cached_embedding(text: str) -> Optional[list]:
    key = make_key("embed", text)
    try:
        cached = redis_client.get(key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass
    return None


def set_cached_embedding(text: str, embedding: list, ttl: int = EMBEDDING_CACHE_TTL):
    key = make_key("embed", text)
    try:
        redis_client.setex(key, ttl, json.dumps(embedding))
    except Exception:
        pass


def get_cached_llm(prompt: str) -> Optional[str]:
    key = make_key("llm", prompt)
    try:
        cached = redis_client.get(key)
        return cached
    except Exception:
        pass
    return None


def set_cached_llm(prompt: str, response: str, ttl: int = LLM_CACHE_TTL):
    key = make_key("llm", prompt)
    try:
        redis_client.setex(key, ttl, response)
    except Exception:
        pass


def _session_key(session_id: str, user_id: str) -> str:
    if not user_id:
        raise ValueError("user_id is required for session access")
    return make_key("session", user_id, session_id)


def store_session(session_id: str, data: dict, *, user_id: str, ttl: int = 3600):
    key = _session_key(session_id, user_id)
    try:
        redis_client.setex(key, ttl, json.dumps(data, default=str))
    except Exception:
        pass


def get_session(session_id: str, *, user_id: str) -> Optional[dict]:
    key = _session_key(session_id, user_id)
    try:
        cached = redis_client.get(key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass
    return None


def clear_session(session_id: str, *, user_id: str):
    key = _session_key(session_id, user_id)
    try:
        redis_client.delete(key)
    except Exception:
        pass
