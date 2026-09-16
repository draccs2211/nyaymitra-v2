from backend.database.connection import get_redis
from backend.config import QUERY_CACHE_TTL, EMBEDDING_CACHE_TTL, LLM_CACHE_TTL
import hashlib
import json
import pickle
from typing import Any, Optional


redis_client = get_redis()


def make_key(prefix: str, data: str) -> str:
    return f"nyaymitra:{prefix}:{hashlib.md5(data.encode()).hexdigest()}"


def get_cached_query(query: str, domain: str = "") -> Optional[dict]:
    key = make_key("query", f"{query}{domain}")
    try:
        cached = redis_client.get(key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass
    return None


def set_cached_query(query: str, domain: str, response: dict, ttl: int = QUERY_CACHE_TTL):
    key = make_key("query", f"{query}{domain}")
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


def store_session(session_id: str, data: dict, ttl: int = 3600):
    key = make_key("session", session_id)
    try:
        redis_client.setex(key, ttl, json.dumps(data, default=str))
    except Exception:
        pass


def get_session(session_id: str) -> Optional[dict]:
    key = make_key("session", session_id)
    try:
        cached = redis_client.get(key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass
    return None


def clear_session(session_id: str):
    key = make_key("session", session_id)
    try:
        redis_client.delete(key)
    except Exception:
        pass
