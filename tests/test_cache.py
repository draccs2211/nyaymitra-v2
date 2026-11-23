"""
Cache isolation tests — query cache and sessions must never cross users.
Run: pytest tests/test_cache.py -v
"""
import pytest
from backend.utils import cache


class FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value

    def delete(self, key):
        self.store.pop(key, None)


@pytest.fixture(autouse=True)
def fake_redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(cache, "redis_client", fake)
    return fake


class TestQueryCacheIsolation:
    def test_same_query_not_shared_across_users(self):
        cache.set_cached_query("landlord deposit", {"final_response": "A"}, scope="user-a")
        assert cache.get_cached_query("landlord deposit", scope="user-a") == {"final_response": "A"}
        assert cache.get_cached_query("landlord deposit", scope="user-b") is None

    def test_language_is_part_of_key(self):
        cache.set_cached_query("q", {"x": 1}, scope="u", language="hi")
        assert cache.get_cached_query("q", scope="u", language="en") is None

    def test_scope_required(self):
        with pytest.raises(ValueError):
            cache.get_cached_query("q", scope="")
        with pytest.raises(ValueError):
            cache.set_cached_query("q", {}, scope="")

    def test_key_is_sha256_and_versioned(self):
        key = cache.make_key("query", "u", "q")
        assert key.startswith(f"nyaymitra:{cache.CACHE_VERSION}:query:")
        assert len(key.rsplit(":", 1)[1]) == 64

    def test_parts_are_unambiguous(self):
        assert cache.make_key("p", "ab", "c") != cache.make_key("p", "a", "bc")


class TestSessionIsolation:
    def test_session_not_readable_by_other_user(self):
        cache.store_session("sess-1", {"history": ["secret"]}, user_id="user-a")
        assert cache.get_session("sess-1", user_id="user-a") == {"history": ["secret"]}
        assert cache.get_session("sess-1", user_id="user-b") is None

    def test_clear_only_affects_owner(self):
        cache.store_session("sess-1", {"h": 1}, user_id="user-a")
        cache.clear_session("sess-1", user_id="user-b")
        assert cache.get_session("sess-1", user_id="user-a") == {"h": 1}

    def test_user_id_required(self):
        with pytest.raises(ValueError):
            cache.get_session("sess-1", user_id="")
