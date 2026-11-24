"""
Security tests (CLAUDE.md issue #10): secret key, CORS, rate limiting,
ownership checks, input limits, error hygiene, query logging.
SQLite + fake Redis + mocked LLMs (fixtures from test_approval_api).
Run: pytest tests/test_security.py -v
"""
import re

import pytest

from backend import config
from backend.database.models import Case, QueryLog
from backend.utils.rate_limit import SlidingWindowLimiter
from tests.test_approval import mocked_agents  # noqa: F401  (fixture)
from tests.test_approval_api import client  # noqa: F401  (fixture)


# ─── Secret key ───────────────────────────────────────────────────────────────

class TestSecretKey:
    @pytest.mark.parametrize("key", ["", "short", "your-secret-key-change-in-production", "x" * 31])
    def test_rejected(self, key):
        with pytest.raises(RuntimeError, match="SECRET_KEY"):
            config.require_secret_key(key)

    def test_accepted(self):
        assert config.require_secret_key("k" * 32) == "k" * 32

    def test_auth_uses_validated_key(self):
        from backend.utils import auth
        assert len(auth.SECRET_KEY) >= config.MIN_SECRET_KEY_LENGTH


# ─── CORS ─────────────────────────────────────────────────────────────────────

class TestCors:
    def _preflight(self, client, origin):
        return client.options("/api/query", headers={
            "Origin": origin, "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Authorization, Content-Type"})

    def test_allowed_origin(self, client):
        r = self._preflight(client, "http://localhost:5173")
        assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
        assert "access-control-allow-credentials" not in r.headers

    def test_other_origin_refused(self, client):
        r = self._preflight(client, "https://evil.example")
        assert "access-control-allow-origin" not in r.headers


# ─── Rate limiting ────────────────────────────────────────────────────────────

class TestRateLimit:
    def test_window(self):
        lim = SlidingWindowLimiter()
        assert all(lim.hit("k", 3, 60) == 0 for _ in range(3))
        assert lim.hit("k", 3, 60) > 0
        assert lim.hit("other", 3, 60) == 0           # keys independent

    def test_window_expires(self):
        lim = SlidingWindowLimiter()
        lim.hit("k", 1, 0.05)
        import time; time.sleep(0.06)
        assert lim.hit("k", 1, 0.05) == 0

    def test_login_brute_force_blocked(self, client):
        codes = [client.post("/api/auth/login", data={"username": "a@b.co", "password": "wrongpass"}).status_code
                 for _ in range(config.RATE_LIMIT_AUTH_PER_MIN + 1)]
        assert codes[:-1] == [401] * config.RATE_LIMIT_AUTH_PER_MIN
        assert codes[-1] == 429

    def test_429_has_retry_after(self, client):
        for _ in range(config.RATE_LIMIT_AUTH_PER_MIN):
            client.post("/api/auth/login", data={"username": "a@b.co", "password": "wrongpass"})
        r = client.post("/api/auth/login", data={"username": "a@b.co", "password": "wrongpass"})
        assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1

    def test_query_limited_per_user(self, client, mocked_agents, monkeypatch):
        import backend.main as main
        mocked_agents(["research"])
        calls = {"n": 0}

        async def fast_pipeline(**kw):
            calls["n"] += 1
            return {"final_response": "ok", "approval_status": "not_required", "run_id": "r"}
        monkeypatch.setattr(main, "run_pipeline", fast_pipeline)
        h = client.headers_for("associate")
        codes = [client.post("/api/query", json={"query": f"q{i}"}, headers=h).status_code
                 for i in range(config.RATE_LIMIT_QUERY_PER_MIN + 1)]
        assert codes[-1] == 429 and calls["n"] == config.RATE_LIMIT_QUERY_PER_MIN
        # another user is unaffected
        other = client.post("/api/query", json={"query": "q"}, headers=client.headers_for("senior"))
        assert other.status_code == 200


# ─── Ownership ────────────────────────────────────────────────────────────────

def _make_case(client, who) -> str:
    r = client.post("/api/cases", json={"title": "Deposit dispute"}, headers=client.headers_for(who))
    assert r.status_code == 200
    return r.json()["id"]


class TestOwnership:
    def test_query_with_foreign_case_id_404(self, client, mocked_agents):
        mocked_agents(["research"])
        case_id = _make_case(client, "senior")
        r = client.post("/api/query", json={"query": "q", "case_id": case_id},
                        headers=client.headers_for("associate"))
        assert r.status_code == 404

    def test_query_with_own_case_id_logged(self, client, mocked_agents):
        mocked_agents(["research"])
        case_id = _make_case(client, "associate")
        r = client.post("/api/query", json={"query": "what is 406", "case_id": case_id},
                        headers=client.headers_for("associate"))
        assert r.status_code == 200
        db = client.Session()
        try:   # background log_query used its own session and committed
            log = db.query(QueryLog).filter(QueryLog.case_id == case_id).one()
            assert log.user_id == client.ids["associate"]
        finally:
            db.close()

    def test_foreign_case_and_documents_404(self, client):
        case_id = _make_case(client, "senior")
        h = client.headers_for("associate")
        assert client.get(f"/api/cases/{case_id}", headers=h).status_code == 404
        assert client.get(f"/api/documents/{case_id}", headers=h).status_code == 404
        hs = client.headers_for("senior")
        assert client.get(f"/api/cases/{case_id}", headers=hs).status_code == 200
        assert client.get(f"/api/documents/{case_id}", headers=hs).status_code == 200


# ─── Input validation / error hygiene ─────────────────────────────────────────

class TestInputs:
    @pytest.mark.parametrize("password", ["short", "é" * 37])   # 74 bytes > bcrypt's 72
    def test_bad_passwords_rejected(self, client, password):
        r = client.post("/api/auth/register",
                        json={"name": "N", "email": "n@x.co", "password": password})
        assert r.status_code == 422

    def test_bad_email_rejected(self, client):
        r = client.post("/api/auth/register", json={"name": "N", "email": "nope", "password": "longenough"})
        assert r.status_code == 422

    def test_register_then_login_case_insensitive_email(self, client):
        r = client.post("/api/auth/register",
                        json={"name": "N", "email": "New.User@X.co", "password": "longenough1"})
        assert r.status_code == 200
        r = client.post("/api/auth/login", data={"username": "new.user@x.co", "password": "longenough1"})
        assert r.status_code == 200 and r.json()["access_token"]

    def test_query_too_long(self, client):
        r = client.post("/api/query", json={"query": "x" * 4001}, headers=client.headers_for("associate"))
        assert r.status_code == 422

    def test_pipeline_error_not_leaked(self, client, monkeypatch):
        import backend.main as main

        async def broken(**kw):
            raise RuntimeError("db password=hunter2 at 10.0.0.5")
        monkeypatch.setattr(main, "run_pipeline", broken)
        r = client.post("/api/query", json={"query": "q"}, headers=client.headers_for("associate"))
        assert r.status_code == 500
        assert "hunter2" not in r.text and "10.0.0.5" not in r.text

    def test_no_token_401(self, client):
        assert client.post("/api/query", json={"query": "q"}).status_code == 401


class TestPasswordHashing:
    def test_roundtrip(self):
        from backend.utils.auth import get_password_hash, verify_password
        h = get_password_hash("correct horse")
        assert h.startswith("$2b$")
        assert verify_password("correct horse", h)
        assert not verify_password("wrong horse", h)

    def test_passlib_era_hash_still_verifies(self):
        # standard $2b$ hash of "password123" (format passlib produced)
        import bcrypt
        from backend.utils.auth import verify_password
        legacy = bcrypt.hashpw(b"password123", bcrypt.gensalt(rounds=4)).decode()
        assert verify_password("password123", legacy)

    def test_overlong_and_malformed(self):
        from backend.utils.auth import get_password_hash, verify_password
        with pytest.raises(ValueError):
            get_password_hash("x" * 73)
        assert not verify_password("x" * 73, "$2b$12$abc")
        assert not verify_password("pw", "not-a-hash")



class TestVersioning:
    def test_prompt_change_changes_fingerprint(self, monkeypatch):
        from backend import versioning
        from backend.agents import analysis
        before = versioning.pipeline_fingerprint()
        monkeypatch.setattr(analysis, "SYSTEM_PROMPT", analysis.SYSTEM_PROMPT + " ")
        assert versioning.pipeline_fingerprint() != before

    def test_corpus_fingerprint_cached_and_resettable(self, monkeypatch):
        from backend import versioning
        from backend.rag.hybrid_search import hybrid_rag
        versioning.reset_corpus_fingerprint()
        first = versioning.corpus_fingerprint()
        monkeypatch.setattr(hybrid_rag.collection, "get", lambda **kw: {"ids": ["changed"]})
        assert versioning.corpus_fingerprint() == first          # cached within TTL
        versioning.reset_corpus_fingerprint()
        assert versioning.corpus_fingerprint() != first          # recomputed
        versioning.reset_corpus_fingerprint()

    def test_cache_version_format(self):
        from backend.versioning import cache_version
        assert re.match(r"^p[0-9a-f]{12}-c([0-9a-f]{12}|unknown)$", cache_version())

    @pytest.mark.parametrize("jur", ["UP", "IN-up", "IN-UPP", "US"])
    def test_invalid_jurisdiction_rejected(self, client, jur):
        r = client.post("/api/query", json={"query": "q", "jurisdiction": jur},
                        headers=client.headers_for("associate"))
        assert r.status_code == 422
