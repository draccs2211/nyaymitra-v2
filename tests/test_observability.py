"""
Observability tests (production plan, Phase 1 item 5): JSON logs, request IDs,
redaction on handlers, access log, /health and /ready, no client text at INFO.
Run: pytest tests/test_observability.py -v
"""
import asyncio
import json
import logging

import pytest

from backend.logging_config import JsonFormatter, RequestIdFilter, request_id_var, setup_logging
from backend.utils.redact import RedactingFilter
from tests.test_agent_outputs import fake_openai
from tests.test_approval_api import client  # noqa: F401  (fixture)
from tests.test_pipeline import make_state


def _record(msg, *args, **extra):
    rec = logging.LogRecord("nyaymitra.test", logging.INFO, __file__, 1, msg, args, None)
    for k, v in extra.items():
        setattr(rec, k, v)
    return rec


class TestFormatting:
    def test_json_has_request_id_extras_and_redaction(self):
        token = request_id_var.set("req-123")
        try:
            rec = _record("client %s", "a@b.co", run_id="r1", tokens_used=42)
            RequestIdFilter().filter(rec)
            RedactingFilter().filter(rec)
            out = json.loads(JsonFormatter().format(rec))
        finally:
            request_id_var.reset(token)
        assert out["request_id"] == "req-123" and out["msg"] == "client [EMAIL]"
        assert out["run_id"] == "r1" and out["tokens_used"] == 42
        assert out["level"] == "INFO" and out["logger"] == "nyaymitra.test"

    def test_setup_is_idempotent(self):
        root = logging.getLogger()
        saved = list(root.handlers), root.level
        try:
            setup_logging(fmt="json")
            setup_logging(fmt="json")
            assert len(root.handlers) == 1
            h = root.handlers[0]
            assert any(isinstance(f, RedactingFilter) for f in h.filters)
            assert any(isinstance(f, RequestIdFilter) for f in h.filters)
        finally:
            root.handlers[:] = saved[0]
            root.setLevel(saved[1])

    def test_request_id_reaches_worker_threads(self):
        """The graph runs in asyncio.to_thread; its logs must keep the request ID."""
        async def main():
            request_id_var.set("rid-thread")
            return await asyncio.to_thread(lambda: request_id_var.get())
        assert asyncio.run(main()) == "rid-thread"


class TestRequestId:
    def test_generated_when_missing(self, client):
        rid = client.get("/health").headers["x-request-id"]
        assert len(rid) == 32

    def test_valid_incoming_kept(self, client):
        assert client.get("/health", headers={"X-Request-ID": "lb-abc.123"}).headers["x-request-id"] == "lb-abc.123"

    @pytest.mark.parametrize("bad", ["x" * 65, "has space", "<script>"])
    def test_invalid_incoming_replaced(self, client, bad):
        assert client.get("/health", headers={"X-Request-ID": bad}).headers["x-request-id"] != bad

    def test_access_log(self, client, caplog):
        with caplog.at_level(logging.INFO, logger="nyaymitra.access"):
            client.get("/health")
        rec = [r for r in caplog.records if r.name == "nyaymitra.access"][-1]
        assert rec.status == 200 and rec.path == "/health" and rec.duration_ms >= 0


class TestHealth:
    @pytest.fixture(autouse=True)
    def _openai_key_configured(self, monkeypatch):
        # don't depend on the developer's .env (CI has no key)
        import backend.config as config
        monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-test-dummy")

    def test_liveness(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_ready(self, client):
        r = client.get("/ready")
        body = r.json()
        assert r.status_code == 200, body
        assert body["status"] in ("ready", "degraded")
        assert body["checks"]["database"]["ok"] and body["checks"]["corpus"]["documents"] > 0

    def test_not_ready_without_openai_key(self, client, monkeypatch):
        import backend.config as config
        monkeypatch.setattr(config, "OPENAI_API_KEY", "")
        r = client.get("/ready")
        assert r.status_code == 503 and r.json()["status"] == "not_ready"

    def test_not_ready_when_db_down(self, client):
        import backend.main as main
        from backend.database.connection import get_db

        class DownSession:
            def execute(self, *_):
                raise ConnectionError("db down")
        main.app.dependency_overrides[get_db] = lambda: DownSession()
        r = client.get("/ready")
        assert r.status_code == 503
        assert r.json()["checks"]["database"] == {"ok": False, "error": "ConnectionError"}

    def test_redis_down_is_degraded_not_failing(self, client, monkeypatch):
        from backend.utils import cache
        monkeypatch.setattr(cache, "redis_available", lambda: False)
        r = client.get("/ready")
        assert r.status_code == 200 and r.json()["status"] == "degraded"


class TestNoClientTextAtInfo:
    def test_orchestrator_reasoning_only_at_debug(self, monkeypatch, caplog):
        from backend.agents import orchestrator
        monkeypatch.setattr(orchestrator, "OpenAI", fake_openai(json.dumps({
            "legal_domain": "property", "reasoning": "Client Ramesh Kumar lost deposit at 12 MG Road"})))
        with caplog.at_level(logging.INFO):
            orchestrator.orchestrator_agent(make_state(user_query="q"))
        assert not any("Ramesh" in r.getMessage() for r in caplog.records)
        with caplog.at_level(logging.DEBUG):
            orchestrator.orchestrator_agent(make_state(user_query="q"))
        assert any("Ramesh" in r.getMessage() and r.levelno == logging.DEBUG for r in caplog.records)
