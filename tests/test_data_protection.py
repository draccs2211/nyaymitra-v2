"""
Data-protection tests (production plan, Phase 1 item 1): tracing defaults,
PII redaction, query-log minimisation, audit trail, retention purge.
No paid APIs; SQLite + fake Redis via test_approval_api fixtures.
Run: pytest tests/test_data_protection.py -v
"""
import logging
from datetime import datetime, timedelta

import pytest

from backend import config
from backend.database.models import AuditLog, QueryLog
from backend.maintenance import purge
from backend.utils.redact import RedactingFilter, keyed_hash, redact
from tests.test_approval import mocked_agents  # noqa: F401  (fixture)
from tests.test_approval_api import client  # noqa: F401  (fixture)


# ─── Tracing defaults ─────────────────────────────────────────────────────────

class TestTracingConfig:
    def test_off_unless_enabled(self):
        env = {"APP_ENV": "production"}
        assert config.configure_tracing(env) == {"enabled": False, "hide_payloads": False}
        assert "LANGSMITH_HIDE_INPUTS" not in env

    @pytest.mark.parametrize("flag", ["LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"])
    def test_production_hides_payloads(self, flag):
        env = {flag: "true"}                       # APP_ENV unset -> production
        assert config.configure_tracing(env)["hide_payloads"] is True
        assert env["LANGSMITH_HIDE_INPUTS"] == env["LANGSMITH_HIDE_OUTPUTS"] == "true"

    def test_development_keeps_payloads(self):
        env = {"APP_ENV": "development", "LANGSMITH_TRACING": "true"}
        assert config.configure_tracing(env)["hide_payloads"] is False

    def test_explicit_raw_opt_in(self):
        env = {"LANGSMITH_TRACING": "true", "LANGSMITH_ALLOW_RAW_TRACES": "true"}
        assert config.configure_tracing(env)["hide_payloads"] is False

    def test_app_env_defaults_to_production(self, monkeypatch):
        monkeypatch.delenv("APP_ENV", raising=False)
        import importlib
        try:
            assert importlib.reload(config).APP_ENV == "production"
        finally:
            monkeypatch.setenv("APP_ENV", "test")
            importlib.reload(config)


# ─── Redaction ────────────────────────────────────────────────────────────────

class TestRedaction:
    @pytest.mark.parametrize("raw,expected", [
        ("mail ramesh.kumar+x@gmail.com now", "mail [EMAIL] now"),
        ("Aadhaar 1234 5678 9012", "Aadhaar [AADHAAR]"),
        ("Aadhaar 123456789012.", "Aadhaar [AADHAAR]."),
        ("PAN ABCDE1234F", "PAN [PAN]"),
        ("call +91 9876543210", "call [PHONE]"),
        ("call 09876543210 or 9876543210", "call [PHONE] or [PHONE]"),
    ])
    def test_patterns(self, raw, expected):
        assert redact(raw) == expected

    @pytest.mark.parametrize("safe", [
        "Section 498A IPC", "Rs 50000 deposit", "case 2023/1234", "pincode 226001", "landline 0522-2611818",
    ])
    def test_legal_text_untouched(self, safe):
        assert redact(safe) == safe

    def test_logging_filter(self):
        rec = logging.LogRecord("t", logging.INFO, __file__, 1, "client %s phoned %s", ("a@b.co", "9876543210"), None)
        RedactingFilter().filter(rec)
        assert rec.getMessage() == "client [EMAIL] phoned [PHONE]"

    def test_keyed_hash(self):
        assert keyed_hash("q", "k1") == keyed_hash("q", "k1")
        assert keyed_hash("q", "k1") != keyed_hash("q", "k2")
        assert len(keyed_hash("q", "k1")) == 64


# ─── Query log minimisation + audit trail (API) ──────────────────────────────

def _rows(client, model, **filters):
    db = client.Session()
    try:
        return db.query(model).filter_by(**filters).all()
    finally:
        db.close()


class TestQueryLogAndAudit:
    QUERY = "Client Ramesh (9876543210) wants deposit back"

    def test_query_log_has_no_raw_text_by_default(self, client, mocked_agents):
        mocked_agents(["research"])
        r = client.post("/api/query", json={"query": self.QUERY}, headers=client.headers_for("associate"))
        assert r.status_code == 200
        log, = _rows(client, QueryLog, user_id=client.ids["associate"])
        assert log.query is None
        assert log.query_chars == len(self.QUERY) and len(log.query_hmac) == 64

    def test_raw_text_only_when_enabled(self, client, mocked_agents, monkeypatch):
        import backend.main as main
        monkeypatch.setattr(main, "STORE_QUERY_TEXT", True)
        mocked_agents(["research"])
        client.post("/api/query", json={"query": self.QUERY}, headers=client.headers_for("associate"))
        log, = _rows(client, QueryLog, user_id=client.ids["associate"])
        assert log.query == self.QUERY

    def test_query_audited_without_text(self, client, mocked_agents):
        mocked_agents(["research", "drafting"])
        r = client.post("/api/query", json={"query": self.QUERY}, headers=client.headers_for("associate"))
        entry, = _rows(client, AuditLog, action="query.run")
        assert entry.user_id == client.ids["associate"]
        assert entry.resource == f"run:{r.json()['run_id']}"
        assert entry.details["approval_status"] == "pending"
        assert "9876543210" not in str(entry.details) and "Ramesh" not in str(entry.details)

    def test_auth_events(self, client):
        client.post("/api/auth/register", json={"name": "N", "email": "n@x.co", "password": "longenough1"})
        client.post("/api/auth/login", data={"username": "n@x.co", "password": "wrong-pass"})
        client.post("/api/auth/login", data={"username": "n@x.co", "password": "longenough1"})
        actions = [a.action for a in _rows(client, AuditLog)]
        assert actions.count("auth.register") == 1
        assert actions.count("auth.login_failed") == 1 and actions.count("auth.login") == 1
        failed, = _rows(client, AuditLog, action="auth.login_failed")
        assert "n@x.co" not in str(failed.details)          # email stored only as HMAC

    def test_case_and_document_access_audited(self, client):
        h = client.headers_for("associate")
        case_id = client.post("/api/cases", json={"title": "T"}, headers=h).json()["id"]
        client.get(f"/api/cases/{case_id}", headers=h)
        client.get(f"/api/documents/{case_id}", headers=h)
        actions = {a.action for a in _rows(client, AuditLog, resource=f"case:{case_id}")}
        assert actions == {"case.create", "case.view", "documents.list"}

    def test_approval_decision_audited(self, client, mocked_agents):
        mocked_agents(["research", "drafting"])
        run_id = client.post("/api/query", json={"query": "draft notice"},
                             headers=client.headers_for("associate")).json()["run_id"]
        client.post(f"/api/approvals/{run_id}", json={"decision": "approved"},
                    headers=client.headers_for("senior"))
        entry, = _rows(client, AuditLog, action="approval.decide")
        assert entry.user_id == client.ids["senior"]
        assert entry.details["decision"] == "approved"
        assert entry.details["requester_id"] == client.ids["associate"]

class TestAuditHelper:
    def test_swallows_db_errors(self):
        from backend.utils.audit import audit

        class BadSession:
            def add(self, *_): raise RuntimeError("db down")
            def rollback(self): self.rolled = True
            def close(self): self.closed = True
        s = BadSession()
        audit(lambda: s, "x.y")        # must not raise
        assert s.rolled and s.closed


# ─── Retention purge ──────────────────────────────────────────────────────────

class TestPurge:
    def test_deletes_only_expired(self, client):
        now = datetime(2026, 11, 25)
        db = client.Session()
        uid = client.ids["associate"]
        db.add_all([
            QueryLog(user_id=uid, query_chars=1, created_at=now - timedelta(days=91)),
            QueryLog(user_id=uid, query_chars=1, created_at=now - timedelta(days=89)),
            AuditLog(action="old", created_at=now - timedelta(days=400)),
            AuditLog(action="new", created_at=now - timedelta(days=10)),
        ])
        db.commit(); db.close()

        dry = purge(client.Session, query_log_days=90, audit_days=365, dry_run=True, now=now)
        assert dry == {"query_logs": 1, "audit_logs": 1}
        assert len(_rows(client, QueryLog)) == 2                       # dry run kept everything

        assert purge(client.Session, query_log_days=90, audit_days=365, now=now) == dry
        assert len(_rows(client, QueryLog)) == 1
        assert [a.action for a in _rows(client, AuditLog)] == ["new"]
