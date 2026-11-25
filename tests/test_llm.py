"""
LLM client hardening tests (production plan, Phase 1 item 2): timeout/retries,
shared client, token accounting through the graph, daily budget.
No paid APIs (fake clients; real OpenAI class only constructed, never called).
Run: pytest tests/test_llm.py -v
"""
import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from backend import llm
from backend.config import LLM_MAX_RETRIES, LLM_TIMEOUT_SECONDS
from backend.database.models import QueryLog
from tests.test_agent_outputs import analysis_json
from tests.test_approval_api import client  # noqa: F401  (fixture)
from tests.test_pipeline import make_state


def fake_client(reply, prompt_tokens=10, completion_tokens=5, record=None):
    """OpenAI-like class returning `reply` (str or fn(kwargs)->str) with usage."""
    class _Client:
        def __init__(self, **kw):
            if record is not None:
                record.append(kw)

            def create(**call):
                text = reply(call) if callable(reply) else reply
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                    usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
                )
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
    return _Client


@pytest.fixture(autouse=True)
def _fresh_clients():
    llm.reset_clients()
    yield
    llm.reset_clients()


class TestClient:
    def test_timeout_and_retries_passed(self):
        made = []
        llm.get_client(fake_client("{}", record=made))
        assert made[0]["timeout"] == LLM_TIMEOUT_SECONDS == 30
        assert made[0]["max_retries"] == LLM_MAX_RETRIES == 2

    def test_real_openai_client_configured(self):
        from openai import OpenAI
        c = llm.get_client(OpenAI)          # constructed only; no request is sent
        assert c.timeout == LLM_TIMEOUT_SECONDS and c.max_retries == LLM_MAX_RETRIES

    def test_client_shared_per_class(self):
        made = []
        cls = fake_client("{}", record=made)
        assert llm.get_client(cls) is llm.get_client(cls)
        assert len(made) == 1

    def test_failed_construction_not_cached(self):
        calls = {"n": 0}

        class Flaky:
            def __init__(self, **kw):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("transient")
        with pytest.raises(RuntimeError):
            llm.get_client(Flaky)
        assert isinstance(llm.get_client(Flaky), Flaky)

    def test_chat_json_returns_usage(self):
        data, usage = llm.chat_json(fake_client('{"a": 1}', 7, 3), "s", "u", max_tokens=10)
        assert data == {"a": 1} and usage.total == 10

    def test_missing_usage_is_zero(self):
        from tests.test_agent_outputs import fake_openai     # fake without usage
        _, usage = llm.chat_json(fake_openai("{}"), "s", "u", max_tokens=10)
        assert usage.total == 0

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            llm.chat_json(fake_client("not json"), "s", "u", max_tokens=10)


class TestTokenAccounting:
    @pytest.mark.asyncio
    async def test_tokens_summed_across_agents(self, monkeypatch):
        from backend.agents import analysis, drafting, guardian, orchestrator, research
        from backend.pipeline import run_pipeline
        monkeypatch.setattr(guardian, "OpenAI", fake_client('{"is_emergency": false}', 10, 1))      # 11
        monkeypatch.setattr(orchestrator, "OpenAI", fake_client(json.dumps(
            {"legal_domain": "property", "requested_actions": ["research", "drafting"]}), 20, 2))   # 22
        monkeypatch.setattr(analysis, "OpenAI", fake_client(analysis_json(), 30, 3))             # 33

        def draft_reply(call):
            if call["messages"][0]["content"] == drafting.EXTRACTION_PROMPT:
                return json.dumps({"document_type": "legal_notice"})
            return json.dumps({"english": "LEGAL NOTICE " * 5, "hindi": "कानूनी नोटिस " * 5})
        monkeypatch.setattr(drafting, "OpenAI", fake_client(draft_reply, 40, 4))                 # 44 x2
        monkeypatch.setattr(research.hybrid_rag, "search", lambda q, d=None, **kw: [])

        r = await run_pipeline("draft notice", "s", "u")
        assert r["tokens_used"] == 11 + 22 + 33 + 44 * 2

    def test_failed_call_counts_zero(self):
        from backend.agents.guardian import guardian_agent
        out = guardian_agent(make_state(user_query="q"))       # OpenAI blocked by conftest
        assert out.get("tokens_used", 0) == 0


class TestDailyBudget:
    def _log_tokens(self, client, who, tokens, days_ago=0):
        db = client.Session()
        db.add(QueryLog(user_id=client.ids[who], query_chars=1, tokens_used=tokens,
                        created_at=datetime.utcnow() - timedelta(days=days_ago)))
        db.commit()
        db.close()

    def _query(self, client, who):
        return client.post("/api/query", json={"query": "q"}, headers=client.headers_for(who))

    @pytest.fixture
    def fast_pipeline(self, monkeypatch):
        import backend.main as main

        async def run(**kw):
            return {"final_response": "ok", "approval_status": "not_required", "run_id": "r",
                    "tokens_used": 123}
        monkeypatch.setattr(main, "run_pipeline", run)
        monkeypatch.setattr(main, "DAILY_TOKEN_BUDGET", 1000)

    def test_firm_budget_shared_by_colleagues(self, client, fast_pipeline):
        self._log_tokens(client, "senior", 1000)            # same firm as associate
        r = self._query(client, "associate")
        assert r.status_code == 429 and "limit" in r.json()["detail"]
        assert self._query(client, "other_senior").status_code == 200   # other firm unaffected

    def test_yesterday_not_counted(self, client, fast_pipeline):
        self._log_tokens(client, "associate", 5000, days_ago=1)
        assert self._query(client, "associate").status_code == 200

    def test_solo_user_counted_alone(self, client, fast_pipeline):
        self._log_tokens(client, "associate", 5000)
        assert self._query(client, "solo").status_code == 200

    def test_tokens_logged_and_returned(self, client, fast_pipeline):
        r = self._query(client, "solo")
        assert r.json()["tokens_used"] == 123
        db = client.Session()
        try:
            assert db.query(QueryLog).filter_by(user_id=client.ids["solo"]).one().tokens_used == 123
        finally:
            db.close()

    def test_zero_budget_means_unlimited(self, client, fast_pipeline, monkeypatch):
        import backend.main as main
        monkeypatch.setattr(main, "DAILY_TOKEN_BUDGET", 0)
        self._log_tokens(client, "associate", 10**9)
        assert self._query(client, "associate").status_code == 200
