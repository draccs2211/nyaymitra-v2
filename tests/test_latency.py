"""
Latency / progress tests (production plan, Phase 1 item 8): guardian and
orchestrator run in parallel; GET /api/runs/{run_id} progress polling.
Mocked LLMs only. Run: pytest tests/test_latency.py -v
"""
import asyncio
import json
import threading
import time
import uuid

import pytest

from tests.test_agent_outputs import analysis_json, mock_drafting
from tests.test_approval_api import client  # noqa: F401  (fixture)
from tests.test_llm import fake_client


# contains Section 406 so analysis_json()'s citation is grounded (else the citation guard forces review)
RESEARCH_DOCS = [{"id": f"d{i}", "content": f"Section 406 IPC criminal breach of trust ({i})",
                  "metadata": {"domain": "property"}} for i in range(5)]


def _mock(monkeypatch, *, slow=0.0, analysis_gate: threading.Event = None, actions=("research",),
          intervals: dict = None):
    from backend.agents import analysis, guardian, orchestrator, research

    def slow_reply(text, name=None):
        def reply(_):
            start = time.perf_counter()
            time.sleep(slow)
            if intervals is not None and name:
                intervals[name] = (start, time.perf_counter())
            return text
        return reply

    def gated_analysis(_):
        if analysis_gate is not None:
            assert analysis_gate.wait(10), "test gate never released"
        return analysis_json()

    monkeypatch.setattr(guardian, "OpenAI", fake_client(slow_reply('{"is_emergency": false}', "guardian")))
    monkeypatch.setattr(orchestrator, "OpenAI", fake_client(slow_reply(json.dumps(
        {"legal_domain": "property", "requested_actions": list(actions)}), "orchestrator")))
    monkeypatch.setattr(analysis, "OpenAI", fake_client(gated_analysis))
    # 5 docs -> confidence high enough that research-only runs complete without review
    monkeypatch.setattr(research.hybrid_rag, "search", lambda q, d=None, **kw: RESEARCH_DOCS)
    mock_drafting(monkeypatch)


@pytest.fixture(autouse=True)
def _fresh_llm_clients():
    from backend import llm
    llm.reset_clients()
    yield
    llm.reset_clients()


class TestParallel:
    def test_guardian_and_orchestrator_overlap(self, monkeypatch):
        """Their LLM calls must overlap in time (wall-clock totals include warm-up noise)."""
        from backend.pipeline import run_pipeline
        spans = {}
        _mock(monkeypatch, slow=0.3, intervals=spans)
        r = asyncio.run(run_pipeline("q", "s", "u"))
        assert r["guardian_status"] == "checked" and r["legal_domain"] == "property"
        (g0, g1), (o0, o1) = spans["guardian"], spans["orchestrator"]
        assert g0 < o1 and o0 < g1, f"no overlap: guardian {spans['guardian']} orchestrator {spans['orchestrator']}"

    def test_tokens_from_both_branches_summed(self, monkeypatch):
        from backend.pipeline import run_pipeline
        _mock(monkeypatch)
        r = asyncio.run(run_pipeline("q", "s", "u"))
        assert r["tokens_used"] == 15 * 3       # guardian + orchestrator + analysis (fake: 15 each)


class TestProgress:
    def test_stage_visible_mid_run(self, monkeypatch):
        from backend.pipeline import run_pipeline, run_progress
        gate = threading.Event()
        _mock(monkeypatch, analysis_gate=gate)
        run_id = str(uuid.uuid4())
        worker = threading.Thread(target=lambda: asyncio.run(run_pipeline("q", "s", "u", run_id=run_id)))
        worker.start()
        try:
            stage, deadline = None, time.time() + 10
            while time.time() < deadline:
                p = asyncio.run(run_progress(run_id))
                if p and p["stage"] == "analysing":
                    stage = p
                    break
                time.sleep(0.05)
            assert stage and stage["status"] == "running", stage
        finally:
            gate.set()
            worker.join(10)
        assert asyncio.run(run_progress(run_id))["status"] == "completed"

    def test_awaiting_approval(self, monkeypatch):
        from backend.pipeline import run_pipeline, run_progress
        _mock(monkeypatch, actions=("research", "drafting"))
        r = asyncio.run(run_pipeline("draft", "s", "u"))
        p = asyncio.run(run_progress(r["run_id"]))
        assert p["status"] == "awaiting_approval" and p["next"] == ["human_approval"]

    def test_unknown_run(self):
        from backend.pipeline import run_progress
        assert asyncio.run(run_progress("nope")) is None


class TestProgressApi:
    def test_client_run_id_and_ownership(self, client, monkeypatch):
        _mock(monkeypatch)
        run_id = str(uuid.uuid4())
        r = client.post("/api/query", json={"query": "q", "run_id": run_id}, headers=client.headers_for("associate"))
        assert r.status_code == 200 and r.json()["run_id"] == run_id
        mine = client.get(f"/api/runs/{run_id}", headers=client.headers_for("associate"))
        assert mine.status_code == 200 and mine.json()["status"] == "completed"
        assert "user_id" not in mine.json()
        assert client.get(f"/api/runs/{run_id}", headers=client.headers_for("senior")).status_code == 404

    def test_duplicate_run_id_rejected(self, client, monkeypatch):
        _mock(monkeypatch)
        run_id = str(uuid.uuid4())
        h = client.headers_for("associate")
        assert client.post("/api/query", json={"query": "q", "run_id": run_id}, headers=h).status_code == 200
        assert client.post("/api/query", json={"query": "q2", "run_id": run_id}, headers=h).status_code == 409

    @pytest.mark.parametrize("bad", ["123", "../../etc", "x" * 36])
    def test_invalid_run_id(self, client, bad):
        r = client.post("/api/query", json={"query": "q", "run_id": bad}, headers=client.headers_for("associate"))
        assert r.status_code == 422
